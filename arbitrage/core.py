from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .data import DataLoader
from .backtest import Backtester, Trade, summarize_trades, kpis
from .config import BASE_CCY, PARAMS

# ---------- small utils ----------

def to_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().fillna(0.0)

# ---------- FX normalize ----------

class FXNormalizer:
    def __init__(self, base_ccy: str, fx_map: Dict[str, pd.DataFrame]):
        self.base = base_ccy
        self.fx = fx_map  # e.g., {"EURUSD": df}

    def convert(self, px: pd.Series, quote_ccy: str) -> pd.Series:
        if quote_ccy == self.base:
            return px
        key = f"{quote_ccy}{self.base}"
        if key in self.fx:
            rate = self.fx[key]["close"].reindex(px.index).ffill()
            return px * rate
        inv_key = f"{self.base}{quote_ccy}"
        if inv_key in self.fx:
            rate = self.fx[inv_key]["close"].reindex(px.index).ffill()
            return px / rate
        raise KeyError(
            f"No FX series to convert {quote_ccy}->{self.base}. "
            f"Available pairs: {list(self.fx.keys())}"
        )

# ---------- pair analyzer ----------

@dataclass
class PairData:
    name: str
    us_close: pd.Series  # original quote ccy
    eu_close: pd.Series  # original quote ccy
    us_ccy: str
    eu_ccy: str

class PairAnalyzer:
    def __init__(self, fx: FXNormalizer, lookback: int = 60):
        self.fx = fx
        self.lookback = lookback

    def to_base_and_align(self, pair: PairData) -> pd.DataFrame:
        us_base = self.fx.convert(pair.us_close, pair.us_ccy)
        eu_base = self.fx.convert(pair.eu_close, pair.eu_ccy)
        df = pd.concat({"US": us_base, "EU": eu_base}, axis=1).dropna()
        return df

    def build_ratio_df(self, pair: PairData) -> pd.DataFrame:
        df = self.to_base_and_align(pair)
        ratio = df["US"] / df["EU"]
        mu = ratio.rolling(self.lookback).mean()
        sigma = ratio.rolling(self.lookback).std(ddof=0)
        z = (ratio - mu) / sigma
        out = pd.DataFrame({
            "US": df["US"],
            "EU": df["EU"],
            "ratio": ratio,
            "mu": mu,
            "sigma": sigma,
            "z": z,
        }).dropna()
        r_us, r_eu = to_returns(out["US"]), to_returns(out["EU"])
        out["roll_corr"] = r_us.rolling(self.lookback).corr(r_eu)
        return out.dropna()

    @staticmethod
    def adf_pvalue(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 20:
            return 1.0
        try:
            return float(adfuller(x.values, autolag="AIC")[1])
        except Exception:
            return 1.0

# ---------- signals ----------

@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: int   # +1 long US/short EU; -1 short US/long EU
    z_at_entry: float

class SignalEngine:
    def __init__(self, params: Dict):
        self.p = params

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        sigs: List[Signal] = []
        in_pos = 0
        for t, row in df.iterrows():
            z = row["z"]
            corr_ok = (row.get("roll_corr", 1.0) or 0.0) >= self.p["min_corr"]
            if in_pos == 0 and corr_ok:
                if z >= self.p["entry_z"]:
                    sigs.append(Signal(t, -1, float(z)))  # short US / long EU
                    in_pos = -1
                elif z <= -self.p["entry_z"]:
                    sigs.append(Signal(t, +1, float(z)))  # long US / short EU
                    in_pos = +1
            elif in_pos != 0:
                if abs(z) <= self.p["exit_z"]:
                    in_pos = 0
        return sigs

# ---------- orchestrator ----------

class Explorer:
    def __init__(self, loader: DataLoader, params: Dict):
        self.loader = loader
        self.params = params

    def run_pair(self, pair_conf: Dict) -> Dict:
        # 1) Load data
        us = self.loader.load_etf_daily(pair_conf["us"]["ticker"])
        eu = self.loader.load_etf_daily(pair_conf["eu"]["ticker"])

        fx_needed: set[str] = set()
        for leg in ("us", "eu"):
            ccy = pair_conf[leg]["ccy"]
            if ccy != BASE_CCY:
                fx_needed.add(f"{ccy}{BASE_CCY}")
                fx_needed.add(f"{BASE_CCY}{ccy}")

        fx_map: Dict[str, pd.DataFrame] = {}
        for k in fx_needed:
            try:
                fx_map[k] = self.loader.load_fx_daily(k)
            except Exception:
                pass

        # 2) FX normalize
        fx_norm = FXNormalizer(BASE_CCY, fx_map)
        pair = PairData(
            name=pair_conf["name"],
            us_close=us["close"],
            eu_close=eu["close"],
            us_ccy=pair_conf["us"]["ccy"],
            eu_ccy=pair_conf["eu"]["ccy"],
        )

        # 3) Build ratio, stationarity/corr checks
        analyzer = PairAnalyzer(fx_norm, self.params["lookback"])
        ratio_df = analyzer.build_ratio_df(pair)

        log_ratio = np.log(ratio_df["ratio"])
        adf_pval = analyzer.adf_pvalue(log_ratio)

        if self.params.get("use_adf_filter", False):
            if adf_pval > 0.10:
                return {"pair": pair.name, "skipped": True, "reason": f"ADF p={adf_pval:.3f}"}

        # 4) Signals + Backtest
        sig_engine = SignalEngine(self.params)
        sigs = sig_engine.generate(ratio_df)

        bt = Backtester(self.params)
        equity, trades, market_time = bt.run(ratio_df, sigs)

        # 5) Analytics
        trade_df = summarize_trades(trades, self.params["position_usd"])
        metrics = kpis(trades, self.params["position_usd"], equity, market_time)
        metrics.update({
            "adf_pvalue": float(adf_pval),
            "avg_roll_corr": float(ratio_df.get("roll_corr", pd.Series(dtype=float)).mean()) if "roll_corr" in ratio_df else 0.0,
            "corr_below_min_pct": float((ratio_df.get("roll_corr", pd.Series(dtype=float)) < self.params["min_corr"]).mean() * 100) if "roll_corr" in ratio_df else 0.0,
        })

        return {
            "pair": pair.name,
            "ratio_df": ratio_df,
            "signals": sigs,
            "equity": equity,
            "trades": trade_df,
            "metrics": metrics,
        }

    def run_all(self, pairs: List[Dict]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for pc in pairs:
            try:
                res = self.run_pair(pc)
                out[pc["name"]] = res
            except Exception as e:
                out[pc["name"]] = {"error": str(e)}
        return out
