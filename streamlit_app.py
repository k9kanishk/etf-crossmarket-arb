# streamlit_app.py
# Self-contained Streamlit UI for Global ETF Cross-Market Arbitrage Explorer

import os
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ====================== CONFIG ======================

BASE_CCY = "USD"

PAIR_CONFIG = [
    {
        "name": "S&P500_US_vs_LSE",
        "us": {"ticker": "SPY",     "ccy": "USD", "venue": "NYSE"},
        "eu": {"ticker": "CSPX.L",  "ccy": "USD", "venue": "LSE"},
        "fx": "EURUSD",
    },
    {
        "name": "EM_US_vs_EU",
        "us": {"ticker": "VWO",     "ccy": "USD", "venue": "NYSE"},
        "eu": {"ticker": "IEMM.MI", "ccy": "EUR", "venue": "BorsaItaliana"},
        "fx": "EURUSD",
    },
]

DEFAULT_PARAMS = {
    "lookback": 60,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "max_holding_days": 5,
    "use_adf_filter": False,
    "min_corr": 0.8,
    "txn_fee_bps": 0.5,
    "slippage_bps": 1.0,
    "borrow_bps": 30.0,
    "latency_bars": 1,
    "position_usd": 1_000_000,
}

# ====================== CORE LOGIC ======================

def to_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().fillna(0.0)

def annualize_rate(bps_per_year: float, days: float) -> float:
    return (bps_per_year / 10_000.0) * (days / 365.0)

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
        raise KeyError(f"No FX series to convert {quote_ccy}->{self.base}")

@dataclass
class PairData:
    name: str
    us_close: pd.Series
    eu_close: pd.Series
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

@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: int  # +1 long US/short EU; -1 short US/long EU
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
            elif in_pos != 0 and abs(z) <= self.p["exit_z"]:
                in_pos = 0
        return sigs

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_ratio: float
    exit_ratio: float
    z_entry: float
    pnl_usd: float
    holding_days: float

class Backtester:
    def __init__(self, params: Dict):
        self.p = params

    def _fill_price(self, s: pd.Series, t: pd.Timestamp, latency: int) -> float:
        idx = s.index.get_indexer([t], method="bfill")[0]
        idx = min(idx + latency, len(s) - 1)
        return float(s.iloc[idx])

    def run(self, df: pd.DataFrame, sigs: List[Signal]) -> Tuple[pd.DataFrame, List[Trade]]:
        trades: List[Trade] = []
        equity = []

        position = 0
        entry_t = None
        entry_ratio = None
        z_entry = None

        gross = self.p["position_usd"]
        fee = self.p["txn_fee_bps"] / 10_000.0
        slip = self.p["slippage_bps"] / 10_000.0
        latency = self.p["latency_bars"]

        sig_iter = iter(sigs)
        next_sig = next(sig_iter, None)

        for i, (t, row) in enumerate(df.iterrows()):
            ratio = float(row["ratio"])

            if position == 0 and next_sig is not None and t >= next_sig.timestamp:
                entry_px = self._fill_price(df["ratio"], t, latency)
                position = next_sig.direction
                entry_t = t
                entry_ratio = entry_px * (1.0 + slip * np.sign(position))
                z_entry = next_sig.z_at_entry
                next_sig = next(sig_iter, None)

            if position != 0:
                holding_days = (t - entry_t).days if isinstance(t, pd.Timestamp) else i
                exit_due_to_time = holding_days >= self.p["max_holding_days"]
                exit_due_to_mean = abs(row["z"]) <= self.p["exit_z"]
                if exit_due_to_time or exit_due_to_mean:
                    exit_px = self._fill_price(df["ratio"], t, latency)
                    exit_px = exit_px * (1.0 - slip * np.sign(position))
                    ratio_ret = (exit_px / entry_ratio - 1.0) * position
                    pnl = gross * ratio_ret
                    pnl -= gross * 2 * fee
                    pnl -= gross * 2 * slip
                    pnl -= gross * annualize_rate(self.p["borrow_bps"], max(holding_days, 1))

                    trades.append(Trade(
                        entry_time=entry_t,
                        exit_time=t,
                        direction=position,
                        entry_ratio=float(entry_ratio),
                        exit_ratio=float(exit_px),
                        z_entry=float(z_entry),
                        pnl_usd=float(pnl),
                        holding_days=float(holding_days),
                    ))
                    position = 0
                    entry_t = None
                    entry_ratio = None
                    z_entry = None

            equity.append((t, trades[-1].pnl_usd if trades else 0.0))

        eq = pd.DataFrame(equity, columns=["time", "last_trade_pnl"]).set_index("time")
        if trades:
            eq["cum_pnl"] = np.cumsum([t.pnl_usd for t in trades] + [0]*(len(eq)-len(trades)))
        else:
            eq["cum_pnl"] = 0.0
        return eq, trades

def summarize_trades(trades: List[Trade], position_usd: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["ret_bps"] = (df["pnl_usd"] / position_usd) * 10_000
    return df

def kpis(trades: List[Trade], position_usd: float) -> dict:
    if not trades:
        return {"trades": 0, "hit_rate": 0.0, "avg_ret_bps": 0.0,
                "sharpe_like": 0.0, "max_drawdown_usd": 0.0}
    df = summarize_trades(trades, position_usd)
    wins = (df["pnl_usd"] > 0).sum()
    total = len(df)
    hit = wins / total if total else 0.0
    avg = df["ret_bps"].mean()
    std = df["ret_bps"].std(ddof=0)
    sharpe = (avg / std * np.sqrt(252)) if std and total > 5 else 0.0
    dd = (df["pnl_usd"].cumsum().cummax() - df["pnl_usd"].cumsum()).max()
    return {
        "trades": int(total),
        "hit_rate": float(hit),
        "avg_ret_bps": float(avg),
        "sharpe_like": float(sharpe),
        "max_drawdown_usd": float(dd),
    }

# ====================== DATA LOADING FOR STREAMLIT ======================

class StreamlitLoader:
    def __init__(self, root: str = "./data", uploads: Dict[str, Optional[io.BytesIO]] = None):
        self.root = root
        self.uploads = uploads or {}

    @st.cache_data(show_spinner=False)
    def _read_cached(_self, key: str, raw: bytes | None, path: str | None) -> pd.DataFrame:
        if raw is not None:
            df = pd.read_csv(io.BytesIO(raw), parse_dates=[0])
        else:
            df = pd.read_csv(path, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()
        if "close" not in df.columns:
            df.columns = [c.lower() for c in df.columns]
        return df

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        up = self.uploads.get(ticker)
        path = os.path.join(self.root, f"{ticker}_daily.csv")
        raw = up.read() if up else None
        return self._read_cached(f"etf::{ticker}", raw, path)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        up = self.uploads.get(pair)
        path = os.path.join(self.root, f"{pair}_daily.csv")
        raw = up.read() if up else None
        return self._read_cached(f"fx::{pair}", raw, path)

# ====================== STREAMLIT UI ======================

st.set_page_config(page_title="ETF Cross-Market Arbitrage", layout="wide")
st.title("Global ETF Correlation & Cross-Market Arbitrage Explorer")
st.caption("Normalize across FX, detect mispricings via z-scores, and backtest mean-reversion trades.")

with st.sidebar:
    st.header("Controls")

    pair_names = [pc["name"] for pc in PAIR_CONFIG]
    selected_pair_name = st.selectbox("Pair", options=pair_names, index=0)
    pair_conf = next(pc for pc in PAIR_CONFIG if pc["name"] == selected_pair_name)

    st.subheader("Strategy Parameters")
    lookback = st.slider("Lookback (days)", 20, 200, int(DEFAULT_PARAMS["lookback"]))
    entry_z = st.slider("Entry |z|", 1.0, 4.0, float(DEFAULT_PARAMS["entry_z"]), 0.1)
    exit_z = st.slider("Exit |z|", 0.1, 2.0, float(DEFAULT_PARAMS["exit_z"]), 0.1)
    max_hold = st.slider("Max holding (days)", 1, 20, int(DEFAULT_PARAMS["max_holding_days"]))
    min_corr = st.slider("Min rolling corr", 0.0, 1.0, float(DEFAULT_PARAMS["min_corr"]), 0.05)
    position_usd = st.number_input("Position per leg (USD)", 50_000, 5_000_000,
                                   int(DEFAULT_PARAMS["position_usd"]), 50_000)
    txn_fee_bps = st.number_input("Txn fee (bps per side)", 0.0, 10.0,
                                  float(DEFAULT_PARAMS["txn_fee_bps"]), 0.1)
    slippage_bps = st.number_input("Slippage (bps per side)", 0.0, 20.0,
                                   float(DEFAULT_PARAMS["slippage_bps"]), 0.1)
    borrow_bps = st.number_input("Borrow cost (annual bps)", 0.0, 1000.0,
                                 float(DEFAULT_PARAMS["borrow_bps"]), 5.0)
    latency_bars = st.number_input("Latency bars", 0, 5, int(DEFAULT_PARAMS["latency_bars"]))

    st.subheader("Upload (optional)")
    st.caption("Provide custom CSVs to override ./data files.")
    up_us = st.file_uploader(f"{pair_conf['us']['ticker']} (ETF daily CSV)", type=["csv"], key="us")
    up_eu = st.file_uploader(f"{pair_conf['eu']['ticker']} (ETF daily CSV)", type=["csv"], key="eu")
    up_eurusd = st.file_uploader("EURUSD (FX daily CSV)", type=["csv"], key="eurusd")

params = dict(
    lookback=lookback,
    entry_z=entry_z,
    exit_z=exit_z,
    max_holding_days=max_hold,
    use_adf_filter=False,
    min_corr=min_corr,
    txn_fee_bps=txn_fee_bps,
    slippage_bps=slippage_bps,
    borrow_bps=borrow_bps,
    latency_bars=latency_bars,
    position_usd=position_usd,
)

uploads_map = {
    pair_conf["us"]["ticker"]: up_us,
    pair_conf["eu"]["ticker"]: up_eu,
}
if up_eurusd is not None:
    uploads_map["EURUSD"] = up_eurusd

loader = StreamlitLoader(root="./data", uploads=uploads_map)

# Load FX data
fx_map = {}
for k in {"EURUSD", "USDGBP", "GBPUSD", "EURGBP"}:
    try:
        fx_map[k] = loader.load_fx_daily(k)
    except Exception:
        pass

fx_norm = FXNormalizer(BASE_CCY, fx_map)

# Load ETF closes
us_df = loader.load_etf_daily(pair_conf["us"]["ticker"])
eu_df = loader.load_etf_daily(pair_conf["eu"]["ticker"])

pair = PairData(
    name=pair_conf["name"],
    us_close=us_df["close"],
    eu_close=eu_df["close"],
    us_ccy=pair_conf["us"]["ccy"],
    eu_ccy=pair_conf["eu"]["ccy"],
)

analyzer = PairAnalyzer(fx_norm, lookback=lookback)
try:
    ratio_df = analyzer.build_ratio_df(pair)
except KeyError as e:
    st.error(f"FX conversion missing: {e}. Upload appropriate FX CSV (e.g., EURUSD) or place it in ./data.")
    st.stop()

sig_engine = SignalEngine(params)
sigs = sig_engine.generate(ratio_df)

bt = Backtester(params)
equity_df, trades = bt.run(ratio_df, sigs)
trade_df = summarize_trades(trades, params["position_usd"]) if trades else pd.DataFrame()
metrics = kpis(trades, params["position_usd"]) if trades else {"trades": 0}

# ====================== PLOTS ======================

left, right = st.columns([2, 1])

with left:
    st.subheader("Ratio & Bands")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ratio_df.index, ratio_df["ratio"], label="ratio")
    ax.plot(ratio_df.index, ratio_df["mu"], label="mean")
    ax.plot(ratio_df.index, ratio_df["mu"] + 2 * ratio_df["sigma"], label="+2σ")
    ax.plot(ratio_df.index, ratio_df["mu"] - 2 * ratio_df["sigma"], label="-2σ")
    ax.legend(loc="best")
    ax.set_title(f"{pair.name} — ratio in {BASE_CCY}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Equity Curve (cum PnL)")
    if not equity_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(equity_df.index, equity_df["cum_pnl"])
        ax2.set_title("Cumulative PnL (USD)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)
    else:
        st.info("No trades were generated with current parameters.")

with right:
    st.subheader("Latest Metrics")
    m = metrics or {}
    c1, c2 = st.columns(2)
    c1.metric("Trades", m.get("trades", 0))
    c2.metric("Hit Rate", f"{m.get('hit_rate', 0.0)*100:.1f}%")
    c1.metric("Avg Ret (bps)", f"{m.get('avg_ret_bps', 0.0):.2f}")
    c2.metric("Sharpe-like", f"{m.get('sharpe_like', 0.0):.2f}")
    st.metric("Max Drawdown (USD)", f"{m.get('max_drawdown_usd', 0.0):,.0f}")

    st.divider()
    st.caption("Rolling correlation (returns)")
    if "roll_corr" in ratio_df.columns:
        fig3, ax3 = plt.subplots(figsize=(6, 2.5))
        ax3.plot(ratio_df.index, ratio_df["roll_corr"])
        ax3.axhline(min_corr, linestyle="--")
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, clear_figure=True)

st.divider()
st.subheader("Trade Log")
if not trade_df.empty:
    st.dataframe(trade_df)
else:
    st.write("No trades yet.")
