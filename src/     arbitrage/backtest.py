from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd

def annualize_rate(bps_per_year: float, days: float) -> float:
    return (bps_per_year / 10_000.0) * (days / 365.0)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int              # +1 long US/short EU; -1 short US/long EU
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

    def run(self, df: pd.DataFrame, sigs: List["Signal"]) -> Tuple[pd.DataFrame, List[Trade]]:
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

            # open position
            if position == 0 and next_sig is not None and t >= next_sig.timestamp:
                entry_px = self._fill_price(df["ratio"], t, latency)
                position = next_sig.direction
                entry_t = t
                entry_ratio = entry_px * (1.0 + slip * np.sign(position))
                z_entry = next_sig.z_at_entry
                next_sig = next(sig_iter, None)

            # manage exits
            if position != 0:
                holding_days = (t - entry_t).days if isinstance(t, pd.Timestamp) else i
                exit_due_to_time = holding_days >= self.p["max_holding_days"]
                exit_due_to_mean = abs(row["z"]) <= self.p["exit_z"]
                if exit_due_to_time or exit_due_to_mean:
                    exit_px = self._fill_price(df["ratio"], t, latency)
                    exit_px = exit_px * (1.0 - slip * np.sign(position))
                    ratio_ret = (exit_px / entry_ratio - 1.0) * position
                    pnl = gross * ratio_ret
                    # round-trip fees + slip (both legs)
                    pnl -= gross * 2 * fee
                    pnl -= gross * 2 * slip
                    # borrow on short leg
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

# ---------- Analytics helpers ----------

def summarize_trades(trades: List[Trade], position_usd: float) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["ret_bps"] = (df["pnl_usd"] / position_usd) * 10_000
    return df

def kpis(trades: List[Trade], position_usd: float) -> dict:
    if not trades:
        return {"trades": 0, "hit_rate": 0.0, "avg_ret_bps": 0.0, "sharpe_like": 0.0, "max_drawdown_usd": 0.0}
    df = summarize_trades(trades, position_usd)
    wins = (df["pnl_usd"] > 0).sum()
    total = len(df)
    hit = wins / total if total else 0.0
    avg = df["ret_bps"].mean()
    std = df["ret_bps"].std(ddof=0)
    sharpe = (avg / std * math.sqrt(252)) if std and total > 5 else 0.0
    dd = (df["pnl_usd"].cumsum().cummax() - df["pnl_usd"].cumsum()).max()
    return {
        "trades": int(total),
        "hit_rate": float(hit),
        "avg_ret_bps": float(avg),
        "sharpe_like": float(sharpe),
        "max_drawdown_usd": float(dd),
    }

