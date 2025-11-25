"""
Streamlit UI for Global ETF Cross-Market Arbitrage Explorer.

This file reuses the reusable arbitrage core/backtest modules to keep the
interactive dashboard in sync with the backtest/CLI pipeline.
"""

import io
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from arbitrage.backtest import Backtester, kpis, summarize_trades
from arbitrage.config import BASE_CCY, PAIR_CONFIG, PARAMS
from arbitrage.core import FXNormalizer, PairAnalyzer, PairData, SignalEngine
from arbitrage.data import DataLoader


DEFAULT_PARAMS = {
    **PARAMS,
    "lookback": 60,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "min_corr": 0.8,
}

# ====================== STREAMLIT DATA LOADER ======================

class StreamlitLoader(DataLoader):
    """DataLoader wrapper that prefers uploaded CSVs and falls back to ./data.

    The class caches results via ``st.cache_data`` to avoid repeated parsing
    when the user tweaks parameters.
    """

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
        # enforce a single 'close' column
        if "close" not in df.columns:
            raise ValueError("Uploaded CSV must contain a 'close' column")
        return df[["close"]]

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        raw = self.uploads.get(ticker)
        path = os.path.join(self.root, f"{ticker}_daily.csv")
        return self._read_cached(f"etf::{ticker}", raw, path)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        raw = self.uploads.get(pair)
        path = os.path.join(self.root, f"{pair}_daily.csv")
        return self._read_cached(f"fx::{pair}", raw, path)


def build_fx_map(loader: DataLoader) -> Dict[str, pd.DataFrame]:
    fx_map: Dict[str, pd.DataFrame] = {}
    for k in {"EURUSD", "USDGBP", "GBPUSD", "EURGBP"}:
        try:
            fx_map[k] = loader.load_fx_daily(k)
        except Exception:
            # optional – only complain later if conversion fails
            pass
    return fx_map


def run_pair(pair_conf: Dict, loader: DataLoader, params: Dict) -> tuple[PairData, pd.DataFrame, List]:
    """Load, FX-normalize, build ratio DF and generate signals for a pair."""

    fx_map = build_fx_map(loader)
    fx_norm = FXNormalizer(BASE_CCY, fx_map)

    us_df = loader.load_etf_daily(pair_conf["us"]["ticker"])
    eu_df = loader.load_etf_daily(pair_conf["eu"]["ticker"])

    pair = PairData(
        name=pair_conf["name"],
        us_close=us_df["close"],
        eu_close=eu_df["close"],
        us_ccy=pair_conf["us"]["ccy"],
        eu_ccy=pair_conf["eu"]["ccy"],
    )

    analyzer = PairAnalyzer(fx_norm, lookback=params["lookback"])
    ratio_df = analyzer.build_ratio_df(pair)

    sig_engine = SignalEngine(params)
    sigs = sig_engine.generate(ratio_df)
    return pair, ratio_df, sigs

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
    key_us = f"us_{pair_conf['name']}"
    key_eu = f"eu_{pair_conf['name']}"
    up_us = st.file_uploader(f"{pair_conf['us']['ticker']} (ETF daily CSV)", type=["csv"], key=key_us)
    up_eu = st.file_uploader(f"{pair_conf['eu']['ticker']} (ETF daily CSV)", type=["csv"], key=key_eu)

    fx_uploads = []
    for leg in ("us", "eu"):
        ccy = pair_conf[leg]["ccy"]
        if ccy != BASE_CCY:
            fx_pair = f"{ccy}{BASE_CCY}"
            key_fx = f"fx_{pair_conf['name']}_{fx_pair}"
            fx_uploads.append(
                (
                    fx_pair,
                    st.file_uploader(f"{fx_pair} (FX daily CSV)", type=["csv"], key=key_fx),
                )
            )

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
    pair_conf["us"]["ticker"]: up_us.getvalue() if up_us else None,
    pair_conf["eu"]["ticker"]: up_eu.getvalue() if up_eu else None,
}
for fx_pair, fx_file in fx_uploads:
    if fx_file is not None:
        uploads_map[fx_pair] = fx_file.getvalue()

loader = StreamlitLoader(root="./data", uploads=uploads_map)
try:
    pair, ratio_df, sigs = run_pair(pair_conf, loader, params)
except KeyError as e:
    st.error(f"FX conversion missing: {e}. Upload appropriate FX CSV (e.g., EURUSD) or place it in ./data.")
    st.stop()

bt = Backtester(params)
equity_df, trades, market_time = bt.run(ratio_df, sigs)
trade_df = summarize_trades(trades, params["position_usd"]) if trades else pd.DataFrame()
metrics = kpis(trades, params["position_usd"], equity_df, market_time) if trades is not None else {"trades": 0}
metrics.update({
    "adf_pvalue": PairAnalyzer.adf_pvalue(np.log(ratio_df["ratio"]).dropna()),
    "avg_roll_corr": float(ratio_df.get("roll_corr", pd.Series(dtype=float)).mean()) if "roll_corr" in ratio_df else 0.0,
    "corr_below_min_pct": float((ratio_df.get("roll_corr", pd.Series(dtype=float)) < min_corr).mean() * 100)
    if "roll_corr" in ratio_df
    else 0.0,
})

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
    st.download_button(
        "Download ratio CSV",
        data=ratio_df.to_csv().encode(),
        file_name=f"{pair.name}_ratio.csv",
        mime="text/csv",
    )

    st.subheader("Equity Curve (cum PnL)")
    if not equity_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(equity_df.index, equity_df["cum_pnl"])
        ax2.set_title("Cumulative PnL (USD)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, clear_figure=True)
        st.download_button(
            "Download equity CSV",
            data=equity_df.to_csv().encode(),
            file_name=f"{pair.name}_equity.csv",
            mime="text/csv",
        )
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
    c1.metric("Avg hold (days)", f"{m.get('avg_hold_days', 0.0):.2f}")
    c2.metric("Median hold", f"{m.get('median_hold_days', 0.0):.2f}")
    st.metric("Equity Sharpe", f"{m.get('equity_sharpe', 0.0):.2f}")
    st.metric("Max Drawdown (USD)", f"{m.get('max_drawdown_usd', 0.0):,.0f}")
    st.metric("Time in market", f"{m.get('time_in_market_pct', 0.0)*100:.1f}%")

    st.divider()
    st.subheader("Stationarity & Corr Filters")
    st.metric("ADF p-value (log ratio)", f"{m.get('adf_pvalue', 0.0):.3f}")
    st.metric("Avg rolling corr", f"{m.get('avg_roll_corr', 0.0):.2f}")
    st.metric("% corr < min", f"{m.get('corr_below_min_pct', 0.0):.1f}%")

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
    st.download_button(
        "Download trades CSV",
        data=trade_df.to_csv().encode(),
        file_name=f"{pair.name}_trades.csv",
        mime="text/csv",
    )
else:
    st.write("No trades yet.")

st.divider()
st.subheader("Cross-pair z-score heatmap")
st.caption("Latest mispricing snapshot across configured pairs (z of US/EU price ratio).")

z_cols = []
for pc in PAIR_CONFIG:
    try:
        _, z_df, _ = run_pair(pc, loader, params)
        z_cols.append(z_df[["z"]].rename(columns={"z": pc["name"]}))
    except Exception:
        continue

if z_cols:
    zscores = pd.concat(z_cols, axis=1).dropna(how="all")
    latest = zscores.tail(1).T.rename(columns={zscores.index[-1]: "latest_z"})

    # Optional: attach a few high-level stats per pair
    summary_rows = []
    for pc in PAIR_CONFIG:
        pair_name = pc["name"]
        # reuse the same config + params on the loader
        try:
            pair_obj, ratio_tmp, sig_tmp = run_pair(pc, loader, params)
            bt_tmp = Backtester(params)
            eq_tmp, trades_tmp, mt_tmp = bt_tmp.run(ratio_tmp, sig_tmp)
            k = kpis(trades_tmp, params["position_usd"], eq_tmp, mt_tmp)
            summary_rows.append({
                "pair": pair_name,
                "trades": k.get("trades", 0),
                "hit_rate": k.get("hit_rate", 0.0),
                "sharpe_like": k.get("sharpe_like", 0.0),
                "equity_sharpe": k.get("equity_sharpe", 0.0),
                "latest_z": float(latest.loc[pair_name, "latest_z"]) if pair_name in latest.index else np.nan,
            })
        except Exception:
            continue

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index("pair")
        st.subheader("Multi-pair summary")
        st.dataframe(summary_df)

    st.dataframe(latest)

    fig4, ax4 = plt.subplots(figsize=(8, 3))
    cax = ax4.imshow(zscores.T, aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax4.set_yticks(range(len(zscores.columns)))
    ax4.set_yticklabels(zscores.columns)
    ax4.set_xticks(range(0, len(zscores.index), max(len(zscores.index)//10, 1)))
    ax4.set_xticklabels(zscores.index.strftime("%Y-%m-%d").to_list()[::max(len(zscores.index)//10, 1)], rotation=45, ha="right")
    ax4.set_title("Z-score heatmap (time vs pair)")
    fig4.colorbar(cax, label="z")
    st.pyplot(fig4, clear_figure=True)
else:
    st.write("Unable to build heatmap – check that all CSVs exist in ./data.")
