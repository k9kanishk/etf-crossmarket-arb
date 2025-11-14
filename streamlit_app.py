import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from arbitrage.config import PARAMS as DEFAULT_PARAMS, PAIR_CONFIG, BASE_CCY
from arbitrage.core import Explorer, PairAnalyzer, FXNormalizer, PairData, SignalEngine
from arbitrage.data import DataLoader, DemoCSVLoader
from arbitrage.backtest import Backtester, summarize_trades, kpis

st.set_page_config(page_title="ETF Cross‑Market Arbitrage", layout="wide")
st.title("Global ETF Correlation & Cross‑Market Arbitrage Explorer")
st.caption("Normalize across FX, detect mispricings via z‑scores, and backtest mean‑reversion signals.")

# ---------------------------- Sidebar Controls --------------------------------
with st.sidebar:
    st.header("Controls")

    pair_names = [pc["name"] for pc in PAIR_CONFIG]
    selected_pair_name = st.selectbox("Pair", options=pair_names, index=0)
    pair_conf = next(pc for pc in PAIR_CONFIG if pc["name"] == selected_pair_name)

    st.subheader("Strategy Parameters")
    lookback = st.slider("Lookback (days)", min_value=20, max_value=200, value=int(DEFAULT_PARAMS["lookback"]))
    entry_z = st.slider("Entry |z|", min_value=1.0, max_value=4.0, step=0.1, value=float(DEFAULT_PARAMS["entry_z"]))
    exit_z = st.slider("Exit |z|", min_value=0.1, max_value=2.0, step=0.1, value=float(DEFAULT_PARAMS["exit_z"]))
    max_hold = st.slider("Max holding (days)", min_value=1, max_value=20, value=int(DEFAULT_PARAMS["max_holding_days"]))
    min_corr = st.slider("Min rolling corr", min_value=0.0, max_value=1.0, step=0.05, value=float(DEFAULT_PARAMS["min_corr"]))
    position_usd = st.number_input("Position per leg (USD)", min_value=50_000, max_value=5_000_000, step=50_000, value=int(DEFAULT_PARAMS["position_usd"]))
    txn_fee_bps = st.number_input("Txn fee (bps per side)", min_value=0.0, max_value=10.0, step=0.1, value=float(DEFAULT_PARAMS["txn_fee_bps"]))
    slippage_bps = st.number_input("Slippage (bps per side)", min_value=0.0, max_value=20.0, step=0.1, value=float(DEFAULT_PARAMS["slippage_bps"]))
    borrow_bps = st.number_input("Borrow cost (annual bps)", min_value=0.0, max_value=1000.0, step=5.0, value=float(DEFAULT_PARAMS["borrow_bps"]))
    latency_bars = st.number_input("Latency bars", min_value=0, max_value=5, step=1, value=int(DEFAULT_PARAMS["latency_bars"]))

    st.subheader("Upload (optional)")
    st.caption("Provide custom CSVs to override ./data files.")
    up_us = st.file_uploader(f"{pair_conf['us']['ticker']} (ETF daily CSV)", type=["csv"], key="us")
    up_eu = st.file_uploader(f"{pair_conf['eu']['ticker']} (ETF daily CSV)", type=["csv"], key="eu")
    # Only needed if currencies differ from base
    up_eurusd = st.file_uploader("EURUSD (FX daily CSV)", type=["csv"], key="eurusd")

# Build params dict
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

# ------------------------------ Loaders ---------------------------------------
class StreamlitLoader(DataLoader):
    def __init__(self, pair_conf, uploads: dict | None = None, root: str = "./data"):
        self.pair_conf = pair_conf
        self.uploads = uploads or {}
        self.root = root

    @st.cache_data(show_spinner=False)
    def _read_cached(_, key: str, raw: bytes | None, path: str | None) -> pd.DataFrame:
        if raw is not None:
            return pd.read_csv(io.BytesIO(raw), parse_dates=[0]).set_index(lambda df: pd.to_datetime(df.index)).sort_index()
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
        df = self._read_cached(f"etf::{ticker}", raw, path)
        return df

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        key_map = {"EURUSD": "EURUSD"}
        up_key = key_map.get(pair)
        up = self.uploads.get(up_key)
        path = os.path.join(self.root, f"{pair}_daily.csv")
        raw = up.read() if up else None
        df = self._read_cached(f"fx::{pair}", raw, path)
        return df

# Map uploads by ticker string expected by loader
uploads_map = {
    pair_conf['us']['ticker']: up_us,
    pair_conf['eu']['ticker']: up_eu,
}
if up_eurusd is not None:
    uploads_map["EURUSD"] = up_eurusd

loader = StreamlitLoader(pair_conf, uploads=uploads_map)

# ------------------------------ Analysis --------------------------------------
# Prepare FX map for diagnostics
fx_needed = {"EURUSD", "USDGBP", "GBPUSD", "EURGBP"}
fx_map = {}
for k in fx_needed:
    try:
        fx_map[k] = loader.load_fx_daily(k)
    except Exception:
        pass

fx_norm = FXNormalizer(BASE_CCY, fx_map)

# Load ETF closes
us_df = loader.load_etf_daily(pair_conf["us"]["ticker"])  # must include 'close'
eu_df = loader.load_etf_daily(pair_conf["eu"]["ticker"])  # must include 'close'

# Build pair and ratio
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

# ------------------------------ Signals & Backtest ----------------------------
sig_engine = SignalEngine(params)
sigs = sig_engine.generate(ratio_df)

bt = Backtester(params)
equity_df, trades = bt.run(ratio_df, sigs)
trade_df = summarize_trades(trades, params["position_usd"]) if trades else pd.DataFrame()
metrics = kpis(trades, params["position_usd"]) if trades else {"trades": 0}

# ------------------------------ Layout ----------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Ratio & Bands")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ratio_df.index, ratio_df["ratio"], label="ratio")
    ax.plot(ratio_df.index, ratio_df["mu"], label="mean")
    ax.plot(ratio_df.index, ratio_df["mu"] + 2*ratio_df["sigma"], label="+2σ")
    ax.plot(ratio_df.index, ratio_df["mu"] - 2*ratio_df["sigma"], label="-2σ")
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
    c2.metric("Sharpe‑like", f"{m.get('sharpe_like', 0.0):.2f}")
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

# ------------------------------ Heatmap (multi‑pair) --------------------------
st.divider()
st.subheader("Signal Heatmap — latest z across pairs")
heat_cols = []
heat_vals = []
for pc in PAIR_CONFIG:
    try:
        ldr = StreamlitLoader(pc, uploads={})
        us = ldr.load_etf_daily(pc["us"]["ticker"]) 
        eu = ldr.load_etf_daily(pc["eu"]["ticker"]) 
        # FXs (best‑effort)
        fmap = {}
        for k in {"EURUSD", "USDGBP", "GBPUSD", "EURGBP"}:
            try:
                fmap[k] = ldr.load_fx_daily(k)
            except Exception:
                pass
        fxn = FXNormalizer(BASE_CCY, fmap)
        pr = PairData(name=pc["name"], us_close=us["close"], eu_close=eu["close"], us_ccy=pc["us"]["ccy"], eu_ccy=pc["eu"]["ccy"])
        an = PairAnalyzer(fxn, lookback=lookback)
        dfp = an.build_ratio_df(pr)
        if not dfp.empty:
            heat_cols.append(pc["name"])
            heat_vals.append(float(dfp["z"].iloc[-1]))
    except Exception:
        continue

if heat_cols:
    hm = pd.DataFrame([heat_vals], columns=heat_cols, index=["latest_z"])
    st.dataframe(hm.style.background_gradient(axis=1, cmap="RdYlGn_r"))
else:
    st.write("Heatmap unavailable (insufficient data across pairs).")
