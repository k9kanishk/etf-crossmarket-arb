# Global ETF Cross-Market Arbitrage Explorer

A compact research toolkit for spotting cross-venue ETF mispricings, normalizing across FX, and backtesting mean-reversion trades. The project ships with demo CSVs for two U.S./European ETF pairs (SPY vs. CSPX.L and VWO vs. IEMM.MI) plus EURUSD FX to align everything into USD.

## Features
- FX normalization into a base currency with support for common crosses (EURUSD/GBPUSD/EURGBP/USDGBP).
- Rolling ratio, mean, and z-score calculation for each U.S./EU ETF pair.
- Signal engine that flags |z| breakouts beyond configurable thresholds and backtests mean-reversion exits or max-hold stops.
- Streamlit dashboard with ratio bands, rolling correlation, trade log, and a cross-pair heatmap of z-scores.
- CLI backtester that can run against demo CSVs or a Tiingo-powered loader for live data.

## Getting started
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI backtest**
   ```bash
   python run.py
   ```
   By default this uses the demo CSVs under `./data`. Flip `USE_VENDOR = True` inside `run.py` and export `TIINGO_API_KEY` to pull fresh data.

3. **Launch the Streamlit UI**
   ```bash
   streamlit run streamlit_app.py
   ```
   The sidebar lets you change lookback, entry/exit z thresholds, costs, and holding constraints. You can optionally upload custom CSVs (date + close columns) to override the bundled data without touching the filesystem.

## Data format
Each CSV is expected to have two columns: a date/datetime index and a `close` column. Example:
```
date,close
2023-01-03,400.10
2023-01-04,403.25
```
Place ETF files as `<TICKER>_daily.csv` and FX files as `<PAIR>_daily.csv` under `./data/` or upload them through the Streamlit app.

## Folder layout
- `arbitrage/`: core logic for FX normalization, pair analysis, signal generation, backtesting, and plotting helpers.
- `streamlit_app.py`: Streamlit dashboard wired to the same core modules.
- `run.py`: small CLI entry point that exercises the pipeline.
- `data/`: demo CSVs for the example ETF and FX series.

## Talking points
If you want a one-liner for interviews: “I built a cross-market ETF arbitrage tracker that normalizes U.S. and European listings via FX, detects >2σ mispricings with a z-score model, and backtests mean-reversion trades with latency, cost, and borrow modeling.”
