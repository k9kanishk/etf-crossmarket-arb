Here is a clean, professional, and formatted `README.md` based on your description. I have added a placeholder for a screenshot and organized it for standard GitHub best practices.

-----

# Global ETF Cross-Market Arbitrage Explorer

> **One-liner:** A cross-market ETF arbitrage tracker that pulls live data, normalizes U.S. and European listings into a common currency, detects \>2σ mispricings with a z-score model, and backtests mean-reversion trades with latency, cost, and borrow modeling.

[Image of financial dashboard visualization]

A compact research toolkit for spotting cross-venue ETF mispricings, normalizing across FX, and backtesting mean-reversion trades.

While the repo ships with sample CSVs for offline testing, the **Streamlit app pulls live daily data from Yahoo Finance** for the tickers defined in `arbitrage/config.py`.

**Current Default Pairs:**

  * `SPY` (US) vs `CSTNL` (Europe)
  * `VWO` (US) vs `IZIZF` (Europe)

-----

## 🚀 Features

  * **FX-Aware Core**

      * Automatic normalization into a base currency (default: USD).
      * Supports common crosses (`EURUSD`, `GBPUSD`, `EURGBP`, `USDGBP`).
      * *Smart Logic:* If both legs are already in the base currency, the FX step is skipped automatically.

  * **Pair Spread Modelling**

      * Calculates rolling ratios, moving averages, and z-scores for U.S./EU ETF pairs.
      * Handles time-alignment across different market holidays/hours.

  * **Signal Engine & Backtester**

      * Flags $|z|$ breakouts beyond configurable thresholds.
      * Backtests mean-reversion exits and max-hold stopouts.
      * Models transaction fees, slippage, and borrow costs.

  * **Interactive Dashboard (Streamlit)**

      * Visualizes Ratio bands and Rolling correlation.
      * Displays Trade logs and Cumulative PnL curves.
      * Includes a **Cross-pair Z-score Heatmap** for scanning the whole universe.

  * **CLI Backtester**

      * Runs the pipeline headlessly against local CSVs or API data.
      * Modular `DataLoader` system (swap between Local CSV, Tiingo, or Yahoo).

-----

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/etf-crossmarket-arb.git
    cd etf-crossmarket-arb
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 📊 Usage

### 1\. Launch the Live Dashboard

Start the Streamlit UI to visualize live data from Yahoo Finance:

```bash
streamlit run streamlit_app.py
```

**What this does:**

  * Fetches daily OHLCV data via `YahooLoader` for pairs in `PAIR_CONFIG`.
  * Normalizes prices to `BASE_CCY` (USD).
  * Computes signals and runs the backtest on the fly.
  * **Controls:** Use the sidebar to adjust Lookback, Entry/Exit $|z|$, Position Size, Fees, and Latency.

### 2\. Run the CLI Backtest

Run the pipeline in the terminal (defaults to offline CSVs):

```bash
python run.py
```

> **Note:** To switch data sources in the CLI, open `run.py` and swap the loader to `YahooLoader` or `TiingoLoader`. (Tiingo requires `TIINGO_API_KEY` in env variables).

-----

## 📂 Project Structure

```text
arbitrage/
├── core.py           # FX normalization, pair alignment, z-score logic
├── backtest.py       # Trade simulation, PnL, equity curves, KPIs
├── config.py         # Configuration (PAIR_CONFIG, BASE_CCY, default params)
└── data.py           # DataLoaders (DemoCSV, Tiingo, Yahoo)
├── data/             # Sample CSVs for offline testing
├── run.py            # CLI entry point
└── streamlit_app.py  # Interactive Dashboard
```

-----

## ⚙️ Configuration & Data

### Modifying Pairs

To track different ETFs, edit `arbitrage/config.py`:

```python
PAIR_CONFIG = [
    {
        "name": "SP500_US_vs_LSE",
        "us": {"ticker": "SPY", "ccy": "USD"},
        "eu": {"ticker": "CSTNL", "ccy": "USD"} # Set ccy to EUR/GBP to trigger FX conversion
    },
    # ... add more pairs
]
```

### Using Custom CSVs

If using `DemoCSVLoader`, ensure files are placed in `./data/` and follow this format:

  * **Filename:** `<TICKER>_daily.csv` (e.g., `EURUSD_daily.csv`, `SPY_daily.csv`)
  * **Columns:** `date` (index) and `close`.

<!-- end list -->

```csv
date,close
2023-01-03,400.10
2023-01-04,403.25
```

-----

## 📝 Assumptions & Limitations

To keep the project lightweight, the current engine assumes:

1.  **Daily Bars:** Analysis is based on End-of-Day (EOD) close prices. Intraday microstructure is not modeled.
2.  **Perfect Borrow:** Both legs are treated as shortable with a configurable annualized borrow cost.
3.  **Market Impact:** Transaction costs are modeled as fixed bps + slippage; order-book depth is not simulated.
4.  **Single Base Currency:** All PnL is reported in `BASE_CCY` (USD). FX risk is modeled via closing rates (no intraday FX basis).

-----

## 🔮 Future Roadmap

  * **Intraday Granularity:** Move to 15-minute bars to capture shorter-term dislocations.
  * **Execution Model:** Add venue-aware simulation with realistic fee schedules and limit order logic.
  * **Risk Controls:** Implement kill switches for correlation breakdowns or volatility spikes.
  * **Regime Analysis:** Automatically pause trading when ADF tests fail (non-mean-reverting regime).

-----

## License

MIT
