# Reporting/base currency
BASE_CCY = "USD"

# Pairs to analyze (tickers are examples; adjust to your data/vendor)
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

# Strategy / backtest parameters
PARAMS = {
    "lookback": 60,            # rolling window for mean/std/corr
    "entry_z": 2.0,            # open when |z| > entry_z
    "exit_z": 0.5,             # close when |z| < exit_z
    "max_holding_days": 5,     # time stop
    "use_adf_filter": False,   # require stationarity (ADF) if True
    "min_corr": 0.8,           # require min rolling corr of returns
    "txn_fee_bps": 0.5,        # per side
    "slippage_bps": 1.0,       # per side
    "borrow_bps": 30.0,        # annualized borrow cost for short leg
    "latency_bars": 1,         # bars between signal and fill
    "position_usd": 1_000_000, # gross per trade leg
}
