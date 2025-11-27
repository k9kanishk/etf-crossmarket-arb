# Reporting/base currency
BASE_CCY = "USD"

# Pairs to analyze (tickers are examples; adjust to your data/vendor)
PAIR_CONFIG = [
    {
        "name": "S&P500_US_vs_LSE",
        "us": {"ticker": "SPY",     "ccy": "USD", "venue": "NYSE"},
        "eu": {"ticker": "CSTNL",  "ccy": "USD", "venue": "LSE"},
    },
    {
        "name": "EM_US_vs_EU",
        "us": {"ticker": "VWO",     "ccy": "USD", "venue": "NYSE"},
        "eu": {"ticker": "IZIZF", "ccy": "EUR", "venue": "BorsaItaliana"},
    },
]

# Strategy / backtest parameters
PARAMS = {
    "lookback": 20,        # was 60
    "entry_z": 1.0,        # was 2.0
    "exit_z": 0.2,         # a bit tighter exit
    "max_holding_days": 5,
    "use_adf_filter": False,
    "min_corr": 0.0,       # was 0.8 (turn off for now)
    "txn_fee_bps": 0.5,
    "slippage_bps": 1.0,
    "borrow_bps": 30.0,
    "latency_bars": 1,
    "position_usd": 1_000_000,
}

