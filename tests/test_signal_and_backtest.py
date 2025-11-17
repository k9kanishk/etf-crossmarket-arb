import pandas as pd

from arbitrage.backtest import Backtester
from arbitrage.config import PARAMS
from arbitrage.core import SignalEngine


def test_signal_and_backtest_simple_trade():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    ratio_vals = [1.0, 1.5, 1.6, 1.2, 1.0, 1.0]
    z_vals = [0.0, 2.1, 1.9, 0.1, 0.0, 0.0]
    df = pd.DataFrame({"ratio": ratio_vals, "z": z_vals}, index=idx)

    params = {
        **PARAMS,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "max_holding_days": 10,
        "position_usd": 100_000,
        "min_corr": 0.0,
    }
    sig_engine = SignalEngine(params)
    sigs = sig_engine.generate(df)
    assert len(sigs) == 1

    bt = Backtester(params)
    equity, trades, market_time = bt.run(df, sigs)
    assert len(trades) == 1
    assert trades[0].pnl_usd > 0
    assert market_time["days_in_market"] > 0
    assert not equity.empty
