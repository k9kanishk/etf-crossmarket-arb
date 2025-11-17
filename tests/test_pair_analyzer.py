import numpy as np
import pandas as pd
import pytest

from arbitrage.config import BASE_CCY
from arbitrage.core import FXNormalizer, PairAnalyzer, PairData


def test_pair_analyzer_ratio_stationarity():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    us_px = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    # EU leg slightly noisier so z-series is defined
    eu_px = us_px / 2.0 + np.random.default_rng(0).normal(0, 0.1, len(idx))

    fx = FXNormalizer(BASE_CCY, {})
    analyzer = PairAnalyzer(fx, lookback=10)
    pair = PairData("TEST", us_px, eu_px, "USD", "USD")
    ratio_df = analyzer.build_ratio_df(pair)

    assert not ratio_df.empty
    # ratio should stay near 2 with modest variance
    assert ratio_df["ratio"].mean() == pytest.approx(2.0, rel=0.05)
    pval = analyzer.adf_pvalue(np.log(ratio_df["ratio"]))
    assert 0 <= pval <= 1
