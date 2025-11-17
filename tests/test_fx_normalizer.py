import pandas as pd
import pytest
from arbitrage.core import FXNormalizer


def test_fx_normalizer_direct_and_inverse():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    fx_df = pd.DataFrame({"close": [1.1, 1.1, 1.1]}, index=idx)
    fx = FXNormalizer("USD", {"EURUSD": fx_df})
    eur_prices = pd.Series([10, 11, 12], index=idx)

    usd_converted = fx.convert(eur_prices, "EUR")
    assert usd_converted.iloc[0] == 11

    fx_inv = FXNormalizer("USD", {"USDEUR": pd.DataFrame({"close": [0.91, 0.91, 0.91]}, index=idx)})
    usd_converted_inv = fx_inv.convert(eur_prices, "EUR")
    assert usd_converted_inv.iloc[1] == pytest.approx(12.089, rel=1e-3)
