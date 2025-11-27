from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional
import functools
import logging
import os

import pandas as pd
import yfinance as yf

# Tiingo is optional – only needed if you actually use TiingoLoader
try:
    from tiingo import TiingoClient
except ImportError:
    TiingoClient = None


# ======================================================================
# Base interface
# ======================================================================

class DataLoader(ABC):
    """
    Vendor-agnostic data interface.

    Implementors must return a DataFrame indexed by date/datetime
    with a single column named 'close'.
    """

    @abstractmethod
    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        raise NotImplementedError


# ======================================================================
# Simple CSV loader (used by tests and as a fallback)
# ======================================================================

class DemoCSVLoader(DataLoader):
    """
    Simple CSV loader for local testing.

    Filenames expected like: ./data/SPY_daily.csv, ./data/EURUSD_daily.csv

    CSV format:
        date,close
        2024-10-01,500.12
        ...
    """

    def __init__(self, root: str = "./data"):
        self.root = root

    def _read(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()

        # Normalise expected column name
        if "close" not in df.columns:
            df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            raise ValueError(f"CSV {path} must contain a 'close' column (or convertible)")

        return df[["close"]]

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        return self._read(f"{self.root}/{ticker}_daily.csv")

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        return self._read(f"{self.root}/{pair}_daily.csv")


# ======================================================================
# Tiingo loader (optional, currently not used in Streamlit app)
# ======================================================================

class TiingoLoader(DataLoader):
    """
    Data loader that pulls daily ETF prices from Tiingo when possible.

    - ETFs: try Tiingo, fall back to local CSVs via DemoCSVLoader
    - FX:   always from local CSVs (EURUSD_daily.csv etc.) for now
    """

    def __init__(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        api_key: Optional[str] = None,
        csv_path: str = "data",
        **_: object,  # swallow any extra kwargs safely
    ) -> None:
        if TiingoClient is None:
            raise ImportError("tiingo package not installed – cannot use TiingoLoader")

        cfg: dict[str, object] = {"session": True}
        cfg["api_key"] = api_key or os.getenv("TIINGO_API_KEY")
        if not cfg["api_key"]:
            raise ValueError(
                "TiingoLoader: set TIINGO_API_KEY env var or pass api_key=."
            )

        self.client = TiingoClient(cfg)
        self.start = start
        self.end = end

        # use same CSV loader you already rely on
        self.csv_loader = DemoCSVLoader(csv_path)

    def _date_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.start:
            kw["startDate"] = self.start
        if self.end:
            kw["endDate"] = self.end
        return kw

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        """Try Tiingo; on any error fall back to CSV."""
        try:
            df = self.client.get_dataframe(
                ticker,
                frequency="daily",
                **self._date_kwargs(),
            )
            if df.empty:
                raise ValueError(f"Tiingo: no data for {ticker}")

            col = "adjClose" if "adjClose" in df.columns else "close"
            series = df[col].dropna()
            if series.empty:
                raise ValueError(f"Tiingo: all NaNs for {ticker}")

            out = pd.DataFrame({"close": series})
            out.index = pd.to_datetime(out.index).tz_localize(None)
            out = out.sort_index()
            return out

        except Exception as e:
            logging.warning(
                "TiingoLoader: failed for %s (%s). Falling back to CSV.", ticker, e
            )
            return self.csv_loader.load_etf_daily(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        """Currently always load FX from local CSVs."""
        return self.csv_loader.load_fx_daily(pair)


# ======================================================================
# Yahoo Finance loader (used by your live Streamlit app)
# ======================================================================

class YahooLoader(DataLoader):
    """
    Live data loader using Yahoo Finance via yfinance.

    - ETFs: tickers taken directly from PAIR_CONFIG (SPY, CSTNL, VWO, IZZIF, ...)
    - FX:   internal 'EURUSD' -> Yahoo 'EURUSD=X', etc.

    Returns DataFrames indexed by date with a single column: 'close'.
    """

    def __init__(self, period: str = "2y") -> None:
        # how much history to pull from Yahoo – tweak if needed
        self.period = period

    def _download(self, symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            period=self.period,
            auto_adjust=True,   # already adjusted, so we can safely use 'Close'
            progress=False,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"YahooLoader: no data returned for {symbol}")

        # prefer Adj Close if present, else Close
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        sub = df[col]

        # sub is usually a Series, but be defensive in case of weird multi-index
        if isinstance(sub, pd.DataFrame):
            # take the first column as a fallback
            series = sub.iloc[:, 0].dropna()
        else:
            series = sub.dropna()

        if series.empty:
            raise ValueError(f"YahooLoader: all close values NaN for {symbol}")

        out = pd.DataFrame({"close": series})
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        return out

    @functools.lru_cache(maxsize=64)
    def _download_cached(self, symbol: str) -> pd.DataFrame:
        # cache by symbol so multiple calls (pairs, FX) don't re-hit Yahoo
        return self._download(symbol)

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        # e.g. "SPY", "CSTNL", "VWO", "IZIZF"
        return self._download_cached(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        # internal "EURUSD" -> Yahoo "EURUSD=X"
        symbol = pair + "=X"
        return self._download_cached(symbol)
