
from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

class DataLoader(ABC):
    """
    Replace these with vendor-specific readers (Refinitiv, Bloomberg, Polygon, Tiingo, etc.).
    Expected output: DataFrame indexed by date/datetime, must include column 'close'.
    """

    @abstractmethod
    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        raise NotImplementedError


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
        # Normalize expected column name
        if "close" not in df.columns:
            df.columns = [c.lower() for c in df.columns]
        return df

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        return self._read(f"{self.root}/{ticker}_daily.csv")

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        return self._read(f"{self.root}/{pair}_daily.csv")

import os
import logging
import pandas as pd
from tiingo import TiingoClient

from .data import DataLoader, DemoCSVLoader  # adjust import if needed


class TiingoLoader(DataLoader):
    """
    Data loader that pulls daily ETF prices from Tiingo when possible.

    - ETFs: try Tiingo, fall back to local CSVs via DemoCSVLoader
    - FX:   always from local CSVs (EURUSD_daily.csv etc.) for now
    """

    def __init__(
        self,
        start: str | None = None,
        end: str | None = None,
        api_key: str | None = None,
        csv_path: str = "data",
        **_: object,  # swallow any extra kwargs safely
    ) -> None:
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
            # Tiingo EOD uses adjClose for adjusted prices
            col = "adjClose" if "adjClose" in df.columns else "close"
            out = pd.DataFrame({"close": df[col]})
            out.index = pd.to_datetime(out.index).tz_localize(None)
            out = out.sort_index()
            return out

        except Exception as e:
            logging.warning(
                "TiingoLoader: failed for %s (%s). Falling back to CSV.", ticker, e
            )
            # this will read e.g. data/CSPX.L_daily.csv which you already have
            return self.csv_loader.load_etf_daily(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        """Currently always load FX from local CSVs."""
        return self.csv_loader.load_fx_daily(pair)

import functools
import pandas as pd
import yfinance as yf

from .data import DataLoader  # adjust import path if needed


class YahooLoader(DataLoader):
    """
    Live data loader using Yahoo Finance via yfinance.

    - ETFs: tickers taken directly from PAIR_CONFIG (SPY, CSTNL, VWO, IZIZF, ...)
    - FX:   internal 'EURUSD' -> Yahoo 'EURUSD=X', etc.
    """

    def __init__(self, start: str | None = None, end: str | None = None) -> None:
        self.start = start
        self.end = end

    def _raw_download(self, symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=self.start,
            end=self.end,
            progress=False,
            auto_adjust=False,  # keep Adj Close available
        )

        # Ensure we really have a DataFrame with rows
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"YahooLoader: no data returned for {symbol}")

        return df

    def _extract_close(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract a single 'close' series from a Yahoo-style OHLCV frame,
        handling both simple and MultiIndex columns.
        """
        # Prefer adjusted close if present
        if "Adj Close" in df.columns:
            sub = df["Adj Close"]
        elif "Close" in df.columns:
            sub = df["Close"]
        else:
            raise ValueError(
                f"YahooLoader: expected 'Adj Close' or 'Close' for {symbol}, "
                f"got columns: {list(df.columns)}"
            )

        # sub may be Series or DataFrame (e.g. MultiIndex first level)
        if isinstance(sub, pd.DataFrame):
            # If it's a single-column DataFrame, take that column
            if sub.shape[1] == 1:
                series = sub.iloc[:, 0]
            else:
                # If somehow we got multiple columns, take the first as a fallback
                series = sub.iloc[:, 0]
        else:
            series = sub

        series = series.dropna()
        if series.empty:
            raise ValueError(f"YahooLoader: all close values NaN for {symbol}")

        out = series.to_frame(name="close")
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        return out

    def _download(self, symbol: str) -> pd.DataFrame:
        df = self._raw_download(symbol)
        return self._extract_close(df, symbol)

    @functools.lru_cache(maxsize=64)
    def _download_cached(self, symbol: str) -> pd.DataFrame:
        return self._download(symbol)

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        return self._download_cached(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        symbol = pair + "=X"
        return self._download_cached(symbol)
