
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

import pandas as pd
import yfinance as yf
import functools

from . import config  # if you need BASE_CCY; optional
from .data import DataLoader  # adjust import to your actual file structure


class YahooLoader(DataLoader):
    """
    Live data loader using Yahoo Finance via yfinance.

    - ETFs: uses tickers as defined in PAIR_CONFIG (SPY, VWO, CSPX.L, IEMM.MI, etc.)
    - FX:   internal code 'EURUSD' -> Yahoo symbol 'EURUSD=X', etc.
    """

    def __init__(self, start: str | None = None, end: str | None = None) -> None:
        self.start = start
        self.end = end

    def _download(self, symbol: str) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=self.start,
            end=self.end,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No data returned from Yahoo for {symbol}")

        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        out = pd.DataFrame({"close": df[col]})
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        return out

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        # PAIR_CONFIG tickers are already Yahoo-compatible: SPY, CSPX.L, VWO, IEMM.MI, ...
        return self._download(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        """
        Internal codes: 'EURUSD', 'GBPUSD', etc.
        Yahoo FX symbols are like 'EURUSD=X', 'GBPUSD=X'.
        """
        symbol = pair + "=X"
        return self._download(symbol)
    
    @functools.lru_cache(maxsize=64)
    def _download_cached(self, symbol: str) -> pd.DataFrame:
        return self._download(symbol)

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        return self._download_cached(ticker)

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        symbol = pair + "=X"
        return self._download_cached(symbol)
