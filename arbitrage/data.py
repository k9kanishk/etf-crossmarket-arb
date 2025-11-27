
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
import pandas as pd
from tiingo import TiingoClient

# ... keep your existing DemoCSVLoader above ...

class TiingoLoader:
    """
    Data loader that pulls daily ETF prices from the Tiingo API.
    FX can still come from your existing CSVs via an internal DemoCSVLoader.
    """

    def __init__(
        self,
        start: str | None = None,
        end: str | None = None,
        api_key: str | None = None,
        csv_path: str = "data",
    ) -> None:
        # Configure Tiingo client
        cfg: dict[str, object] = {"session": True}
        cfg["api_key"] = api_key or os.getenv("TIINGO_API_KEY")
        if not cfg["api_key"]:
            raise ValueError(
                "TiingoLoader: set TIINGO_API_KEY env var or pass api_key=."
            )
        self.client = TiingoClient(cfg)
        self.start = start
        self.end = end

        # Use your existing CSV loader for FX for now
        self.fx_csv_loader = DemoCSVLoader(csv_path)

    # ---- helpers ---------------------------------------------------------

    def _date_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.start:
            kw["startDate"] = self.start
        if self.end:
            kw["endDate"] = self.end
        return kw

    # ---- public API used by Explorer ------------------------------------

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        """
        Return a DataFrame with a DatetimeIndex and a 'close' column,
        using adjClose if available.
        """
        df = self.client.get_dataframe(
            ticker,
            frequency="daily",
            **self._date_kwargs(),
        )

        col = "adjClose" if "adjClose" in df.columns else "close"
        out = pd.DataFrame({"close": df[col]})
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        return out

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        """
        For now, just delegate FX to the CSV loader, so you keep using
        your existing EURUSD_daily.csv etc.
        Later you can swap this to Tiingo's FX endpoint.
        """
        return self.fx_csv_loader.load_fx_daily(pair)


import os
import datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()  # loads TIINGO_API_KEY from .env if present


class TiingoLoader(DataLoader):
    """
    DataLoader implementation using Tiingo's REST API.

    - Equities/ETFs: https://api.tiingo.com/tiingo/daily/<ticker>/prices
    - FX:           https://api.tiingo.com/tiingo/fx/prices?tickers=eurusd
    """

    def __init__(
        self,
        api_key: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ):
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not set in environment or passed to TiingoLoader")

        # default: last 5 years
        end_date = dt.date.today() if end is None else dt.date.fromisoformat(end)
        start_date = end_date - dt.timedelta(days=365 * 5) if start is None else dt.date.fromisoformat(start)

        self.start = start_date.isoformat()
        self.end = end_date.isoformat()

        self.base_eod = "https://api.tiingo.com/tiingo/daily"
        self.base_fx = "https://api.tiingo.com/tiingo/fx/prices"

    def _get_json(self, url: str, params: dict) -> list[dict]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        """
        Load daily OHLC for an ETF/stock and return a DF with a 'close' column
        (uses adjusted close if available).
        """
        url = f"{self.base_eod}/{ticker}/prices"
        data = self._get_json(
            url,
            params={
                "startDate": self.start,
                "endDate": self.end,
                "format": "json",
            },
        )
        if not data:
            raise ValueError(f"No Tiingo EOD data for ticker {ticker}")

        df = pd.DataFrame(data)
        # Tiingo returns 'date' as ISO string, 'adjClose' and 'close'
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        if "adjClose" in df.columns:
            df["close"] = df["adjClose"]
        # keep only 'close'
        return df[["close"]]

    def load_fx_daily(self, pair: str) -> pd.DataFrame:
        """
        Load daily FX series. For 'EURUSD' we query Tiingo with 'eurusd'.
        """
        tiingo_pair = pair.lower()  # e.g. 'eurusd'
        data = self._get_json(
            self.base_fx,
            params={
                "tickers": tiingo_pair,
                "startDate": self.start,
                "endDate": self.end,
            },
        )
        if not data:
            raise ValueError(f"No Tiingo FX data for pair {pair}")

        # Tiingo FX returns a list per ticker; take first element's priceData
        price_data = data[0]["priceData"]
        df = pd.DataFrame(price_data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        # FX has 'mid' price – use as 'close'
        if "mid" in df.columns:
            df["close"] = df["mid"]

        return df[["close"]]
