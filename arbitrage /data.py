
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
