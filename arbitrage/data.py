
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

# ... keep DataLoader and DemoCSVLoader exactly as they are above ...

class TiingoLoader(DataLoader):
    """
    Data loader that pulls daily ETF prices from the Tiingo API.

    - ETFs: from Tiingo (EOD)
    - FX:   currently delegated to DemoCSVLoader, using local CSVs.
    """

    def __init__(
        self,
        start: str | None = None,
        end: str | None = None,
        api_key: str | None = None,
        csv_path: str = "data",   # <-- accepts csv_path now
        **_: object,              # <-- swallows any extra kwargs safely
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

        # Reuse CSV loader for FX
        self.fx_csv_loader = DemoCSVLoader(csv_path)

    # ---- helpers ---------------------------------------------------------

    def _date_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.start:
            kw["startDate"] = self.start
        if self.end:
            kw["endDate"] = self.end
        return kw

    # ---- public API used by Explorer / Streamlit ------------------------

    def load_etf_daily(self, ticker: str) -> pd.DataFrame:
        """
        Return a DataFrame with DatetimeIndex and 'close' column
        (using adjClose if available).
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
        For now, FX still comes from your CSVs (e.g. EURUSD_daily.csv).
        """
        return self.fx_csv_loader.load_fx_daily(pair)

