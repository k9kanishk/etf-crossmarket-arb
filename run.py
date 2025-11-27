from copy import deepcopy
from pathlib import Path

from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader, TiingoLoader

DATA_DIR = Path(__file__).resolve().parent / "data"
USE_VENDOR = True  # turn this on when you want Tiingo data


def main() -> None:
    if USE_VENDOR:
        # Use Tiingo for ETFs, CSVs for FX
        loader = TiingoLoader(
            start=None,        # or "2015-01-01"
            end=None,          # or "2024-12-31"
            csv_path=str(DATA_DIR),
        )
        params = deepcopy(PARAMS)  # keep your stricter prod-like params
    else:
        loader = DemoCSVLoader(str(DATA_DIR))
        params = deepcopy(PARAMS)
        params.update({
            "lookback": 20,
            "entry_z": 1.0,
            "exit_z": 0.2,
            "min_corr": 0.0,
        })

    explorer = Explorer(loader, params)
    results = explorer.run_all(PAIR_CONFIG)

    for name, res in results.items():
        print(f"\n=== {name} ===")
        if "error" in res:
            print("ERROR:", res["error"])
        elif res.get("skipped"):
            print("SKIPPED:", res.get("reason", ""))
        else:
            print(res.get("metrics", {}))


if __name__ == "__main__":
    main()
