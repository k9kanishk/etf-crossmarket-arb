from pathlib import Path
from copy import deepcopy
from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader  # or Vendor loader later

USE_VENDOR = False  # flip to True when you use real API data
DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    if USE_VENDOR:
        from arbitrage.data import TiingoLoader
        loader = TiingoLoader()
        params = deepcopy(PARAMS)  # keep strict for real data
    else:
        loader = DemoCSVLoader(str(DATA_DIR))
        params = deepcopy(PARAMS)
        # relax only for mock CSVs
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
