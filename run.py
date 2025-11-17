from copy import deepcopy
from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader  # or Vendor loader later

USE_VENDOR = False  # flip to True when you use real API data


def main() -> None:
    if USE_VENDOR:
        from arbitrage.data import TiingoLoader
        loader = TiingoLoader()
        params = deepcopy(PARAMS)  # keep strict for real data
    else:
        loader = DemoCSVLoader("./data")
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
        print(res["metrics"])


if __name__ == "__main__":
    main()
