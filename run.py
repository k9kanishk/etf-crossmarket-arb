import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader  # or Vendor loader later (TiingoLoader etc.)

USE_VENDOR = False  # set True when you wire your real data loader


def main() -> None:
    if USE_VENDOR:
        from arbitrage.data import TiingoLoader  # example, if you implement it
        loader = TiingoLoader()
    else:
        loader = DemoCSVLoader("./data")

    explorer = Explorer(loader, PARAMS)
    results = explorer.run_all(PAIR_CONFIG)

    for name, res in results.items():
        print(f"\n=== {name} ===")
        if "metrics" in res:
            print(res["metrics"])
        else:
            print(res)


if __name__ == "__main__":
    main()
