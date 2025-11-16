from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader, TiingoLoader

USE_TIINGO = True  # flip to False if you want to fall back to CSVs


if __name__ == "__main__":
    if USE_TIINGO:
        loader = TiingoLoader()  # uses TIINGO_API_KEY from .env
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
