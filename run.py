from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import DemoCSVLoader

if __name__ == "__main__":
    loader = DemoCSVLoader("./data")   # swap for your vendor-backed loader later
    explorer = Explorer(loader, PARAMS)
    results = explorer.run_all(PAIR_CONFIG)
    for name, res in results.items():
        print(f"\n=== {name} ===")
        if "metrics" in res:
            print(res["metrics"])
        else:
            print(res)
