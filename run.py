from copy import deepcopy

from arbitrage.config import PARAMS, PAIR_CONFIG
from arbitrage.core import Explorer
from arbitrage.data import YahooLoader  # import new loader


def main() -> None:
    loader = YahooLoader(start=None, end=None)  # full history, or add dates
    params = deepcopy(PARAMS)

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
