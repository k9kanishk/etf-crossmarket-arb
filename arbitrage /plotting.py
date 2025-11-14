
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_ratio_bands(df: pd.DataFrame, title: str = "Ratio & Bands") -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["ratio"], label="ratio")
    plt.plot(df.index, df["mu"], label="mean")
    plt.plot(df.index, df["mu"] + 2 * df["sigma"], label="+2σ")
    plt.plot(df.index, df["mu"] - 2 * df["sigma"], label="-2σ")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_signal_heatmap(zscores: pd.DataFrame, title: str = "Signal Heatmap") -> None:
    # zscores: index=date, columns=pair_name -> z
    plt.figure(figsize=(10, 5))
    plt.imshow(zscores.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="z-score")
    plt.yticks(range(len(zscores.columns)), zscores.columns)
    plt.title(title)
    plt.tight_layout()
    plt.show()
