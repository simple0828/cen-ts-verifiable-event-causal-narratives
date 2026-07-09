from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bars(table: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or metric not in table:
        return
    ax = table.groupby("method")[metric].mean().sort_values().plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel(metric)
    ax.set_title(f"Mean {metric} by method")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

