from __future__ import annotations

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}

