from __future__ import annotations

import numpy as np


def random_window_effects(values: np.ndarray, pre: int, post: int, repeats: int = 64, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n <= pre + post + 2:
        return np.zeros(repeats)
    starts = rng.integers(pre, n - post, size=repeats)
    effects = []
    for idx in starts:
        before = values[idx - pre:idx]
        after = values[idx:idx + post]
        effects.append(float(np.nanmean(after) - np.nanmean(before)))
    return np.asarray(effects)

