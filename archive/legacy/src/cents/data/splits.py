from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def temporal_split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> TemporalSplit:
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    indices = np.arange(n)
    return TemporalSplit(indices[:train_end], indices[train_end:val_end], indices[val_end:])

