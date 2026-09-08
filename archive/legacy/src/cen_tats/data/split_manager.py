from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cen_tats.io_utils import stable_hash, write_json


@dataclass(frozen=True)
class V5Split:
    train_core: np.ndarray
    prompt_dev: np.ndarray
    model_val: np.ndarray
    test: np.ndarray
    split_hash: str

    def as_dict(self) -> dict:
        return {
            "train_core": self.train_core.tolist(),
            "prompt_dev": self.prompt_dev.tolist(),
            "model_val": self.model_val.tolist(),
            "test": self.test.tolist(),
            "split_hash": self.split_hash,
        }


def temporal_v5_split(n: int, ratios: tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2)) -> V5Split:
    if n < 10:
        raise ValueError("Need at least 10 time points for v5 split")
    train_end = int(n * ratios[0])
    prompt_end = train_end + int(n * ratios[1])
    val_end = prompt_end + int(n * ratios[2])
    idx = np.arange(n)
    payload = {
        "n": n,
        "ratios": ratios,
        "train_end": train_end,
        "prompt_end": prompt_end,
        "val_end": val_end,
    }
    return V5Split(idx[:train_end], idx[train_end:prompt_end], idx[prompt_end:val_end], idx[val_end:], stable_hash(payload))


def save_split(dataset: str, dates: pd.Series, split: V5Split, out_dir: str | Path = "data/processed/splits") -> Path:
    payload = split.as_dict()
    payload["dataset"] = dataset
    payload["date_ranges"] = {}
    for key in ["train_core", "prompt_dev", "model_val", "test"]:
        arr = getattr(split, key)
        payload["date_ranges"][key] = {
            "start": str(pd.to_datetime(dates.iloc[int(arr[0])]).date()) if len(arr) else None,
            "end": str(pd.to_datetime(dates.iloc[int(arr[-1])]).date()) if len(arr) else None,
            "n": int(len(arr)),
        }
    path = Path(out_dir) / f"{dataset}_v5.json"
    write_json(path, payload)
    return path


def assign_window_split(anchor: int, horizon: int, split: V5Split) -> str | None:
    label_end = anchor + horizon
    for name in ["train_core", "prompt_dev", "model_val", "test"]:
        arr = getattr(split, name)
        if len(arr) and int(arr[0]) <= anchor <= int(arr[-1]) and label_end <= int(arr[-1]):
            return name
    return None
