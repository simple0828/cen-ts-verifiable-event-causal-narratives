from __future__ import annotations

from pathlib import Path

import numpy as np

from cen_tats.io_utils import stable_hash


def cache_key(texts: list[str], prompt_hash: str, model_name: str, module_name: str, dataset: str, params: dict) -> str:
    return stable_hash(
        {
            "input_hash": stable_hash(texts),
            "prompt_hash": prompt_hash,
            "model_name": model_name,
            "model_parameters": params,
            "module_name": module_name,
            "dataset": dataset,
        },
        n=24,
    )


def save_embedding_cache(path: str | Path, embeddings: np.ndarray, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, meta=np.array([meta], dtype=object))


def load_embedding_cache(path: str | Path) -> tuple[np.ndarray, dict] | None:
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return data["embeddings"], dict(data["meta"][0])
