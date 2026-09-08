from __future__ import annotations

import numpy as np
import pandas as pd

from cents.models.numeric_baselines import ridge_window_forecast
from cents.text.text_embedder import fit_text_embeddings


def add_text_embeddings(df: pd.DataFrame, cache_path: str, n_components: int = 16) -> tuple[pd.DataFrame, list[str]]:
    arr = fit_text_embeddings(df, cache_path=cache_path, n_components=n_components)
    out = df.copy()
    cols = []
    for i in range(arr.shape[1]):
        col = f"text_emb_{i}"
        out[col] = arr[:, i]
        cols.append(col)
    return out, cols


def text_fusion_forecast(df: pd.DataFrame, base_cols: list[str], target_col: str, history: int, horizon: int, cache_path: str):
    fused, text_cols = add_text_embeddings(df, cache_path)
    return ridge_window_forecast(fused, base_cols + text_cols, target_col, history, horizon)

