from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cents.data.splits import temporal_split_indices


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"text_rows", "has_text"}
    ]


def make_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x_values = df[feature_cols].astype(float).to_numpy()
    y_values = df[target_col].astype(float).to_numpy()
    xs, ys, anchors = [], [], []
    for end in range(history, len(df) - horizon + 1):
        xs.append(x_values[end - history:end].reshape(-1))
        ys.append(y_values[end:end + horizon])
        anchors.append(end)
    return np.asarray(xs), np.asarray(ys), anchors


def persistence_forecast(df: pd.DataFrame, target_col: str, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    y_values = df[target_col].astype(float).to_numpy()
    y_true, y_pred, anchors = [], [], []
    for end in range(history, len(df) - horizon + 1):
        y_true.append(y_values[end:end + horizon])
        y_pred.append(np.repeat(y_values[end - 1], horizon))
        anchors.append(end)
    return np.asarray(y_true), np.asarray(y_pred), anchors


def ridge_window_forecast(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    history: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x, y, anchors = make_windows(df, feature_cols, target_col, history, horizon)
    split = temporal_split_indices(len(x))
    if len(split.test) == 0 or len(split.train) < 5:
        return y, np.zeros_like(y), anchors
    model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0))
    model.fit(x[split.train], y[split.train])
    pred = model.predict(x[split.test])
    return y[split.test], pred, [anchors[i] for i in split.test]

