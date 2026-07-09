from __future__ import annotations

import numpy as np


def forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom)) if not np.all(np.isnan(denom)) else float("nan")
    if y_true.shape[-1] > 1:
        true_dir = np.sign(np.diff(y_true, axis=-1))
        pred_dir = np.sign(np.diff(y_pred, axis=-1))
        directional_accuracy = float(np.mean(true_dir == pred_dir))
    else:
        directional_accuracy = float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "trend_f1": directional_accuracy,
    }

