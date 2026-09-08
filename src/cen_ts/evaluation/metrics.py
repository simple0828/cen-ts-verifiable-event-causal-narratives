from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_diff_std: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.var(y_true) + 1e-12)
    nmse = float(mse / denom)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
    true_delta = y_true[:, -1] - y_true[:, 0] if y_true.ndim == 2 and y_true.shape[1] > 1 else y_true.reshape(len(y_true), -1)[:, -1]
    pred_delta = y_pred[:, -1] - y_pred[:, 0] if y_pred.ndim == 2 and y_pred.shape[1] > 1 else y_pred.reshape(len(y_pred), -1)[:, -1]
    threshold = 0.1 * (train_diff_std if train_diff_std and train_diff_std > 0 else np.std(true_delta) + 1e-8)
    true_trend = np.where(true_delta > threshold, 1, np.where(true_delta < -threshold, -1, 0))
    pred_trend = np.where(pred_delta > threshold, 1, np.where(pred_delta < -threshold, -1, 0))
    directional_accuracy = float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))
    trend_macro_f1 = float(f1_score(true_trend, pred_trend, labels=[-1, 0, 1], average="macro", zero_division=0))
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "NMSE": nmse,
        "sMAPE": smape,
        "Directional Accuracy": directional_accuracy,
        "Trend Macro-F1": trend_macro_f1,
    }
