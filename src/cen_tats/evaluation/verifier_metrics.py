from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def corruption_benchmark_scores(good_scores: list[float], bad_scores: list[float], threshold: float) -> dict:
    y = np.asarray([1] * len(good_scores) + [0] * len(bad_scores))
    s = np.asarray(good_scores + bad_scores, dtype=float)
    pred = (s >= threshold).astype(int)
    if len(set(y.tolist())) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "correct_event_mean_score": float(np.mean(good_scores)) if good_scores else 0.0,
        "wrong_event_mean_score": float(np.mean(bad_scores)) if bad_scores else 0.0,
        "wrong_keep_rate": float(np.mean(np.asarray(bad_scores) >= threshold)) if bad_scores else 0.0,
        "wrong_delete_rate": float(np.mean(np.asarray(good_scores) < threshold)) if good_scores else 0.0,
    }
