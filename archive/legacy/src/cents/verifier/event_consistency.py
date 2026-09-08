from __future__ import annotations

import numpy as np
import pandas as pd

from cents.verifier.counterfactual_tests import random_window_effects


def score_event_consistency(
    event: dict,
    df: pd.DataFrame,
    target_variable: str = "OT",
    pre_window: int = 3,
    seed: int = 2026,
) -> dict:
    dates = pd.to_datetime(df["date"])
    values = df[target_variable].astype(float).to_numpy()
    event_time = pd.to_datetime(event.get("time"), errors="coerce")
    matches = np.flatnonzero(dates == event_time)
    if len(matches) == 0:
        return {**event, "direction_score": 0.0, "lag_score": 0.0, "counterfactual_score": 0.0, "final_consistency_score": 0.0}
    idx = int(matches[0])
    lag_min = max(0, int(event.get("expected_lag_min", 1)))
    lag_max = max(lag_min + 1, int(event.get("expected_lag_max", lag_min + 2)))
    pre_start = max(0, idx - pre_window)
    post_start = min(len(values), idx + lag_min)
    post_end = min(len(values), idx + lag_max + 1)
    if pre_start >= idx or post_start >= post_end:
        return {**event, "direction_score": 0.0, "lag_score": 0.0, "counterfactual_score": 0.0, "final_consistency_score": 0.0}
    pre = values[pre_start:idx]
    post = values[post_start:post_end]
    delta = float(np.nanmean(post) - np.nanmean(pre))
    polarity = event.get("polarity", "uncertain")
    if polarity == "positive":
        direction_score = 1.0 if delta > 0 else 0.0
    elif polarity == "negative":
        direction_score = 1.0 if delta < 0 else 0.0
    else:
        scale = np.nanstd(values) + 1e-8
        direction_score = float(min(1.0, abs(delta) / scale))
    baseline = random_window_effects(values, pre_window, max(1, post_end - post_start), seed=seed)
    random_mean = float(np.nanmean(np.abs(baseline))) if len(baseline) else 0.0
    effect = abs(delta)
    counterfactual_score = float(1.0 / (1.0 + np.exp(-(effect - random_mean))))
    lag_score = float(min(1.0, effect / (np.nanstd(values) + 1e-8)))
    final = 0.4 * direction_score + 0.3 * lag_score + 0.3 * counterfactual_score
    return {
        **event,
        "observed_delta": delta,
        "direction_score": float(direction_score),
        "lag_score": float(lag_score),
        "counterfactual_score": float(counterfactual_score),
        "final_consistency_score": float(final),
    }


def verify_events(events: list[dict], df: pd.DataFrame, target_variable: str = "OT", seed: int = 2026) -> list[dict]:
    return [score_event_consistency(event, df, target_variable, seed=seed) for event in events]

