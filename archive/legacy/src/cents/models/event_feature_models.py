from __future__ import annotations

import pandas as pd

from cents.models.numeric_baselines import ridge_window_forecast
from cents.text.event_extractor import events_to_frame


EVENT_FEATURE_COLS = ["event_count", "pos_events", "neg_events", "event_confidence", "verified_event_score"]


def add_event_features(df: pd.DataFrame, events: list[dict], verifier_threshold: float | None = None) -> pd.DataFrame:
    features = events_to_frame(events, df["date"], verifier_threshold=verifier_threshold)
    out = df.copy()
    for col in EVENT_FEATURE_COLS:
        out[col] = features[col].astype(float).to_numpy()
    return out


def event_feature_forecast(
    df: pd.DataFrame,
    base_cols: list[str],
    target_col: str,
    events: list[dict],
    history: int,
    horizon: int,
    verifier_threshold: float | None = None,
):
    enriched = add_event_features(df, events, verifier_threshold=verifier_threshold)
    return ridge_window_forecast(enriched, base_cols + EVENT_FEATURE_COLS, target_col, history, horizon)

