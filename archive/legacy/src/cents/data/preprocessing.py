from __future__ import annotations

import pandas as pd


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if text.upper() == "NA":
        return ""
    return " ".join(text.split())


def infer_frequency(dates: pd.Series) -> str:
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(parsed) < 3:
        return "unknown"
    median_days = parsed.diff().dropna().dt.days.median()
    if median_days <= 2:
        return "daily"
    if 5 <= median_days <= 9:
        return "weekly"
    if 25 <= median_days <= 35:
        return "monthly"
    return "unknown"


def recommended_windows(freq: str) -> tuple[list[int], list[int]]:
    if freq == "daily":
        return [30, 60, 90], [7, 14, 30]
    if freq == "weekly":
        return [12, 24, 36], [4, 8, 12]
    if freq == "monthly":
        return [12, 24, 36], [3, 6, 12]
    return [24], [6]

