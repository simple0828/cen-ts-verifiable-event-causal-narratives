from __future__ import annotations

import numpy as np


def event_series(events: list[dict], n: int, representation: str = "confidence_signed") -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for ev in events:
        event_type = ev.get("event_type", "other")
        arr = series.setdefault(event_type, np.zeros(n, dtype=float))
        idx = ev.get("time_index")
        if idx is None or not (0 <= int(idx) < n):
            continue
        sign = 1.0 if ev.get("expected_direction") == "positive" else -1.0 if ev.get("expected_direction") == "negative" else 0.0
        conf = float(ev.get("extraction_confidence", 0.5))
        if representation == "binary":
            value = 1.0
        elif representation == "count":
            value = 1.0
        elif representation == "signed_intensity":
            value = sign
        else:
            value = sign * conf if sign else conf * 0.25
        arr[int(idx)] += value
    return series
