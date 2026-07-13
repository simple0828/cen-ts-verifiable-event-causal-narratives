from __future__ import annotations

import numpy as np


def summarize_window(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        v = float(values[-1]) if len(values) else 0.0
        return {
            "start": v,
            "end": v,
            "change": 0.0,
            "relative_change": 0.0,
            "slope": 0.0,
            "volatility": 0.0,
            "max_rise": 0.0,
            "max_drop": 0.0,
            "change_point": 0,
            "local_min": v,
            "local_max": v,
        }
    diffs = np.diff(values)
    denom = abs(values[0]) + 1e-8
    change_point = int(np.argmax(np.abs(diffs))) + 1
    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 2 else float(diffs[-1])
    return {
        "start": float(values[0]),
        "end": float(values[-1]),
        "change": float(values[-1] - values[0]),
        "relative_change": float((values[-1] - values[0]) / denom),
        "slope": slope,
        "volatility": float(np.std(diffs)),
        "max_rise": float(np.max(diffs)),
        "max_drop": float(np.min(diffs)),
        "change_point": change_point,
        "local_min": float(np.min(values)),
        "local_max": float(np.max(values)),
    }
