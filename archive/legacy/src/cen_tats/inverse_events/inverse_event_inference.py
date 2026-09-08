from __future__ import annotations

import numpy as np

from cen_tats.inverse_events.series_summarizer import summarize_window


def infer_event_from_history(values: np.ndarray) -> dict:
    s = summarize_window(values)
    vol = s["volatility"]
    change = s["change"]
    if change > max(vol, 1e-8):
        direction = "positive"
        event_type = "positive_price_shock"
    elif change < -max(vol, 1e-8):
        direction = "negative"
        event_type = "negative_price_shock"
    else:
        direction = "neutral"
        event_type = "stable_or_mixed_pattern"
    intensity = "strong" if abs(change) > 2 * max(vol, 1e-8) else "moderate" if abs(change) > max(vol, 1e-8) else "weak"
    pattern = "sudden" if abs(s["max_rise"]) > 1.5 * max(vol, 1e-8) or abs(s["max_drop"]) > 1.5 * max(vol, 1e-8) else "gradual"
    return {
        "inferred_event_type": event_type,
        "direction": direction,
        "intensity": intensity,
        "pattern_type": pattern,
        "observable_start": int(s["change_point"]),
        "observable_duration": int(max(1, len(values) - s["change_point"])),
        "confidence": float(min(0.95, 0.45 + abs(change) / (3 * max(vol, 1e-8) + abs(change) + 1e-8))),
        "numeric_evidence": [
            f"slope={s['slope']:.6g}",
            f"change={s['change']:.6g}",
            f"change_point=t-{len(values) - s['change_point']}",
            f"volatility={s['volatility']:.6g}",
        ],
        "alternative_hypotheses": ["seasonal_or_noise"] if intensity == "weak" else [],
        "summary": s,
    }
