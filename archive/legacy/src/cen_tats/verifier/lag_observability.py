from __future__ import annotations


def observable(event_index: int, lag_min: int, anchor: int) -> bool:
    return event_index + lag_min <= anchor


def effect_status(event_index: int, lag_min: int, lag_max: int, anchor: int) -> str:
    if anchor < event_index + lag_min:
        return "not_started"
    if anchor <= event_index + lag_max:
        return "partially_observed"
    return "observed"
