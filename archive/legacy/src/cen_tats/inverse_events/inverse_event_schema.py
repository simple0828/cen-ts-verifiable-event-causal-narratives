from __future__ import annotations

REQUIRED_INVERSE_FIELDS = {
    "inferred_event_type",
    "direction",
    "intensity",
    "pattern_type",
    "observable_start",
    "observable_duration",
    "confidence",
    "numeric_evidence",
    "alternative_hypotheses",
}


def validate_inverse_event(event: dict) -> list[str]:
    return [f"missing:{field}" for field in REQUIRED_INVERSE_FIELDS if field not in event]
