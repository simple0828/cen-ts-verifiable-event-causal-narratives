from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Direction = Literal["positive", "negative", "neutral", "uncertain"]
Intensity = Literal["weak", "moderate", "strong", "unknown"]
Factuality = Literal["observed", "forecast", "opinion", "rumor", "unknown"]


@dataclass(frozen=True)
class Event:
    event_id: str
    report_time: str
    event_time: str | None
    actor: str
    action: str
    object: str
    event_type: str
    affected_target: str
    expected_direction: Direction
    intensity: Intensity
    candidate_lag_min: int
    candidate_lag_max: int
    expected_duration: int
    factuality: Factuality
    source: str
    evidence_span: str
    extraction_confidence: float
    canonical_event_id: str | None = None
    independent_source_count: int = 1
    duplicate_count: int = 0
    time_index: int | None = None
    domain: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_EVENT_FIELDS = set(Event.__dataclass_fields__.keys()) - {"canonical_event_id", "independent_source_count", "duplicate_count", "time_index", "domain"}


def validate_event(event: dict, raw_text: str | None = None) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_EVENT_FIELDS:
        if field not in event:
            errors.append(f"missing:{field}")
    if event.get("expected_direction") not in {"positive", "negative", "neutral", "uncertain"}:
        errors.append("bad_direction")
    if event.get("intensity") not in {"weak", "moderate", "strong", "unknown"}:
        errors.append("bad_intensity")
    if event.get("factuality") not in {"observed", "forecast", "opinion", "rumor", "unknown"}:
        errors.append("bad_factuality")
    if raw_text is not None and event.get("evidence_span") and event["evidence_span"] not in raw_text:
        errors.append("evidence_not_grounded")
    return errors
