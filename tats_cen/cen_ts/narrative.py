from __future__ import annotations

from typing import Any

from .event_validation import narrative_eligible
from .schemas import EventExtractionResult, EventRecord, TextVariantRecord


DEFAULT_NO_EVENT_TEXT = "No material event was identified in the provided text."


class EventNarrativeRenderer:
    def __init__(self, *, max_events: int = 5, no_event_text: str = DEFAULT_NO_EVENT_TEXT, include_confidence: bool = False):
        self.max_events = max_events
        self.no_event_text = no_event_text
        self.include_confidence = include_confidence

    def render(self, result: EventExtractionResult) -> str:
        eligible = [event for event in result.events if narrative_eligible(event)][: self.max_events]
        if result.no_event or not eligible:
            return self.no_event_text
        return "\n\n".join(self.render_event(event) for event in eligible)

    def render_event(self, event: EventRecord) -> str:
        event_type = event.event_type_canonical or event.event_type_raw or "unknown"
        event_time = event.event_time_start or "unknown"
        actor = event.actor or "unknown"
        obj = event.object or "unknown"
        if event.candidate_lag_min is None or event.candidate_lag_max is None:
            lag = "unknown"
        else:
            lag = f"{event.candidate_lag_min}-{event.candidate_lag_max} time steps"
        lines = [
            f"Event type: {event_type}.",
            f"Event time: {event_time}.",
            f"Actor and action: {actor} {event.action} {obj}.",
            f"Factuality: {event.factuality}.",
            f"Expected direction: {event.expected_direction}.",
            f"Candidate lag: {lag}.",
            f"Evidence: {event.evidence_span}.",
        ]
        if self.include_confidence:
            lines.append(f"Extraction confidence: {event.extraction_confidence:.3f}.")
        return "\n".join(lines)


class NarrativeGenerator:
    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        raise NotImplementedError("NarrativeGenerator remains reserved; use deterministic EventNarrativeRenderer in P3A.")
