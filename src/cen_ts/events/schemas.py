from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


EVENT_SCHEMA_VERSION = "event_schema_v1"
EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": EVENT_SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": ["events", "no_event", "no_event_reason"],
    "properties": {
        "events": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "event_time_start", "event_time_end", "actor", "action", "object", "location",
                    "event_type_raw", "event_type_canonical", "affected_target", "expected_direction",
                    "intensity", "candidate_lag_min", "candidate_lag_max", "expected_duration",
                    "factuality", "evidence_span", "source_name", "extraction_confidence",
                ],
                "properties": {
                    "event_time_start": {"type": ["string", "null"]},
                    "event_time_end": {"type": ["string", "null"]},
                    "actor": {"type": ["string", "null"]},
                    "action": {"type": "string", "minLength": 1},
                    "object": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "event_type_raw": {"type": "string", "minLength": 1},
                    "event_type_canonical": {"type": ["string", "null"]},
                    "affected_target": {"type": ["string", "null"]},
                    "expected_direction": {"enum": ["positive", "negative", "neutral", "uncertain", "unknown"]},
                    "intensity": {"enum": ["weak", "moderate", "strong", "unknown"]},
                    "candidate_lag_min": {"type": ["integer", "null"]},
                    "candidate_lag_max": {"type": ["integer", "null"]},
                    "expected_duration": {"type": ["integer", "null"], "minimum": 0},
                    "factuality": {"enum": ["observed", "announced", "forecast", "opinion", "hypothetical", "unknown"]},
                    "evidence_span": {"type": "string", "minLength": 1},
                    "source_name": {"type": ["string", "null"]},
                    "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "no_event": {"type": "boolean"},
        "no_event_reason": {"type": ["string", "null"]},
    },
}
EVENT_SCHEMA_JSON = json.dumps(EVENT_SCHEMA, ensure_ascii=False, sort_keys=True)
EXPECTED_DIRECTIONS = {"positive", "negative", "neutral", "uncertain", "unknown"}
INTENSITIES = {"weak", "moderate", "strong", "unknown"}
FACTUALITIES = {"observed", "announced", "forecast", "opinion", "hypothetical", "unknown"}
TEMPORAL_STATUSES = {"past", "current", "future", "undated", "invalid"}
GROUNDING_STATUSES = {"exact", "normalized_exact", "unsupported"}


def make_event_id(source_text_hash: str, event_index: int, schema_version: str = EVENT_SCHEMA_VERSION) -> str:
    material = f"{source_text_hash}:{event_index}:{schema_version}".encode("utf-8")
    return "evt_" + hashlib.sha256(material).hexdigest()[:24]


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    source_row_id: int
    source_text_hash: str
    report_time: str
    event_time_start: str | None
    event_time_end: str | None
    actor: str | None
    action: str
    object: str | None
    location: str | None
    event_type_raw: str
    event_type_canonical: str | None
    affected_target: str | None
    expected_direction: str
    intensity: str
    candidate_lag_min: int | None
    candidate_lag_max: int | None
    expected_duration: int | None
    factuality: str
    evidence_span: str
    source_name: str | None
    extraction_confidence: float
    temporal_status: str = "undated"
    grounding_status: str = "unsupported"
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.source_row_id, int) or isinstance(self.source_row_id, bool):
            raise TypeError("source_row_id must be an integer")
        if not self.event_id.strip() or not self.source_text_hash.strip() or not self.report_time.strip():
            raise ValueError("event_id, source_text_hash, and report_time must be non-empty")
        if not self.action.strip() or not self.event_type_raw.strip() or not self.evidence_span.strip():
            raise ValueError("action, event_type_raw, and evidence_span must be non-empty")
        if self.expected_direction not in EXPECTED_DIRECTIONS:
            raise ValueError(f"invalid expected_direction: {self.expected_direction}")
        if self.intensity not in INTENSITIES:
            raise ValueError(f"invalid intensity: {self.intensity}")
        if self.factuality not in FACTUALITIES:
            raise ValueError(f"invalid factuality: {self.factuality}")
        if self.temporal_status not in TEMPORAL_STATUSES:
            raise ValueError(f"invalid temporal_status: {self.temporal_status}")
        if self.grounding_status not in GROUNDING_STATUSES:
            raise ValueError(f"invalid grounding_status: {self.grounding_status}")
        if isinstance(self.extraction_confidence, bool) or not isinstance(self.extraction_confidence, (int, float)):
            raise TypeError("extraction_confidence must be numeric")
        if not 0.0 <= float(self.extraction_confidence) <= 1.0:
            raise ValueError("extraction_confidence must be between 0 and 1")
        if self.candidate_lag_min is not None and self.candidate_lag_max is not None:
            if self.candidate_lag_min > self.candidate_lag_max:
                raise ValueError("candidate_lag_min must be <= candidate_lag_max")
        if self.expected_duration is not None and self.expected_duration < 0:
            raise ValueError("expected_duration must be non-negative")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_row_id: int,
        source_text_hash: str,
        report_time: str,
        event_index: int,
        schema_version: str = EVENT_SCHEMA_VERSION,
    ) -> "EventRecord":
        required = {
            "event_time_start", "event_time_end", "actor", "action", "object", "location",
            "event_type_raw", "event_type_canonical", "affected_target", "expected_direction", "intensity",
            "candidate_lag_min", "candidate_lag_max", "expected_duration", "factuality", "evidence_span",
            "source_name", "extraction_confidence",
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing:
            raise ValueError("event fields missing: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError("unexpected event fields: " + ", ".join(sorted(unknown)))
        optional_strings = {
            "event_time_start", "event_time_end", "actor", "object", "location",
            "event_type_canonical", "affected_target", "source_name",
        }
        payload = dict(data)
        for name in optional_strings:
            value = payload.get(name)
            payload[name] = None if value is None or str(value).strip().lower() in {"", "null", "unknown"} else str(value).strip()
        payload["event_id"] = make_event_id(source_text_hash, event_index, schema_version)
        payload["source_row_id"] = source_row_id
        payload["source_text_hash"] = source_text_hash
        payload["report_time"] = report_time
        payload["action"] = str(payload.get("action", "")).strip()
        payload["event_type_raw"] = str(payload.get("event_type_raw", "")).strip()
        payload["evidence_span"] = str(payload.get("evidence_span", "")).strip()
        payload["expected_direction"] = str(payload.get("expected_direction", "unknown")).lower()
        payload["intensity"] = str(payload.get("intensity", "unknown")).lower()
        payload["factuality"] = str(payload.get("factuality", "unknown")).lower()
        payload["extraction_confidence"] = float(payload.get("extraction_confidence"))
        payload["candidate_lag_min"] = _optional_int(payload.get("candidate_lag_min"))
        payload["candidate_lag_max"] = _optional_int(payload.get("candidate_lag_max"))
        payload["expected_duration"] = _optional_int(payload.get("expected_duration"))
        payload["temporal_status"] = str(payload.get("temporal_status", "undated")).lower()
        payload["grounding_status"] = str(payload.get("grounding_status", "unsupported")).lower()
        warnings = payload.get("warnings", [])
        payload["warnings"] = [str(item) for item in warnings] if isinstance(warnings, list) else [str(warnings)]
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "unknown"}):
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid integer")
    return int(value)


@dataclass(frozen=True)
class EventExtractionResult:
    source_row_id: int
    source_text_hash: str
    report_time: str
    events: list[EventRecord]
    no_event: bool
    no_event_reason: str | None
    schema_version: str
    prompt_version: str
    model: str
    request_id: str | None
    raw_response_hash: str
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVENT_SCHEMA_VERSION}")
        if self.no_event and self.events:
            raise ValueError("no_event=true cannot contain events")
        if not self.no_event and not self.events:
            raise ValueError("no_event=false must contain at least one event")
        if any(event.source_row_id != self.source_row_id for event in self.events):
            raise ValueError("event source_row_id mismatch")
        if any(event.source_text_hash != self.source_text_hash for event in self.events):
            raise ValueError("event source_text_hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    relation: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InverseEventRecord:
    timestamp: str
    event_text: str
    inverse_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationRecord:
    timestamp: str
    candidate_text: str
    is_supported: bool
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NarrativeRecord:
    timestamp: str
    narrative_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    description: str = ""


@dataclass(frozen=True)
class TextVariantRecord:
    timestamp: str
    source_text: str
    text: str
    mode: str
    text_column: str
    metadata: dict[str, Any] = field(default_factory=dict)
