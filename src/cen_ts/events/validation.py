from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from datetime import datetime

from .schemas import EventRecord


DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")


def normalize_for_grounding(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned[:10], cleaned[:7], cleaned[:4]):
        try:
            if len(candidate) == 4:
                return datetime(int(candidate), 1, 1)
            if len(candidate) == 7:
                return datetime.fromisoformat(candidate + "-01")
            parsed = datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except (ValueError, TypeError):
            continue
    return None


def validate_event(event: EventRecord, source_text: str) -> EventRecord:
    warnings = list(event.warnings)
    if event.evidence_span in source_text:
        grounding = "exact"
    elif normalize_for_grounding(event.evidence_span) in normalize_for_grounding(source_text):
        grounding = "normalized_exact"
        warnings.append("evidence_whitespace_or_unicode_normalized")
    else:
        grounding = "unsupported"
        warnings.append("evidence_not_found_in_source")

    report_time = _parse_time(event.report_time)
    start_time = _parse_time(event.event_time_start)
    end_time = _parse_time(event.event_time_end)
    if event.event_time_start and start_time is None:
        temporal = "invalid"
        warnings.append("event_time_start_unparseable")
    elif event.event_time_end and end_time is None:
        temporal = "invalid"
        warnings.append("event_time_end_unparseable")
    elif start_time and end_time and start_time > end_time:
        temporal = "invalid"
        warnings.append("event_time_start_after_end")
    elif start_time is None:
        temporal = "undated"
    elif report_time is None:
        temporal = "undated"
        warnings.append("report_time_unparseable")
    elif start_time > report_time:
        temporal = "future"
        if event.factuality == "observed":
            temporal = "invalid"
            warnings.append("future_event_marked_observed")
    elif start_time.date() == report_time.date():
        temporal = "current"
    else:
        temporal = "past"

    evidence_dates = DATE_PATTERN.findall(event.evidence_span)
    if event.event_time_start and not evidence_dates:
        warnings.append("event_time_not_explicitly_dated_in_evidence")
    return replace(event, grounding_status=grounding, temporal_status=temporal, warnings=sorted(set(warnings)))


def validate_and_deduplicate(events: list[EventRecord], source_text: str) -> tuple[list[EventRecord], list[str]]:
    validated: list[EventRecord] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    evidence_counts: dict[str, int] = {}
    for event in events:
        checked = validate_event(event, source_text)
        key = (
            normalize_for_grounding(checked.evidence_span),
            normalize_for_grounding(checked.action).lower(),
            checked.event_type_raw.strip().lower(),
            checked.factuality,
        )
        if key in seen:
            warnings.append(f"duplicate_event_removed:{checked.event_id}")
            continue
        seen.add(key)
        evidence_key = normalize_for_grounding(checked.evidence_span)
        evidence_counts[evidence_key] = evidence_counts.get(evidence_key, 0) + 1
        validated.append(checked)
    if any(count > 1 for count in evidence_counts.values()):
        warnings.append("shared_evidence_for_distinct_events")
    return validated, sorted(set(warnings))


def narrative_eligible(event: EventRecord) -> bool:
    return event.grounding_status in {"exact", "normalized_exact"} and event.temporal_status != "invalid"
