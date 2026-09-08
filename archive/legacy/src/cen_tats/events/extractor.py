from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from cen_tats.events.schema import Event
from cen_tats.io_utils import stable_hash

TYPE_RULES: list[tuple[str, list[str]]] = [
    ("supply_reduction", ["embargo", "shortage", "supply disruption", "production cut", "sanction", "outage"]),
    ("supply_increase", ["shale", "production", "output", "reserve", "drilling", "supply"]),
    ("demand_increase", ["demand growth", "increased demand", "consumption growth", "recovery"]),
    ("demand_reduction", ["recession", "slowdown", "weak demand", "lower demand", "decline in demand"]),
    ("price_increase", ["price increase", "prices rose", "higher prices", "rise in", "surge"]),
    ("price_decrease", ["price decrease", "prices fell", "lower prices", "decline", "drop"]),
    ("policy_regulation", ["policy", "regulation", "government", "tax", "standard"]),
    ("conflict_security", ["war", "conflict", "attack", "security", "military"]),
    ("climate_extreme", ["hurricane", "storm", "drought", "flood", "weather", "climate"]),
    ("health_shock", ["influenza", "pandemic", "virus", "disease", "health"]),
    ("economic_shock", ["inflation", "unemployment", "trade", "economy", "economic"]),
]

POSITIVE_WORDS = ["increase", "increased", "rise", "rose", "higher", "surge", "growth", "tight", "shortage"]
NEGATIVE_WORDS = ["decrease", "decline", "fell", "fall", "lower", "drop", "reduction", "weak", "slowdown"]
FORECAST_WORDS = ["may", "might", "could", "likely", "predicted", "forecast", "expected", "will", "would"]
OPINION_WORDS = ["opinion", "believe", "suggests", "argues"]
RUMOR_WORDS = ["rumor", "rumour", "unconfirmed"]


def _first_sentence(text: str, max_chars: int = 260) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return "No information available"
    parts = re.split(r"(?<=[.!?;])\s+", clean)
    span = parts[0].strip()
    if len(span) > max_chars:
        span = span[:max_chars].rsplit(" ", 1)[0].strip()
    return span or clean[:max_chars]


def classify_event_type(text: str) -> str:
    lower = text.lower()
    for event_type, needles in TYPE_RULES:
        if any(n in lower for n in needles):
            return event_type
    return "other"


def infer_direction(text: str) -> str:
    lower = text.lower()
    pos = sum(w in lower for w in POSITIVE_WORDS)
    neg = sum(w in lower for w in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "uncertain"


def infer_factuality(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in RUMOR_WORDS):
        return "rumor"
    if any(w in lower for w in OPINION_WORDS):
        return "opinion"
    if any(w in lower for w in FORECAST_WORDS):
        return "forecast"
    return "observed"


def extract_events_for_frame(df: pd.DataFrame, domain: str, target: str, text_col: str = "aux_text") -> list[dict]:
    events: list[dict] = []
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ""))
        if not text or text == "No information available":
            continue
        evidence = _first_sentence(text)
        event_type = classify_event_type(evidence)
        direction = infer_direction(evidence)
        factuality = infer_factuality(evidence)
        confidence = 0.55
        if event_type != "other":
            confidence += 0.15
        if direction != "uncertain":
            confidence += 0.1
        if factuality == "observed":
            confidence += 0.1
        intensity = "moderate" if any(w in evidence.lower() for w in ["surge", "sharp", "major", "quadrupling"]) else "weak"
        report_time = pd.to_datetime(row["date"]).isoformat()
        payload = {
            "report_time": report_time,
            "event_time": report_time,
            "actor": domain,
            "action": event_type.replace("_", " "),
            "object": target,
            "event_type": event_type,
            "affected_target": target,
            "expected_direction": direction,
            "intensity": intensity,
            "candidate_lag_min": 1,
            "candidate_lag_max": 7 if domain == "Energy" else 3,
            "expected_duration": 7 if domain == "Energy" else 3,
            "factuality": factuality,
            "source": "Time-MMD aligned text",
            "evidence_span": evidence,
            "extraction_confidence": round(min(confidence, 0.95), 4),
            "time_index": int(idx),
            "domain": domain,
        }
        event_id = stable_hash(payload)
        ev = Event(event_id=event_id, canonical_event_id=event_id, **payload)
        events.append(ev.to_dict())
    return events


def event_type_counts(events: list[dict]) -> Counter:
    return Counter(ev["event_type"] for ev in events)
