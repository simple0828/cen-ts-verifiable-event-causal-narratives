from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from cents.data.preprocessing import clean_text
from cents.utils.io import read_jsonl, write_jsonl

POSITIVE = {
    "increase", "increased", "increasing", "rise", "rising", "rose", "gain", "growth",
    "higher", "recover", "improve", "surge", "above", "upward", "boost", "strong",
}
NEGATIVE = {
    "decrease", "decreased", "decreasing", "fall", "falling", "fell", "drop", "decline",
    "lower", "weak", "shortage", "risk", "loss", "below", "downward", "recession",
}
EVENT_TYPES = {
    "energy": ["oil", "gas", "gasoline", "energy", "crude", "fuel"],
    "health": ["health", "disease", "covid", "hospital", "death", "case"],
    "traffic": ["traffic", "road", "congestion", "vehicle", "transit"],
    "market": ["price", "market", "demand", "supply", "income", "economy"],
    "weather": ["weather", "storm", "rain", "temperature", "climate"],
    "policy": ["policy", "government", "regulation", "tax", "ban"],
}


def _stable_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def infer_polarity(text: str) -> str:
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    pos = len(tokens & POSITIVE)
    neg = len(tokens & NEGATIVE)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    if pos or neg:
        return "uncertain"
    return "neutral"


def infer_event_type(text: str, domain: str = "") -> str:
    lower = f"{domain} {text}".lower()
    for label, words in EVENT_TYPES.items():
        if any(word in lower for word in words):
            return label
    return "other"


def extract_rule_based_event(row: pd.Series, domain: str, target_variable: str, variables: list[str]) -> list[dict]:
    text = clean_text(row.get("text", "") or row.get("fact", ""))
    if len(text.split()) < 5:
        return []
    polarity = infer_polarity(text)
    event_type = infer_event_type(text, domain)
    confidence = 0.35
    if polarity in {"positive", "negative"}:
        confidence += 0.2
    if event_type != "other":
        confidence += 0.15
    phrase = text[:240]
    event_id = _stable_id(domain, row.get("date"), phrase)
    return [{
        "event_id": event_id,
        "time": str(row.get("date")),
        "source_text_id": event_id,
        "event_phrase": phrase,
        "event_type": event_type,
        "entities": [w for w in re.findall(r"\b[A-Z][A-Za-z0-9-]+\b", phrase)[:5]],
        "target_variable": target_variable if target_variable in variables else (variables[0] if variables else target_variable),
        "polarity": polarity,
        "expected_lag_min": 1,
        "expected_lag_max": 3,
        "magnitude": "medium" if confidence >= 0.6 else "weak",
        "confidence": float(min(confidence, 0.9)),
        "rationale": "Rule-based extraction from direction and domain keywords; use verifier before trusting.",
    }]


def extract_events_for_domain(
    df: pd.DataFrame,
    domain: str,
    target_variable: str,
    cache_path: str | Path | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    if cache_path and Path(cache_path).exists():
        return read_jsonl(cache_path)
    numeric_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = df.tail(max_rows).iterrows() if max_rows else df.iterrows()
    events: list[dict] = []
    for _, row in rows:
        events.extend(extract_rule_based_event(row, domain, target_variable, numeric_vars))
    if cache_path:
        write_jsonl(cache_path, events)
    return events


def events_to_frame(events: list[dict], dates: pd.Series, verifier_threshold: float | None = None) -> pd.DataFrame:
    date_index = pd.DataFrame({"date": pd.to_datetime(dates)})
    if not events:
        for col in ["event_count", "pos_events", "neg_events", "event_confidence", "verified_event_score"]:
            date_index[col] = 0.0
        return date_index
    ev = pd.DataFrame(events)
    ev["date"] = pd.to_datetime(ev["time"], errors="coerce")
    if verifier_threshold is not None and "final_consistency_score" in ev:
        ev = ev[ev["final_consistency_score"] >= verifier_threshold]
    ev["pos"] = (ev["polarity"] == "positive").astype(float)
    ev["neg"] = (ev["polarity"] == "negative").astype(float)
    ev["score"] = ev.get("final_consistency_score", ev["confidence"]).astype(float)
    agg = ev.groupby("date", as_index=False).agg(
        event_count=("event_id", "size"),
        pos_events=("pos", "sum"),
        neg_events=("neg", "sum"),
        event_confidence=("confidence", "mean"),
        verified_event_score=("score", "mean"),
    )
    out = date_index.merge(agg, on="date", how="left").fillna(0.0)
    return out

