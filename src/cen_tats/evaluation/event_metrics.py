from __future__ import annotations


def event_extraction_proxy_metrics(events: list[dict]) -> dict:
    if not events:
        return {"event_count": 0, "evidence_groundedness": 0.0, "hallucination_rate": 0.0}
    grounded = [bool(ev.get("evidence_span")) for ev in events]
    return {
        "event_count": len(events),
        "event_type_count": len(set(ev.get("event_type") for ev in events)),
        "evidence_groundedness": sum(grounded) / len(grounded),
        "hallucination_rate": 1.0 - sum(grounded) / len(grounded),
    }
