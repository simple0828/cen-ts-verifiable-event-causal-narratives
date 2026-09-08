from __future__ import annotations


def explanation_summary(events: list[dict], graph: dict | None = None) -> dict[str, float]:
    if not events:
        return {"event_consistency": 0.0, "avg_selected_events": 0.0, "avg_tokens": 0.0, "groundedness": 0.0}
    scores = [float(e.get("final_consistency_score", e.get("confidence", 0.0))) for e in events]
    avg_tokens = sum(len(str(e.get("rationale", "")).split()) for e in events) / len(events)
    grounded = 0.0
    if graph:
        sources = {edge.get("source") for edge in graph.get("edges", [])}
        grounded = sum(1 for e in events if e.get("event_id") in sources) / max(1, len(events))
    return {
        "event_consistency": float(sum(scores) / len(scores)),
        "avg_selected_events": float(len(events)),
        "avg_tokens": float(avg_tokens),
        "groundedness": float(grounded),
    }

