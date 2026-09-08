from __future__ import annotations


def render_event_narrative(event: dict, verification: dict | None = None, causal: bool = False) -> str:
    verification = verification or {}
    direction = verification.get("corrected_direction", event.get("expected_direction", "uncertain"))
    lag_min = verification.get("corrected_lag_min", event.get("candidate_lag_min", 0))
    lag_max = verification.get("corrected_lag_max", event.get("candidate_lag_max", 0))
    score = verification.get("verification_score", event.get("extraction_confidence", 0.0))
    causal_text = "has an accepted historical lagged association with the target" if causal and verification.get("supporting_graph_edge") else "has no accepted causal-association edge"
    return (
        f"Event fact: A {event.get('event_type')} event was reported on {event.get('report_time')}. "
        f"Target relevance: The event {causal_text}. "
        f"Expected direction: {direction}. "
        f"Estimated lag: {lag_min} to {lag_max} time steps. "
        f"Expected duration: {event.get('expected_duration', 0)} time steps. "
        f"Textual evidence: {event.get('evidence_span')}. "
        f"Numerical consistency: {verification.get('effect_status', 'unknown')}; observable={verification.get('observable', False)}. "
        f"Verification confidence: {float(score):.4f}."
    )


def render_timepoint_narrative(items: list[tuple[dict, dict]], top_k: int = 3, causal: bool = True) -> str:
    ranked = sorted(items, key=lambda x: x[1].get("verification_score", x[0].get("extraction_confidence", 0.0)), reverse=True)[:top_k]
    return " <EVENT_SEP> ".join(render_event_narrative(ev, ver, causal=causal) for ev, ver in ranked)
