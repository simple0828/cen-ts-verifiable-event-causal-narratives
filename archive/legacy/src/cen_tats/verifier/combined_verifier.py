from __future__ import annotations

from cen_tats.verifier.causal_verifier import causal_score
from cen_tats.verifier.cycle_verifier import cycle_score
from cen_tats.verifier.evidence_verifier import evidence_score
from cen_tats.verifier.lag_observability import effect_status, observable


def verify_event(event: dict, raw_text: str, graph: dict, inverse_event: dict, anchor: int, horizon: int, weights: dict, threshold: float) -> dict:
    s_evidence, evidence_reasons = evidence_score(event, raw_text)
    s_causal, edge, causal_reasons = causal_score(event, graph, anchor, horizon)
    lag_min = int(edge.get("lag_min", event.get("candidate_lag_min", 0))) if edge else int(event.get("candidate_lag_min", 0))
    lag_max = int(edge.get("lag_max", event.get("candidate_lag_max", lag_min))) if edge else int(event.get("candidate_lag_max", lag_min))
    event_index = int(event.get("time_index", anchor))
    obs = observable(event_index, lag_min, anchor)
    s_cycle, cycle_reasons = cycle_score(event, inverse_event)
    lag_fit = s_cycle if obs else 0.0
    score = (
        weights.get("evidence", 0.35) * s_evidence
        + weights.get("causal", 0.35) * s_causal
        + (weights.get("cycle", 0.2) * s_cycle if obs else 0.0)
        + (weights.get("lag_fit", 0.1) * lag_fit if obs else 0.0)
    )
    direction = edge.get("direction") if edge and edge.get("direction") in {"positive", "negative"} else event.get("expected_direction")
    return {
        "event_id": event["event_id"],
        "verified": bool(score >= threshold),
        "verification_score": float(score),
        "evidence_score": s_evidence,
        "causal_relevance": s_causal,
        "cycle_consistency": s_cycle,
        "lag_fit": lag_fit,
        "observable": obs,
        "corrected_direction": direction,
        "corrected_lag_min": lag_min,
        "corrected_lag_max": lag_max,
        "effect_status": effect_status(event_index, lag_min, lag_max, anchor),
        "decision_reason": evidence_reasons + causal_reasons + (cycle_reasons if obs else []),
        "supporting_graph_edge": edge.get("edge_id") if edge else None,
    }


def select_threshold(model_val_scores: list[float]) -> float:
    if not model_val_scores:
        return 0.5
    ordered = sorted(model_val_scores)
    return float(ordered[max(0, int(0.4 * len(ordered)) - 1)])
