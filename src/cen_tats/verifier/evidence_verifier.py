from __future__ import annotations


def evidence_score(event: dict, raw_text: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    span = str(event.get("evidence_span", ""))
    if span and span in raw_text:
        score += 0.45
    else:
        reasons.append("evidence_span_not_grounded")
    if event.get("factuality") == "observed":
        score += 0.25
    else:
        reasons.append(f"factuality_{event.get('factuality')}")
    if event.get("actor") and event.get("action") and event.get("object"):
        score += 0.15
    else:
        reasons.append("incomplete_actor_action_object")
    score += 0.15 * float(event.get("extraction_confidence", 0.0))
    return float(min(1.0, score)), reasons
