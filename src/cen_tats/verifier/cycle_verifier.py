from __future__ import annotations


def cycle_score(event: dict, inverse_event: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    ev_dir = event.get("expected_direction", "uncertain")
    inv_dir = inverse_event.get("direction", "neutral")
    if ev_dir == "uncertain" or inv_dir == "neutral" or ev_dir == inv_dir:
        score += 0.45
    else:
        reasons.append("direction_mismatch")
    if event.get("intensity") == inverse_event.get("intensity") or inverse_event.get("intensity") == "weak":
        score += 0.2
    else:
        reasons.append("intensity_mismatch")
    if event.get("event_type", "").split("_")[0] in inverse_event.get("inferred_event_type", ""):
        score += 0.2
    else:
        reasons.append("type_abstract_mismatch")
    score += 0.15 * float(inverse_event.get("confidence", 0.0))
    return float(min(1.0, score)), reasons
