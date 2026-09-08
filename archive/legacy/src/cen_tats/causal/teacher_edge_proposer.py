from __future__ import annotations


def propose_edges(event_types: list[str], target: str, domain: str, max_lag: int = 7) -> list[dict]:
    proposals: list[dict] = []
    for event_type in sorted(set(event_types)):
        if event_type == "other":
            confidence = 0.35
            direction = "uncertain"
        elif any(k in event_type for k in ["reduction", "shortage", "conflict", "shock", "increase"]):
            confidence = 0.68
            direction = "positive" if domain == "Energy" else "uncertain"
        elif any(k in event_type for k in ["decrease", "demand_reduction"]):
            confidence = 0.62
            direction = "negative"
        else:
            confidence = 0.55
            direction = "uncertain"
        proposals.append(
            {
                "event_type": event_type,
                "target": target,
                "candidate_direction": direction,
                "candidate_lags": list(range(1, max_lag + 1)),
                "mechanism": f"Offline Augur-style teacher heuristic for {domain}; no official Augur code was available in this run.",
                "teacher_confidence": confidence,
                "teacher_source": "heuristic_offline_not_official_augur",
            }
        )
    return proposals
