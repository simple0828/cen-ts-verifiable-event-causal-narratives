from __future__ import annotations


def build_causal_narrative(events: list[dict], graph: dict | None = None, top_k: int = 5) -> str:
    ranked = sorted(
        events,
        key=lambda e: float(e.get("final_consistency_score", e.get("confidence", 0.0))),
        reverse=True,
    )[:top_k]
    lines = ["Verified causal event narrative:"]
    if not ranked:
        lines.append("No sufficiently verified causal event was selected.")
    for idx, event in enumerate(ranked, start=1):
        lines.extend([
            f"{idx}. {event.get('event_phrase', '')[:180]}",
            f"   - Time: {event.get('time', '')}",
            f"   - Target: {event.get('target_variable', 'OT')}",
            f"   - Expected effect: {event.get('polarity', 'uncertain')}",
            f"   - Lag: {event.get('expected_lag_min', 0)}-{event.get('expected_lag_max', 0)}",
            f"   - Consistency score: {float(event.get('final_consistency_score', event.get('confidence', 0.0))):.3f}",
            f"   - Evidence: {event.get('rationale', '')}",
        ])
    lines.extend([
        "",
        "Forecasting instruction:",
        "Use only verified causal chains above as textual context. Do not rely on unverified or low-confidence events. If evidence is weak, state uncertainty.",
    ])
    return "\n".join(lines)

