from __future__ import annotations

from cen_tats.causal.graph_query import find_edges, lag_windows_overlap


def causal_score(event: dict, graph: dict, anchor: int, horizon: int) -> tuple[float, dict | None, list[str]]:
    reasons: list[str] = []
    edges = find_edges(graph, event.get("event_type", ""), event.get("affected_target"), accepted_only=True)
    if not edges:
        return 0.0, None, ["no_accepted_graph_edge"]
    event_index = int(event.get("time_index", anchor))
    best_edge = None
    best_score = 0.0
    for edge in edges:
        overlap = lag_windows_overlap(event_index, edge, anchor, horizon)
        direction_ok = edge.get("direction") in {"uncertain", event.get("expected_direction")} or event.get("expected_direction") == "uncertain"
        score = 0.45 * float(overlap) + 0.25 * float(direction_ok) + 0.15 * float(edge.get("statistical_confidence", 0.0)) + 0.15 * float(edge.get("bootstrap_sign_stability", 0.0))
        if score > best_score:
            best_score = score
            best_edge = edge
    if best_edge is None:
        reasons.append("no_lag_overlap")
    return float(min(1.0, best_score)), best_edge, reasons
