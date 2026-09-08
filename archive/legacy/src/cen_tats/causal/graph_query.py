from __future__ import annotations


def find_edges(graph: dict, event_type: str, target: str | None = None, accepted_only: bool = True) -> list[dict]:
    edges = []
    for edge in graph.get("edges", []):
        if edge.get("event_type") != event_type:
            continue
        if target is not None and edge.get("target") != target:
            continue
        if accepted_only and not edge.get("accepted"):
            continue
        edges.append(edge)
    return edges


def lag_windows_overlap(event_index: int, edge: dict, anchor: int, horizon: int) -> bool:
    impact_start = event_index + int(edge.get("lag_min", 0))
    impact_end = event_index + int(edge.get("lag_max", 0)) + int(edge.get("duration", 0))
    forecast_start = anchor + 1
    forecast_end = anchor + horizon
    return max(impact_start, forecast_start) <= min(impact_end, forecast_end)
