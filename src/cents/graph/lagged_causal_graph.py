from __future__ import annotations

import pandas as pd

from cents.graph.causal_discovery import lagged_correlations


def build_event_variable_graph(df: pd.DataFrame, events: list[dict], target_variable: str = "OT") -> dict:
    nodes = [{"id": target_variable, "node_type": "variable"}]
    edges = lagged_correlations(df, max_lag=3, threshold=0.35)
    for event in events:
        event_id = event.get("event_id") or event.get("source_text_id")
        nodes.append({"id": event_id, "node_type": "event", "label": event.get("event_phrase", "")[:100]})
        score = float(event.get("final_consistency_score", event.get("confidence", 0.0)))
        edges.append({
            "source": event_id,
            "target": event.get("target_variable", target_variable),
            "edge_type": "event_to_var",
            "polarity": event.get("polarity", "uncertain"),
            "lag": [int(event.get("expected_lag_min", 0)), int(event.get("expected_lag_max", 0))],
            "confidence": float(event.get("confidence", 0.0)),
            "evidence_score": score,
        })
    return {"nodes": nodes, "edges": edges}

