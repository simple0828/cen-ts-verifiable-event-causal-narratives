from __future__ import annotations

import numpy as np
import pandas as pd

from cen_tats.causal.event_series_builder import event_series
from cen_tats.causal.lagged_tests import distributed_lag_regression, event_study_bootstrap, granger_style_test, permutation_test
from cen_tats.causal.teacher_edge_proposer import propose_edges
from cen_tats.io_utils import stable_hash


def build_graph(df: pd.DataFrame, events: list[dict], train_indices: list[int] | np.ndarray, target: str, domain: str, max_lag: int = 7, min_support: int = 5) -> dict:
    train_indices = np.asarray(train_indices, dtype=int)
    y_all = df[target].astype(float).to_numpy()
    y = y_all[train_indices]
    train_events = [ev for ev in events if ev.get("time_index") in set(train_indices.tolist())]
    series_all = event_series(train_events, len(df), representation="confidence_signed")
    proposals = propose_edges(list(series_all.keys()), target, domain, max_lag=max_lag)
    edges: list[dict] = []
    for proposal in proposals:
        event_type = proposal["event_type"]
        values = series_all.get(event_type, np.zeros(len(df)))[train_indices]
        support = int(np.sum(np.abs(values) > 1e-12))
        lag_max = min(max_lag, max(proposal["candidate_lags"] or [1]))
        dl = distributed_lag_regression(values, y, lag_max)
        gr = granger_style_test(values, y, lag_max)
        es = event_study_bootstrap(values, y, 1, lag_max)
        perm = permutation_test(values, y, lag_max)
        stat_conf = float(max(0.0, min(1.0, dl["delta_r2"] * 10 + (1.0 - dl["p_value"]) * 0.3)))
        direction = dl["coef_sign"] if dl["coef_sign"] != "neutral" else proposal["candidate_direction"]
        serious_conflict = proposal["candidate_direction"] in {"positive", "negative"} and direction in {"positive", "negative"} and direction != proposal["candidate_direction"]
        accepted = bool(
            support >= min_support
            and (dl["p_value"] < 0.10 or es["ci_low"] * es["ci_high"] > 0)
            and es["direction_stability"] >= 0.55
            and perm["permutation_p_value"] <= 0.30
            and not serious_conflict
        )
        edge = {
            "edge_id": stable_hash({"event_type": event_type, "target": target, "domain": domain}),
            "event_type": event_type,
            "target": target,
            "direction": direction,
            "lag_min": 1,
            "lag_max": lag_max,
            "duration": lag_max,
            "teacher_confidence": proposal["teacher_confidence"],
            "statistical_confidence": stat_conf,
            "bootstrap_sign_stability": es["direction_stability"],
            "support_count": support,
            "accepted": accepted,
            "split_used": "train_core",
            "evidence": {
                "distributed_lag": dl,
                "granger": gr,
                "event_study": es,
                "permutation": perm,
                "teacher": proposal,
            },
        }
        edges.append(edge)
    return {
        "graph_id": stable_hash({"domain": domain, "target": target, "edges": edges}),
        "dataset": domain,
        "target": target,
        "split_used": "train_core",
        "edge_count": len(edges),
        "accepted_edge_count": int(sum(e["accepted"] for e in edges)),
        "edges": edges,
    }
