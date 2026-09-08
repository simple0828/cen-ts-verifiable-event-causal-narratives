from cen_tats.causal.graph_query import lag_windows_overlap


def test_lag_overlap() -> None:
    edge = {"lag_min": 1, "lag_max": 3, "duration": 2}
    assert lag_windows_overlap(event_index=10, edge=edge, anchor=10, horizon=3)
    assert not lag_windows_overlap(event_index=1, edge=edge, anchor=10, horizon=3)
