from cen_tats.verifier.lag_observability import observable


def test_lag_observability() -> None:
    assert not observable(event_index=10, lag_min=3, anchor=11)
    assert observable(event_index=10, lag_min=1, anchor=11)
