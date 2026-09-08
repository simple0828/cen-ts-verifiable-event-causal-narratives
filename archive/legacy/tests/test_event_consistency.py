from __future__ import annotations

import pandas as pd

from cents.verifier.event_consistency import score_event_consistency


def test_event_consistency_positive() -> None:
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "OT": [1, 1, 1, 1, 2, 3, 4, 4, 4, 4]})
    event = {"time": "2020-01-04", "polarity": "positive", "expected_lag_min": 1, "expected_lag_max": 3}
    scored = score_event_consistency(event, df, "OT")
    assert scored["final_consistency_score"] > 0

