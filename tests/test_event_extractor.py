from __future__ import annotations

import pandas as pd

from cents.text.event_extractor import extract_rule_based_event


def test_rule_based_event_schema() -> None:
    row = pd.Series({"date": "2020-01-01", "text": "Gasoline prices increased because crude oil demand rose."})
    events = extract_rule_based_event(row, "Energy", "OT", ["OT"])
    assert events
    assert events[0]["polarity"] == "positive"
    assert events[0]["target_variable"] == "OT"

