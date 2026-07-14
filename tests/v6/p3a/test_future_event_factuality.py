from dataclasses import replace

from cen_ts.event_validation import validate_event
from cen_ts.schemas import EventRecord
from conftest import valid_model_payload


def test_future_observed_event_is_invalid():
    payload = valid_model_payload()["events"][0]
    payload["event_time_start"] = "2021-01-01"
    payload["factuality"] = "observed"
    event = EventRecord.from_dict(payload, source_row_id=1, source_text_hash="a" * 64, report_time="2020-01-01", event_index=0)
    checked = validate_event(event, "The agency announced a new rule.")
    assert checked.temporal_status == "invalid"
    assert "future_event_marked_observed" in checked.warnings
