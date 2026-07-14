import pytest

from cen_ts.schemas import EventRecord


def test_event_schema_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        EventRecord.from_dict({}, source_row_id=1, source_text_hash="a" * 64, report_time="2020-01-01", event_index=0)
