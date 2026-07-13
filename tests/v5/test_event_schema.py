from cen_tats.events.schema import validate_event


def test_event_schema_required_fields() -> None:
    event = {
        "event_id": "e1",
        "report_time": "2020-01-01T00:00:00",
        "event_time": "2020-01-01T00:00:00",
        "actor": "Energy",
        "action": "supply reduction",
        "object": "OT",
        "event_type": "supply_reduction",
        "affected_target": "OT",
        "expected_direction": "positive",
        "intensity": "moderate",
        "candidate_lag_min": 1,
        "candidate_lag_max": 7,
        "expected_duration": 7,
        "factuality": "observed",
        "source": "unit",
        "evidence_span": "Oil supply fell.",
        "extraction_confidence": 0.8,
    }
    assert validate_event(event, raw_text="Oil supply fell.") == []
