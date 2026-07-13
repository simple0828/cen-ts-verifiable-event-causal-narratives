from cen_tats.events.schema import validate_event


def test_evidence_span_grounding() -> None:
    event = {
        "event_id": "e1",
        "report_time": "2020-01-01",
        "event_time": "2020-01-01",
        "actor": "A",
        "action": "B",
        "object": "OT",
        "event_type": "other",
        "affected_target": "OT",
        "expected_direction": "uncertain",
        "intensity": "weak",
        "candidate_lag_min": 1,
        "candidate_lag_max": 3,
        "expected_duration": 3,
        "factuality": "observed",
        "source": "unit",
        "evidence_span": "not in source",
        "extraction_confidence": 0.5,
    }
    assert "evidence_not_grounded" in validate_event(event, raw_text="source text")
