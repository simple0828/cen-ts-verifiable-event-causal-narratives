import json
from pathlib import Path

from cen_ts.events.extractor import EVENT_EXTRACTION_PROMPT
from cen_ts.events.run import dry_run
from cen_ts.events.schemas import EVENT_SCHEMA_JSON, EventRecord
from cen_ts.events.validation import validate_event


def test_prompt_and_schema_are_embedded():
    assert "{{source_text}}" in EVENT_EXTRACTION_PROMPT
    assert json.loads(EVENT_SCHEMA_JSON)["additionalProperties"] is False
    assert not Path("prompts").exists()


def test_grounded_event_validation():
    event = EventRecord.from_dict(
        {
            "event_time_start": "2024-01-01", "event_time_end": None, "actor": "Agency",
            "action": "issued an alert", "object": None, "location": None,
            "event_type_raw": "alert", "event_type_canonical": None, "affected_target": None,
            "expected_direction": "unknown", "intensity": "unknown", "candidate_lag_min": None,
            "candidate_lag_max": None, "expected_duration": None, "factuality": "observed",
            "evidence_span": "Agency issued an alert", "source_name": None, "extraction_confidence": 0.9,
        },
        source_row_id=1, source_text_hash="abc", report_time="2024-01-02", event_index=0,
    )
    checked = validate_event(event, "Agency issued an alert on 2024-01-01.")
    assert checked.grounding_status == "exact"
    assert checked.temporal_status == "past"


def test_event_extraction_dry_run_makes_no_api_calls():
    result = dry_run(Path("configs/event_extraction.yaml"))
    assert result["status"] == "ok"
    assert result["api_calls"] == 0
