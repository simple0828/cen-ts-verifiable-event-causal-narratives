from cen_tats.narratives.narrative_validator import validate_narrative
from cen_tats.narratives.renderer import render_event_narrative


def test_narrative_faithfulness() -> None:
    event = {"event_type": "supply_reduction", "report_time": "2020", "evidence_span": "Supply fell.", "expected_duration": 7}
    verifier = {"corrected_direction": "positive", "corrected_lag_min": 1, "corrected_lag_max": 7, "verification_score": 0.8}
    text = render_event_narrative(event, verifier, causal=True)
    assert validate_narrative(text, event, verifier) == []
