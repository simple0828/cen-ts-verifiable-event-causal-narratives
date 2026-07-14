from cen_ts.narrative import DEFAULT_NO_EVENT_TEXT, EventNarrativeRenderer
from cen_ts.p3a_pipeline import result_from_dict
from conftest import valid_model_payload


def test_unsupported_event_removed_from_narrative(extractor_factory):
    extractor, _ = extractor_factory(valid_model_payload("not in source"))
    outcome = extractor.extract(source_row_id=1, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert outcome.result.events[0].grounding_status == "unsupported"
    assert EventNarrativeRenderer().render(outcome.result) == DEFAULT_NO_EVENT_TEXT
