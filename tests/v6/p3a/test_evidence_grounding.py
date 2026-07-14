from conftest import valid_model_payload


def test_evidence_grounding_exact(extractor_factory):
    extractor, _ = extractor_factory(valid_model_payload("agency announced"))
    outcome = extractor.extract(source_row_id=1, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert outcome.result.events[0].grounding_status == "exact"
