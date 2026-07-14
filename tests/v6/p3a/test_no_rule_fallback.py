from cen_ts.llm_client import LLMRequestError


def test_api_failure_has_no_rule_or_raw_text_fallback(extractor_factory):
    extractor, _ = extractor_factory(error=LLMRequestError("forced"))
    source = "The agency announced a new rule."
    outcome = extractor.extract(source_row_id=1, source_text=source, report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert outcome.status == "failed"
    assert outcome.result is None
    assert outcome.raw_model_text is None
