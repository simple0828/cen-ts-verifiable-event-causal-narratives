import pytest

from conftest import valid_model_payload


def test_confidence_range(extractor_factory):
    payload = valid_model_payload()
    payload["events"][0]["extraction_confidence"] = 1.1
    extractor, _ = extractor_factory(payload)
    outcome = extractor.extract(source_row_id=1, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert outcome.status == "failed"
    assert outcome.schema_success is False
