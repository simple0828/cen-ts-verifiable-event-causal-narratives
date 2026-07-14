from conftest import valid_model_payload


def test_lag_order_rejected(extractor_factory):
    payload = valid_model_payload()
    payload["events"][0]["candidate_lag_min"] = 5
    payload["events"][0]["candidate_lag_max"] = 2
    extractor, _ = extractor_factory(payload)
    outcome = extractor.extract(source_row_id=1, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert outcome.status == "failed"
