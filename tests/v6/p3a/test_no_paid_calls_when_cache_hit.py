def test_no_paid_calls_when_cache_hit(extractor_factory):
    extractor, fake = extractor_factory()
    kwargs = dict(source_row_id=7, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    extractor.extract(**kwargs)
    calls_before = extractor.ledger.read()["calls"]
    extractor.extract(**kwargs)
    assert extractor.ledger.read()["calls"] == calls_before
    assert fake.calls == 1
