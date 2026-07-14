def test_cache_hit_no_new_call(extractor_factory):
    extractor, fake = extractor_factory()
    kwargs = dict(source_row_id=3, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    assert extractor.extract(**kwargs).status == "success"
    assert extractor.extract(**kwargs).cache_status == "hit"
    assert fake.calls == 1
