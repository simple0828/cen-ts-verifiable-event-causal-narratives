from cen_ts.narrative import EventNarrativeRenderer


def test_narrative_deterministic(extractor_factory):
    extractor, _ = extractor_factory()
    outcome = extractor.extract(source_row_id=1, source_text="The agency announced a new rule.", report_time="2020-01-01", dataset_name="Environment", domain="environment", target_description="Air Quality Index")
    renderer = EventNarrativeRenderer()
    assert renderer.render(outcome.result) == renderer.render(outcome.result)
