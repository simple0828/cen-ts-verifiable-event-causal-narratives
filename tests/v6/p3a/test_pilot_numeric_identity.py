import json

import pandas as pd

from cen_ts.variant_builder import build_text_variant


def test_pilot_numeric_identity(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "OT": [1.0, 2.0], "prior_history_avg": [0.5, 1.5], "fact": ["a", "b"]}).to_csv(source, index=False)
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"source_row_id": 0, "event_fact": "event", "event_count": 1, "extraction_status": "success", "source_text_hash": "h", "prompt_version": "p_extract_v1"}) + "\n", encoding="utf-8")
    manifest = build_text_variant(source, tmp_path / "out.csv", "event", "event_fact", input_jsonl=events)
    assert manifest["non_text_columns_identical"] is True
