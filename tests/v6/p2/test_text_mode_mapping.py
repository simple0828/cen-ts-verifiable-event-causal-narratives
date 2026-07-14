import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tats_cen"))

from cen_ts.text_modes import default_text_column_for_mode, resolve_text_column


def test_text_mode_mapping():
    assert default_text_column_for_mode("raw") == "fact"
    assert default_text_column_for_mode("constant") == "constant_fact"
    assert default_text_column_for_mode("shuffled") == "shuffled_fact"
    assert default_text_column_for_mode("event") == "event_fact"
    assert default_text_column_for_mode("causal_event") == "causal_fact"
    assert default_text_column_for_mode("verified_event") == "verified_fact"
    assert default_text_column_for_mode("apo_event") == "apo_fact"
    assert resolve_text_column("event", "custom_event_text") == "custom_event_text"
