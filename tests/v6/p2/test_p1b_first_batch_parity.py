import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_p1b_first_batch_parity():
    parity = json.loads((ROOT / "results" / "v6" / "p2" / "p1b_parity.json").read_text(encoding="utf-8"))
    assert parity["first_batch_numeric_parity"] is True
    assert parity["first_batch_token_parity"] is True
    assert parity["pooled_embedding_hash_equal"] is True
    assert parity["projected_text_hash_equal"] is True
    assert parity["combined_input_hash_equal"] is True
