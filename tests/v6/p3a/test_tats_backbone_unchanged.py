import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tats_backbone_unchanged():
    expected = json.loads((ROOT / "results/v6/p2/p1b_parity.json").read_text(encoding="utf-8"))["tats_cen_itransformer_sha256"]
    assert hashlib.sha256((ROOT / "tats_cen/models/iTransformer.py").read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / "third_party/TaTS/models/iTransformer.py").read_bytes()).hexdigest() == expected
