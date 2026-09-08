import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tats_backbone_unchanged():
    expected = json.loads((ROOT / "vendor/tats/UPSTREAM.json").read_text(encoding="utf-8"))["model_blob_sha256"]["models/iTransformer.py"]
    for folder in ("vendor/tats", "third_party/TaTS"):
        source = (ROOT / folder / "models/iTransformer.py").read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(source).hexdigest() == expected
