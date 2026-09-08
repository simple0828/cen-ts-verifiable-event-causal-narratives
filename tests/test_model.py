import hashlib
import json
from pathlib import Path

from cen_ts.models import forward_smoke


def test_itransformer_matches_pinned_upstream_hash():
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party/tats"
    expected = json.loads((source / "UPSTREAM.json").read_text(encoding="utf-8"))["model_blob_sha256"]["models/iTransformer.py"]
    actual = hashlib.sha256((source / "models/iTransformer.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert actual == expected


def test_itransformer_minimal_forward():
    assert forward_smoke() == (2, 48, 13)
