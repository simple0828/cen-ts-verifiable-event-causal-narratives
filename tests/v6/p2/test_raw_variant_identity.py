import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_raw_variant_non_text_identity():
    manifest = json.loads((ROOT / "results" / "v6" / "p2" / "environment_raw_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "raw"
    assert manifest["non_text_columns_identical"] is True
    assert set(manifest["changed_columns"]) <= {"fact"}
