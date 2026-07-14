import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_constant_variant_fact_only():
    manifest = json.loads((ROOT / "results" / "v6" / "p2" / "environment_constant_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "constant"
    assert manifest["changed_columns"] == ["fact"]
    assert manifest["non_text_columns_identical"] is True
