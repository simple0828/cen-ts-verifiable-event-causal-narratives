import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_non_text_columns_identical_for_all_variants():
    for name in ["raw", "constant", "shuffled"]:
        manifest = json.loads((ROOT / "results" / "v6" / "p2" / f"environment_{name}_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_non_text_identity_sha256"] == manifest["output_non_text_identity_sha256"]
        assert manifest["non_text_columns_identical"] is True
