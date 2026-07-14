import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_tats_cen_initial_copy_manifest_matches_upstream():
    manifest = json.loads((ROOT / "results" / "v6" / "p2" / "tats_cen_initial_manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == "a053503674c61c54d101d01d47c9d680288a7c9a"
    assert manifest["content_matches_upstream"] is True
    assert manifest["copied_file_count"] > 0
