import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_shuffle_within_split():
    manifest = json.loads((ROOT / "results" / "v6" / "p2" / "environment_shuffled_manifest.json").read_text(encoding="utf-8"))
    ranges = manifest["split_ranges"]
    for row in manifest["permutation_mapping"][:: max(1, len(manifest["permutation_mapping"]) // 1000)]:
        start, end = ranges[row["split"]]
        assert start <= row["target_row"] < end
        assert start <= row["source_row"] < end
