import json
from pathlib import Path


def test_p1_shuffle_mapping_stays_within_split() -> None:
    audit = json.loads(Path("results/v6/p1/environment_data_audit.json").read_text(encoding="utf-8"))
    mapping = json.loads(Path("results/v6/p1/shuffled_text_mapping.json").read_text(encoding="utf-8"))["mapping"]
    bounds = audit["split_boundaries"]
    for target_idx_text, item in list(mapping.items())[:500]:
        target_idx = int(target_idx_text)
        source_idx = int(item["source_text_index"])
        start, end = bounds[item["split"]]
        assert start <= target_idx < end
        assert start <= source_idx < end
