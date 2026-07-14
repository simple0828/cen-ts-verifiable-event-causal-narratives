import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_no_validation_or_test_calls():
    manifest = json.loads((ROOT / "results/v6/p3a/split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_validation_calls"] == 0
    assert manifest["official_test_calls"] == 0
    sample = [json.loads(line) for line in (ROOT / "results/v6/p3a/pilot_sample.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert {item["split"] for item in sample} <= {"train_core", "prompt_dev"}
