from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.p3a_pipeline import load_config, split_bounds, stratified_unique_sample, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/p3a/event_extraction_pilot.local.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    frame = pd.read_csv(ROOT / config["dataset"]["source_csv"])
    bounds = split_bounds(len(frame), config)
    seed = int(config["pilot"]["seed"])
    train = stratified_unique_sample(
        frame, start=bounds["train_core"][0], end=bounds["train_core"][1],
        count=int(config["pilot"]["train_core_unique_texts"]), seed=seed,
    )
    for item in train:
        item["split"] = "train_core"
    prompt = stratified_unique_sample(
        frame, start=bounds["prompt_dev"][0], end=bounds["prompt_dev"][1],
        count=int(config["pilot"]["prompt_dev_unique_texts"]), seed=seed + 1,
        excluded_hashes={item["source_text_hash"] for item in train},
    )
    for item in prompt:
        item["split"] = "prompt_dev"
    sample = train + prompt
    forbidden = [item for item in sample if item["source_row_id"] >= bounds["official_validation"][0]]
    if forbidden:
        raise AssertionError("pilot sample includes official validation/test rows")
    write_jsonl(ROOT / "results/v6/p3a/pilot_sample.jsonl", sample)
    manifest = {
        "seed": seed,
        "requested": {"train_core": int(config["pilot"]["train_core_unique_texts"]), "prompt_dev": int(config["pilot"]["prompt_dev_unique_texts"])},
        "selected": {"train_core": len(train), "prompt_dev": len(prompt), "total": len(sample)},
        "all_source_text_hashes_unique": len({item["source_text_hash"] for item in sample}) == len(sample),
        "official_validation_rows": 0,
        "official_test_rows": 0,
        "coverage": {
            "length_bins": sorted({item.get("length_bin") for item in sample}),
            "date_segments": sorted({item["date_segment"] for item in sample}),
            "multiple_dates": sum(item["has_multiple_dates"] for item in sample),
            "forecast_language": sum(item["has_forecast_language"] for item in sample),
            "multiple_sources": sum(item["has_multiple_sources"] for item in sample),
            "background_candidates": sum(item["background_candidate"] for item in sample),
        },
    }
    write_json(ROOT / "results/v6/p3a/pilot_sample_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
