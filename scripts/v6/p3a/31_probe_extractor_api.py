from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.api_config import APIConfig
from cen_ts.p3a_pipeline import build_extractor, load_config, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/p3a/event_extraction_pilot.local.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    sample = [item for item in read_jsonl(ROOT / "results/v6/p3a/pilot_sample.jsonl") if item["split"] == "train_core"][:2]
    if len(sample) != 2:
        raise RuntimeError("Build the pilot sample before API probe")
    extractor = build_extractor(config)
    outcomes = []
    for item in sample:
        outcome = extractor.extract(
            source_row_id=int(item["source_row_id"]), source_text=item["source_text"], report_time=item["report_time"],
            dataset_name="Environment", domain="environment", target_description="Air Quality Index", force_refresh=False,
        )
        outcomes.append(outcome.to_dict(include_result=True, include_raw_text=False))
        if outcome.status != "success":
            break
    success = len(outcomes) == 2 and all(item["api_success"] and item["json_parse_success"] and item["schema_success"] for item in outcomes)
    api_config = APIConfig.from_env()
    payload = {
        "api_probe_success": success,
        "model": api_config.model,
        "safe_config": api_config.safe_manifest(),
        "sample_count": len(outcomes),
        "outcomes": outcomes,
        "token_usage_available": all(item.get("input_tokens") is not None and item.get("output_tokens") is not None for item in outcomes),
        "request_ids_available": all(bool(item.get("request_id")) for item in outcomes),
        "cache_writes": sum(item.get("cache_status") in {"miss", "corrupt", "force_refresh"} for item in outcomes),
    }
    write_json(ROOT / "results/v6/p3a/api_probe.json", payload)
    print(json.dumps({"api_probe_success": success, "model": api_config.model, "samples": len(outcomes)}, indent=2))
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
