from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cen_ts.utils.paths import project_path
from .extractor import EVENT_EXTRACTION_PROMPT
from .pipeline import build_extractor, load_config, split_bounds, stratified_unique_sample, write_jsonl
from .schemas import EVENT_SCHEMA_JSON


def dry_run(config_path: Path) -> dict:
    config = load_config(project_path(config_path))
    frame = pd.read_csv(project_path(config["dataset"]["source_csv"]), nrows=5)
    missing = {config["dataset"][key] for key in ("date_column", "text_column", "target_column")} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing configured columns: {sorted(missing)}")
    json.loads(EVENT_SCHEMA_JSON)
    required_tokens = {"{{source_text}}", "{{report_time}}", "{{schema_json}}"}
    if not all(token in EVENT_EXTRACTION_PROMPT for token in required_tokens):
        raise AssertionError("Embedded event prompt is incomplete")
    return {"status": "ok", "rows_checked": len(frame), "api_calls": 0, "columns": list(frame.columns)}


def extract(config_path: Path, output: Path, limit: int | None = None) -> dict:
    config = load_config(project_path(config_path))
    frame = pd.read_csv(project_path(config["dataset"]["source_csv"]))
    bounds = split_bounds(len(frame), config)
    sampling = config["sampling"]
    samples = stratified_unique_sample(
        frame,
        start=bounds["train_core"][0],
        end=bounds["train_core"][1],
        count=int(sampling["train_core_unique_texts"]),
        seed=int(sampling["seed"]),
    )
    if limit is not None:
        samples = samples[:limit]
    extractor = build_extractor(config)
    records = []
    failures = 0
    for item in samples:
        outcome = extractor.extract(
            source_row_id=int(item["source_row_id"]),
            source_text=str(item["source_text"]),
            report_time=str(item["report_time"]),
            dataset_name=str(config["dataset"]["name"]),
            domain="environment",
            target_description="Air Quality Index",
        )
        if outcome.result is None:
            failures += 1
        else:
            records.append(outcome.result.to_dict())
    write_jsonl(project_path(output), records)
    return {"requested": len(samples), "written": len(records), "failures": failures, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract grounded events from the training split.")
    parser.add_argument("--config", type=Path, default=Path("configs/event_extraction.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/events.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = dry_run(args.config) if args.dry_run else extract(args.config, args.output, args.limit)
    print(json.dumps(result, indent=2))
    return 0
