from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.api_config import APIConfig
from cen_ts.event_extractor import ExtractionOutcome
from cen_ts.narrative import EventNarrativeRenderer
from cen_ts.p3a_pipeline import build_extractor, load_config, normalize_text, read_jsonl, result_from_dict, write_json, write_jsonl
from cen_ts.variant_builder import build_text_variant


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def extraction_metrics(outcomes: list[dict[str, Any]], ledger: dict[str, int]) -> dict[str, Any]:
    attempted = len(outcomes)
    api_success = [item for item in outcomes if item["api_success"]]
    schema_success = [item for item in outcomes if item["schema_success"] and item.get("result")]
    results = [item["result"] for item in schema_success]
    events = [event for result in results for event in result["events"]]
    grounding = Counter(event["grounding_status"] for event in events)
    temporal_valid = sum(event["temporal_status"] != "invalid" for event in events)
    latencies = [float(item["latency_seconds"]) for item in outcomes if item.get("latency_seconds") is not None and item["cache_status"] != "hit"]
    input_tokens = [int(item["input_tokens"]) for item in outcomes if item.get("input_tokens") is not None and item["cache_status"] != "hit"]
    output_tokens = [int(item["output_tokens"]) for item in outcomes if item.get("output_tokens") is not None and item["cache_status"] != "hit"]
    failure_types = Counter(item.get("failure_type") or "none" for item in outcomes if item["status"] != "success")
    event_counts = [len(result["events"]) for result in results]
    return {
        "attempted_texts": attempted,
        "api_request_successes": ledger["successful_calls"],
        "primary_api_or_cache_successes": len(api_success),
        "cache_hits": sum(item["cache_status"] == "hit" for item in outcomes),
        "cache_misses": sum(item["cache_status"] != "hit" for item in outcomes),
        "api_calls_total_ledger": ledger["sent_calls"],
        "api_budget_reservations_total_ledger": ledger["calls"],
        "api_retries_total_ledger": ledger["retries"],
        "json_parse_rate": sum(item["json_parse_success"] for item in api_success) / max(1, len(api_success)),
        "schema_valid_rate": len(schema_success) / max(1, len(api_success)),
        "grounding_exact_rate": grounding["exact"] / max(1, len(events)),
        "grounding_normalized_exact_rate": grounding["normalized_exact"] / max(1, len(events)),
        "grounding_valid_rate": (grounding["exact"] + grounding["normalized_exact"]) / max(1, len(events)),
        "unsupported_evidence_rate": grounding["unsupported"] / max(1, len(events)),
        "temporal_valid_rate": temporal_valid / max(1, len(events)),
        "no_event_rate": sum(result["no_event"] for result in results) / max(1, len(results)),
        "average_events_per_text": statistics.mean(event_counts) if event_counts else 0.0,
        "event_count_distribution": dict(sorted(Counter(event_counts).items())),
        "forecast_rate": sum(event["factuality"] == "forecast" for event in events) / max(1, len(events)),
        "observed_rate": sum(event["factuality"] == "observed" for event in events) / max(1, len(events)),
        "unknown_direction_rate": sum(event["expected_direction"] == "unknown" for event in events) / max(1, len(events)),
        "unknown_lag_rate": sum(event["candidate_lag_min"] is None or event["candidate_lag_max"] is None for event in events) / max(1, len(events)),
        "average_confidence": statistics.mean([float(event["extraction_confidence"]) for event in events]) if events else 0.0,
        "low_confidence_rate": sum(float(event["extraction_confidence"]) < 0.6 for event in events) / max(1, len(events)),
        "average_input_tokens": statistics.mean(input_tokens) if input_tokens else None,
        "average_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "total_input_tokens": ledger["input_tokens"],
        "total_output_tokens": ledger["output_tokens"],
        "average_latency_seconds": statistics.mean(latencies) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.5),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "failure_type_distribution": dict(failure_types),
        "actual_cost": sum(float(item["actual_cost"]) for item in outcomes if item.get("actual_cost") is not None) if any(item.get("actual_cost") is not None for item in outcomes) else "unavailable",
    }


def normalized_set(values: list[str]) -> set[str]:
    return {normalize_text(value).lower() for value in values if normalize_text(value)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def stability_metrics(pairs: list[tuple[dict[str, Any], dict[str, Any]]], requested: int) -> dict[str, Any]:
    comparisons = []
    for first, second in pairs:
        a, b = first["result"], second["result"]
        a_events, b_events = a["events"], b["events"]
        comparisons.append({
            "source_row_id": first["source_row_id"],
            "event_count_match": len(a_events) == len(b_events),
            "event_type_raw_jaccard": jaccard(normalized_set([event["event_type_raw"] for event in a_events]), normalized_set([event["event_type_raw"] for event in b_events])),
            "action_normalized_match": normalized_set([event["action"] for event in a_events]) == normalized_set([event["action"] for event in b_events]),
            "evidence_span_overlap": jaccard(normalized_set([event["evidence_span"] for event in a_events]), normalized_set([event["evidence_span"] for event in b_events])),
            "factuality_match": [event["factuality"] for event in a_events] == [event["factuality"] for event in b_events],
            "direction_match": [event["expected_direction"] for event in a_events] == [event["expected_direction"] for event in b_events],
            "no_event_match": a["no_event"] == b["no_event"],
        })
    def avg(name: str) -> float:
        return sum(float(item[name]) for item in comparisons) / max(1, len(comparisons))
    return {
        "requested_samples": requested,
        "completed_pairs": len(comparisons),
        "event_count_consistency_rate": avg("event_count_match"),
        "event_type_raw_jaccard": avg("event_type_raw_jaccard"),
        "action_normalized_match_rate": avg("action_normalized_match"),
        "evidence_span_overlap": avg("evidence_span_overlap"),
        "factuality_consistency_rate": avg("factuality_match"),
        "direction_consistency_rate": avg("direction_match"),
        "no_event_consistency_rate": avg("no_event_match"),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/p3a/event_extraction_pilot.local.yaml")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    probe = json.loads((ROOT / "results/v6/p3a/api_probe.json").read_text(encoding="utf-8"))
    if not probe.get("api_probe_success"):
        raise RuntimeError("API probe failed; refusing subsequent paid calls")
    sample = read_jsonl(ROOT / "results/v6/p3a/pilot_sample.jsonl")
    if not sample or any(item["split"] not in {"train_core", "prompt_dev"} for item in sample):
        raise RuntimeError("pilot sample is missing or contains forbidden splits")
    api_config = APIConfig.from_env()
    extractor = build_extractor(config)
    ledger_before = extractor.ledger.read()
    stability_requested = min(int(config["pilot"]["stability_repeat_samples"]), 20)
    reserve_for_repair_or_retry = 6
    available_new = max(0, api_config.max_calls - ledger_before["calls"] - stability_requested - reserve_for_repair_or_retry)
    probe_rows = {int(item["source_row_id"]) for item in probe.get("outcomes", [])}
    train = [item for item in sample if item["split"] == "train_core"]
    prompt_dev = [item for item in sample if item["split"] == "prompt_dev"]
    desired_total = min(len(sample), available_new + len(probe_rows))
    prompt_count = min(len(prompt_dev), max(1, round(desired_total * 0.2))) if desired_total else 0
    train_count = min(len(train), desired_total - prompt_count)
    selected = train[:train_count] + prompt_dev[:prompt_count]
    if len(selected) < desired_total:
        remaining_ids = {item["source_row_id"] for item in selected}
        selected += [item for item in sample if item["source_row_id"] not in remaining_ids][: desired_total - len(selected)]

    def extract_item(item: dict[str, Any], force_refresh: bool) -> ExtractionOutcome:
        return extractor.extract(
            source_row_id=int(item["source_row_id"]), source_text=item["source_text"], report_time=item["report_time"],
            dataset_name="Environment", domain="environment", target_description="Air Quality Index", force_refresh=force_refresh,
        )

    outcomes_obj: list[ExtractionOutcome] = []
    with ThreadPoolExecutor(max_workers=min(api_config.concurrency, int(config["extractor"]["concurrency"]))) as pool:
        future_to_item = {pool.submit(extract_item, item, args.force_refresh): item for item in selected}
        for future in as_completed(future_to_item):
            outcomes_obj.append(future.result())
    outcomes_obj.sort(key=lambda item: item.source_row_id)
    raw_records = [item.to_dict(include_result=True, include_raw_text=True) for item in outcomes_obj]
    records = [item.to_dict(include_result=True, include_raw_text=False) for item in outcomes_obj]
    write_jsonl(ROOT / "results/v6/p3a/pilot_extractions_raw.jsonl", raw_records)
    write_jsonl(ROOT / "results/v6/p3a/pilot_extractions_validated.jsonl", records)
    failures = [item for item in records if item["status"] != "success"]
    write_jsonl(ROOT / "results/v6/p3a/pilot_failures.jsonl", failures)
    write_jsonl(ROOT / "results/v6/p3a/retry_queue.jsonl", [{"source_row_id": item["source_row_id"], "source_text_hash": item["source_text_hash"], "failure_type": item["failure_type"]} for item in failures])

    primary_by_row = {item["source_row_id"]: item for item in records if item["status"] == "success" and item.get("result")}
    stability_candidates = [item for item in train if item["source_row_id"] in primary_by_row][:stability_requested]
    repeat_records: list[dict[str, Any]] = []
    for item in stability_candidates:
        outcome = extract_item(item, True)
        if outcome.status == "success" and outcome.result:
            repeat_records.append(outcome.to_dict(include_result=True, include_raw_text=False))
        if extractor.ledger.read()["calls"] >= api_config.max_calls:
            break
    repeat_by_row = {item["source_row_id"]: item for item in repeat_records}
    pairs = [(primary_by_row[row_id], repeat_by_row[row_id]) for row_id in sorted(primary_by_row.keys() & repeat_by_row.keys())]
    stability = stability_metrics(pairs, stability_requested)
    write_json(ROOT / "results/v6/p3a/extraction_stability.json", stability)

    rendering = config["rendering"]
    renderer = EventNarrativeRenderer(
        max_events=int(rendering["max_events_per_timestamp"]), no_event_text=str(rendering["no_event_text"]),
        include_confidence=bool(rendering.get("include_confidence", False)),
    )
    narratives = []
    event_input = []
    for item in records:
        if item["status"] == "success" and item.get("result"):
            result = result_from_dict(item["result"])
            event_fact = renderer.render(result)
            narratives.append({"source_row_id": item["source_row_id"], "source_text_hash": item["source_text_hash"], "event_fact": event_fact, "event_count": len(result.events), "no_event": result.no_event, "prompt_version": result.prompt_version})
            event_input.append({"source_row_id": item["source_row_id"], "source_text_hash": item["source_text_hash"], "event_fact": event_fact, "event_count": len(result.events), "extraction_status": "success", "prompt_version": result.prompt_version})
        else:
            event_input.append({"source_row_id": item["source_row_id"], "source_text_hash": item["source_text_hash"], "event_fact": "", "event_count": 0, "extraction_status": "failed", "prompt_version": "p_extract_v1"})
    write_jsonl(ROOT / "results/v6/p3a/pilot_event_narratives.jsonl", narratives)
    event_input_path = ROOT / "results/v6/p3a/pilot_variant_input.jsonl"
    write_jsonl(event_input_path, event_input)
    variant_manifest = build_text_variant(
        source_csv=ROOT / config["dataset"]["source_csv"], output_csv=ROOT / "data/v6/p3a/Environment_event_pilot.csv",
        mode="event", text_column="event_fact", manifest_path=ROOT / "results/v6/p3a/pilot_csv_manifest.json", input_jsonl=event_input_path,
    )

    frame = pd.read_csv(ROOT / config["dataset"]["source_csv"])
    source_by_row = {int(item["source_row_id"]): item for item in sample}
    narrative_by_row = {int(item["source_row_id"]): item for item in narratives}
    sorted_success = [item for item in records if item["status"] == "success" and item.get("result")]
    high = sorted(sorted_success, key=lambda item: max([event["extraction_confidence"] for event in item["result"]["events"]] or [1.0]), reverse=True)[:10]
    low = sorted(sorted_success, key=lambda item: min([event["extraction_confidence"] for event in item["result"]["events"]] or [1.0]))[:10]
    multi = sorted(sorted_success, key=lambda item: len(item["result"]["events"]), reverse=True)[:10]
    boundary = failures[:10] + sorted(sorted_success, key=lambda item: source_by_row[item["source_row_id"]]["char_length"], reverse=True)[:10]
    chosen: list[tuple[str, dict[str, Any]]] = []
    used: set[int] = set()
    for label, group in (("high_confidence", high), ("low_confidence", low), ("multi_event", multi), ("failure_or_boundary", boundary)):
        added = 0
        for item in group:
            if item["source_row_id"] not in used:
                chosen.append((label, item)); used.add(item["source_row_id"]); added += 1
            if added >= 10:
                break
    for item in records:
        if len(chosen) >= 40:
            break
        if item["source_row_id"] not in used:
            chosen.append(("category_backfill", item)); used.add(item["source_row_id"])
    queue_path = ROOT / "results/v6/p3a/manual_annotation_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "date", "source_text", "extracted_events", "event_fact", "schema_valid", "grounding_valid", "temporal_valid", "review_event_count_correct", "review_evidence_correct", "review_factuality_correct", "review_missing_event", "review_hallucinated_event", "review_notes"]
    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, item in chosen[:40]:
            row_id = int(item["source_row_id"])
            result = item.get("result") or {}
            writer.writerow({
                "row_id": row_id, "date": frame.at[row_id, "date"], "source_text": frame.at[row_id, "fact"],
                "extracted_events": json.dumps(result.get("events", []), ensure_ascii=False),
                "event_fact": narrative_by_row.get(row_id, {}).get("event_fact", ""),
                "schema_valid": item["schema_success"], "grounding_valid": item["grounding_success"], "temporal_valid": item["temporal_success"],
                "review_event_count_correct": "", "review_evidence_correct": "", "review_factuality_correct": "",
                "review_missing_event": "", "review_hallucinated_event": "", "review_notes": "",
            })
    queue_ids = {item["source_row_id"] for _, item in chosen[:40]}
    actual_multi_ids = {item["source_row_id"] for item in sorted_success if len(item["result"]["events"]) >= 2}
    boundary_ids = {item["source_row_id"] for item in boundary}
    write_json(ROOT / "results/v6/p3a/manual_annotation_queue_manifest.json", {
        "queue_rows": min(40, len(chosen)),
        "unique_source_rows": len(queue_ids),
        "review_columns_left_blank": True,
        "requested_category_counts": {"high_confidence": 10, "low_confidence": 10, "multi_event": 10, "failure_or_boundary": 10},
        "available_or_selected_category_counts": {
            "high_confidence_candidates_in_queue": len(queue_ids & {item["source_row_id"] for item in high}),
            "relative_low_confidence_candidates_in_queue": len(queue_ids & {item["source_row_id"] for item in low}),
            "actual_multi_event_candidates_in_primary_pilot": len(queue_ids & actual_multi_ids),
            "failure_or_boundary_candidates_in_queue": len(queue_ids & boundary_ids),
        },
        "multi_event_shortfall": max(0, 10 - len(queue_ids & actual_multi_ids)),
        "note": "No single-event row is counted as a true multi-event example.",
    })

    ledger = extractor.ledger.read()
    metrics = extraction_metrics(records, ledger)
    metrics["stability"] = {key: value for key, value in stability.items() if key != "comparisons"}
    metrics["pilot_unique_texts"] = len({item["source_text_hash"] for item in records})
    metrics["selected_split_counts"] = dict(Counter(source_by_row[item["source_row_id"]]["split"] for item in records))
    metrics["official_validation_calls"] = 0
    metrics["official_test_calls"] = 0
    metrics["event_tats_training_runs"] = 0
    metrics["pilot_csv_numeric_identity"] = bool(variant_manifest["non_text_columns_identical"])
    write_json(ROOT / "results/v6/p3a/api_usage.json", metrics)
    write_json(ROOT / "results/v6/p3a/cache_statistics.json", {"cache_hits": metrics["cache_hits"], "cache_misses": metrics["cache_misses"], "cache_directory": config["cache"]["directory"], "corrupt_entries": sum(item["cache_status"] == "corrupt" for item in records)})
    print(json.dumps({"pilot_unique_texts": metrics["pilot_unique_texts"], "llm_calls": ledger["calls"], "cache_hits": metrics["cache_hits"], "failures": len(failures), "stability_pairs": stability["completed_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
