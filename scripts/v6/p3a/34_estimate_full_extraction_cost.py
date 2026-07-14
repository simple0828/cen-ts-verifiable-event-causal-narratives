from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tats_cen"))

from cen_ts.p3a_pipeline import load_config, normalize_text, split_bounds, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/p3a/event_extraction_pilot.local.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    frame = pd.read_csv(ROOT / config["dataset"]["source_csv"])
    bounds = split_bounds(len(frame), config)
    usage = json.loads((ROOT / "results/v6/p3a/api_usage.json").read_text(encoding="utf-8"))
    normalized = frame[config["dataset"]["text_column"]].map(normalize_text)
    cache_reuse = usage["cache_hits"] / max(1, usage["cache_hits"] + usage["cache_misses"])
    success_rate = usage["api_request_successes"] / max(1, usage["api_calls_total_ledger"])
    retry_rate = usage["api_retries_total_ledger"] / max(1, usage["api_calls_total_ledger"])
    avg_in = float(usage.get("average_input_tokens") or 0.0)
    avg_out = float(usage.get("average_output_tokens") or 0.0)
    avg_latency = float(usage.get("average_latency_seconds") or 0.0)
    cache_dir = ROOT / config["cache"]["directory"]
    cache_files = list(cache_dir.rglob("*.json")) if cache_dir.exists() else []
    avg_cache_bytes = sum(path.stat().st_size for path in cache_files) / max(1, len(cache_files))

    scopes = {
        "train_only": (0, bounds["prompt_dev"][1]),
        "train_plus_validation": (0, bounds["official_validation"][1]),
        "full_dataset": (0, len(frame)),
    }
    price_in = config.get("price_per_million_input_tokens")
    price_out = config.get("price_per_million_output_tokens")
    estimates = {}
    for name, (start, end) in scopes.items():
        unique = int(normalized.iloc[start:end][normalized.iloc[start:end].ne("")].nunique())
        unique_requests = math.ceil(unique * (1 - cache_reuse))
        calls = math.ceil(unique_requests * (1 + retry_rate) / max(success_rate, 1e-9))
        input_tokens = math.ceil(unique_requests * avg_in)
        output_tokens = math.ceil(unique_requests * avg_out)
        cost = "unavailable"
        if price_in is not None and price_out is not None:
            cost = input_tokens / 1_000_000 * float(price_in) + output_tokens / 1_000_000 * float(price_out)
        estimates[name] = {
            "normalized_unique_texts": unique,
            "estimated_unique_requests": unique_requests,
            "estimated_api_calls": calls,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_runtime_seconds_at_configured_concurrency": calls * avg_latency / max(1, int(config["extractor"]["concurrency"])),
            "estimated_cache_bytes": math.ceil(unique_requests * avg_cache_bytes),
            "estimated_cost": cost,
        }
    payload = {
        "basis": {"cache_reuse_rate": cache_reuse, "api_success_rate": success_rate, "retry_rate": retry_rate, "average_input_tokens": avg_in, "average_output_tokens": avg_out, "average_latency_seconds": avg_latency, "average_cache_bytes": avg_cache_bytes},
        "estimates": estimates,
        "pricing_source": "local_config" if price_in is not None and price_out is not None else "unavailable",
    }
    write_json(ROOT / "results/v6/p3a/full_extraction_estimate.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
