from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cen_ts.paths import DEFAULT_GPT2_PATH

from cen_ts.p3a_pipeline import DATE_PATTERN, FORECAST_PATTERN, MULTI_SOURCE_PATTERN, load_config, normalize_text, source_text_hash, split_bounds, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6/p3a/event_extraction_pilot.local.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    source = ROOT / config["dataset"]["source_csv"]
    frame = pd.read_csv(source)
    text_column = config["dataset"]["text_column"]
    date_column = config["dataset"]["date_column"]
    texts = frame[text_column]
    raw_nonempty = texts.fillna("").astype(str).str.strip()
    normalized = texts.map(normalize_text)
    nonempty = normalized[normalized.ne("")]
    bounds = split_bounds(len(frame), config)

    tokenizer_status = "loaded"
    token_counts: list[int] = []
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(DEFAULT_GPT2_PATH, local_files_only=True)
        token_counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in nonempty]
    except Exception as exc:
        tokenizer_status = f"unavailable:{type(exc).__name__}"

    split_hashes: dict[str, str] = {}
    split_sets: dict[str, set[str]] = {}
    split_manifest: dict[str, object] = {"row_count": len(frame), "splits": {}}
    for name, (start, end) in bounds.items():
        values = normalized.iloc[start:end].tolist()
        hashes = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values]
        split_hash = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
        split_hashes[name] = split_hash
        split_sets[name] = {value for value in values if value}
        split_manifest["splits"][name] = {
            "index_range": [start, end],
            "date_range": [str(frame.iloc[start][date_column]), str(frame.iloc[end - 1][date_column])],
            "text_hash": split_hash,
            "row_count": end - start,
        }
    split_manifest["split_hash"] = hashlib.sha256(json.dumps(split_manifest["splits"], sort_keys=True).encode("utf-8")).hexdigest()
    split_manifest["llm_allowed_splits"] = ["train_core", "prompt_dev"]
    split_manifest["official_validation_calls"] = 0
    split_manifest["official_test_calls"] = 0
    write_json(ROOT / "results/v6/p3a/split_manifest.json", split_manifest)

    date_counts = nonempty.map(lambda value: len(DATE_PATTERN.findall(value)))
    char_lengths = nonempty.str.len()
    audit = {
        "source_csv": str(source.relative_to(ROOT)),
        "total_rows": len(frame),
        "split_boundaries": {key: list(value) for key, value in bounds.items()},
        "text_nonempty_rate": float(normalized.ne("").mean()),
        "raw_unique_nonempty_texts": int(raw_nonempty[raw_nonempty.ne("")].nunique()),
        "normalized_unique_nonempty_texts": int(nonempty.nunique()),
        "exact_duplicate_rate_nonempty": float(1 - nonempty.nunique() / max(1, len(nonempty))),
        "average_characters": float(char_lengths.mean()),
        "average_gpt2_tokens": float(sum(token_counts) / len(token_counts)) if token_counts else None,
        "gpt2_tokenizer_status": tokenizer_status,
        "text_length_quantiles": {str(q): float(char_lengths.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)},
        "contains_date_rate": float((date_counts >= 1).mean()),
        "multiple_dates_rate": float((date_counts >= 2).mean()),
        "forecast_language_rate": float(nonempty.str.contains(FORECAST_PATTERN).mean()),
        "multiple_sources_rate": float(nonempty.str.contains(MULTI_SOURCE_PATTERN).mean()),
        "max_text_length": int(char_lengths.max()),
        "max_gpt2_tokens": max(token_counts) if token_counts else None,
        "truncation_risk_count_over_1800_gpt2_tokens": sum(count > 1800 for count in token_counts),
        "cross_split_exact_duplicate_texts": {
            f"{left}__{right}": len(split_sets[left] & split_sets[right])
            for index, left in enumerate(bounds) for right in list(bounds)[index + 1 :]
        },
        "target_semantics": {
            "status": "resolved", "target": "Air Quality Index", "unit": "AQI index points", "frequency": "daily",
            "source": "local Time-MMD DescriptionOfOT.png and numerical/Environment/Environment.csv",
        },
    }
    write_json(ROOT / "results/v6/p3a/input_audit.json", audit)
    report = f"""# P3A Environment Event Extraction Input Audit

- Rows: {audit['total_rows']}
- Official train / validation / test: `{bounds['train_core'][0]}..{bounds['prompt_dev'][1]}` / `{bounds['official_validation'][0]}..{bounds['official_validation'][1]}` / `{bounds['official_test'][0]}..{bounds['official_test'][1]}`
- P3A train_core / prompt_dev: `{bounds['train_core']}` / `{bounds['prompt_dev']}`
- Non-empty text rate: {audit['text_nonempty_rate']:.6f}
- Raw / normalized unique non-empty texts: {audit['raw_unique_nonempty_texts']} / {audit['normalized_unique_nonempty_texts']}
- Exact duplicate rate: {audit['exact_duplicate_rate_nonempty']:.6f}
- Average characters / GPT-2 tokens: {audit['average_characters']:.2f} / {audit['average_gpt2_tokens']}
- Max characters / GPT-2 tokens: {audit['max_text_length']} / {audit['max_gpt2_tokens']}
- Date / multiple-date rates: {audit['contains_date_rate']:.6f} / {audit['multiple_dates_rate']:.6f}
- Forecast-language rate: {audit['forecast_language_rate']:.6f}
- Multiple-source heuristic rate: {audit['multiple_sources_rate']:.6f}
- GPT-2 truncation-risk texts (>1800 tokens): {audit['truncation_risk_count_over_1800_gpt2_tokens']}
- Target semantics: resolved as daily Air Quality Index (AQI), from local Time-MMD materials.

Normalization is used only for deduplication and hashing. LLM requests retain original text, dates, numbers, entities, and casing.
"""
    out = ROOT / "reports/v6/p3a/input_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(json.dumps({"rows": len(frame), "normalized_unique_texts": audit["normalized_unique_nonempty_texts"], "split_hash": split_manifest["split_hash"]}, indent=2))


if __name__ == "__main__":
    main()
