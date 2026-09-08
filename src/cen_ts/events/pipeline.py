from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .api_config import APIConfig
from .extractor import EventExtractor
from .schemas import EventExtractionResult, EventRecord
from cen_ts.utils.paths import ROOT
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
FORECAST_PATTERN = re.compile(r"\b(?:forecast|expect(?:ed|s|ing)?|may|might|could|likely|project(?:ed|s|ing)?|predict(?:ed|s|ion)?)\b", re.I)
MULTI_SOURCE_PATTERN = re.compile(r"\b(?:according to|reported by|sources?|officials? said|researchers? said)\b|;", re.I)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["split"].get("use_model_val") or config["split"].get("use_test"):
        raise ValueError("Event extraction forbids validation and test text calls")
    if config["extractor"].get("external_knowledge_allowed") is not False:
        raise ValueError("Event extraction requires external_knowledge_allowed=false")
    return config


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).strip().replace("\r\n", "\n").replace("\r", "\n").split())


def source_text_hash(text: Any) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def split_bounds(n_rows: int, config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    train_end = int(n_rows * float(config["dataset"]["official_train_ratio"]))
    test_count = int(n_rows * float(config["dataset"]["official_test_ratio"]))
    test_start = n_rows - test_count
    train_core_end = int(train_end * float(config["split"]["train_core_ratio_within_train"]))
    return {
        "train_core": (0, train_core_end),
        "prompt_dev": (train_core_end, train_end),
        "official_validation": (train_end, test_start),
        "official_test": (test_start, n_rows),
    }


def split_for_row(row_id: int, bounds: dict[str, tuple[int, int]]) -> str:
    for name, (start, end) in bounds.items():
        if start <= row_id < end:
            return name
    raise ValueError(f"row outside split ranges: {row_id}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_extractor(config: dict[str, Any]) -> EventExtractor:
    api_config = APIConfig.from_env()
    return EventExtractor(
        api_config,
        cache_directory=ROOT / config["cache"]["directory"],
        ledger_path=ROOT / ".cache" / "cen_ts" / "api_budget_ledger.json",
        max_retries=int(config["extractor"]["max_retries"]),
        max_output_tokens=int(config["extractor"]["max_output_tokens"]),
    )


def result_from_dict(payload: dict[str, Any]) -> EventExtractionResult:
    return EventExtractionResult(
        source_row_id=int(payload["source_row_id"]),
        source_text_hash=str(payload["source_text_hash"]),
        report_time=str(payload["report_time"]),
        events=[EventRecord(**event) for event in payload.get("events", [])],
        no_event=bool(payload["no_event"]),
        no_event_reason=payload.get("no_event_reason"),
        schema_version=str(payload["schema_version"]),
        prompt_version=str(payload["prompt_version"]),
        model=str(payload["model"]),
        request_id=payload.get("request_id"),
        raw_response_hash=str(payload["raw_response_hash"]),
        warnings=list(payload.get("warnings", [])),
    )


def text_features(text: str, row_id: int, start: int, end: int) -> dict[str, Any]:
    normalized = normalize_text(text)
    relative = (row_id - start) / max(1, end - start)
    return {
        "char_length": len(normalized),
        "date_count": len(DATE_PATTERN.findall(normalized)),
        "has_multiple_dates": len(DATE_PATTERN.findall(normalized)) >= 2,
        "has_forecast_language": bool(FORECAST_PATTERN.search(normalized)),
        "has_multiple_sources": bool(MULTI_SOURCE_PATTERN.search(normalized)),
        "date_segment": "early" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "late",
        "background_candidate": bool(re.search(r"\b(?:is defined as|refers to|background|general information|no relevant information)\b", normalized, re.I)),
    }


def stratified_unique_sample(
    frame: pd.DataFrame,
    *,
    start: int,
    end: int,
    count: int,
    seed: int,
    excluded_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_hashes = set(excluded_hashes or set())
    unique: dict[str, dict[str, Any]] = {}
    for row_id in range(start, end):
        text = frame.at[row_id, "fact"]
        normalized = normalize_text(text)
        if not normalized:
            continue
        digest = source_text_hash(normalized)
        if digest in excluded_hashes or digest in unique:
            continue
        features = text_features(normalized, row_id, start, end)
        unique[digest] = {
            "source_row_id": row_id,
            "report_time": str(frame.at[row_id, "date"]),
            "source_text": str(text),
            "source_text_hash": digest,
            **features,
        }
    candidates = list(unique.values())
    if len(candidates) <= count:
        return sorted(candidates, key=lambda item: item["source_row_id"])
    lengths = pd.Series([item["char_length"] for item in candidates])
    q1, q2 = lengths.quantile([1 / 3, 2 / 3]).tolist()
    for item in candidates:
        item["length_bin"] = "short" if item["char_length"] <= q1 else "medium" if item["char_length"] <= q2 else "long"
    rng = random.Random(seed)
    rng.shuffle(candidates)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        labels = [
            f"length:{item['length_bin']}", f"date:{item['date_segment']}",
            "multi_date" if item["has_multiple_dates"] else "single_date",
            "forecast" if item["has_forecast_language"] else "non_forecast",
            "multi_source" if item["has_multiple_sources"] else "single_source",
            "background" if item["background_candidate"] else "event_candidate",
        ]
        for label in labels:
            buckets.setdefault(label, []).append(item)
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    labels = sorted(buckets)
    while len(selected) < count and labels:
        next_labels = []
        for label in labels:
            while buckets[label] and buckets[label][0]["source_text_hash"] in selected_hashes:
                buckets[label].pop(0)
            if buckets[label]:
                item = buckets[label].pop(0)
                selected.append(item)
                selected_hashes.add(item["source_text_hash"])
                if len(selected) >= count:
                    break
                next_labels.append(label)
        labels = next_labels
    if len(selected) < count:
        for item in candidates:
            if item["source_text_hash"] not in selected_hashes:
                selected.append(item)
                selected_hashes.add(item["source_text_hash"])
                if len(selected) >= count:
                    break
    return sorted(selected, key=lambda item: item["source_row_id"])
