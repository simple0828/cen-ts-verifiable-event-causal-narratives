from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_KEY_FIELDS = (
    "provider", "base_url_hash", "model", "temperature", "prompt_hash", "schema_hash",
    "schema_version", "source_text_hash", "report_time", "target_description_hash",
)


def cache_key(metadata: dict[str, Any]) -> str:
    missing = [field for field in CACHE_KEY_FIELDS if field not in metadata]
    if missing:
        raise ValueError("Incomplete cache key; missing: " + ", ".join(missing))
    canonical = json.dumps({field: metadata[field] for field in CACHE_KEY_FIELDS}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CacheLookup:
    status: str
    key: str
    record: dict[str, Any] | None = None
    warning: str | None = None


class LLMCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, metadata: dict[str, Any], *, force_refresh: bool = False) -> CacheLookup:
        key = cache_key(metadata)
        path = self.path_for(key)
        if force_refresh:
            return CacheLookup("force_refresh", key)
        if not path.exists():
            return CacheLookup("miss", key)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("cache_key") != key or record.get("request_metadata") != {field: metadata[field] for field in CACHE_KEY_FIELDS}:
                raise ValueError("cache metadata mismatch")
            if not isinstance(record.get("raw_response"), str) or "parse_status" not in record:
                raise ValueError("cache payload incomplete")
            return CacheLookup("hit", key, record=record)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return CacheLookup("corrupt", key, warning=f"cache_corrupt:{type(exc).__name__}")

    def put(self, metadata: dict[str, Any], record: dict[str, Any]) -> Path:
        key = cache_key(metadata)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload["cache_key"] = key
        payload["request_metadata"] = {field: metadata[field] for field in CACHE_KEY_FIELDS}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temp.replace(path)
        return path
