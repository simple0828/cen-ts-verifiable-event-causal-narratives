from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_api_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:6]}...{value[-3:]}"


@dataclass(frozen=True, repr=False)
class APIConfig:
    api_key: str
    base_url: str
    model: str
    max_calls: int = 80
    max_input_tokens: int = 400_000
    max_output_tokens: int = 120_000
    timeout: float = 120.0
    concurrency: int = 4
    api_style: str = "chat"

    @classmethod
    def from_env(cls) -> "APIConfig":
        required = {name: os.environ.get(name, "").strip() for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CEN_EXTRACTOR_MODEL")}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
        config = cls(
            api_key=required["OPENAI_API_KEY"],
            base_url=required["OPENAI_BASE_URL"].rstrip("/"),
            model=required["CEN_EXTRACTOR_MODEL"],
            max_calls=_positive_int_env("CEN_EXTRACTOR_MAX_CALLS", 80),
            max_input_tokens=_positive_int_env("CEN_EXTRACTOR_MAX_INPUT_TOKENS", 400_000),
            max_output_tokens=_positive_int_env("CEN_EXTRACTOR_MAX_OUTPUT_TOKENS", 120_000),
            timeout=_positive_float_env("CEN_EXTRACTOR_TIMEOUT", 120.0),
            concurrency=_positive_int_env("CEN_EXTRACTOR_CONCURRENCY", 4),
            api_style=os.environ.get("CEN_EXTRACTOR_API_STYLE", "chat").strip().lower(),
        )
        if config.api_style not in {"chat", "responses"}:
            raise ValueError("CEN_EXTRACTOR_API_STYLE must be chat or responses")
        return config

    @property
    def base_url_hash(self) -> str:
        return sha256_text(self.base_url)

    @property
    def config_hash(self) -> str:
        return sha256_text(json.dumps({"base_url": self.base_url, "model": self.model}, sort_keys=True))

    def safe_manifest(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "base_url_hash": self.base_url_hash,
            "model": self.model,
            "config_hash": self.config_hash,
            "api_key_present": bool(self.api_key),
            "max_calls": self.max_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
            "concurrency": self.concurrency,
            "api_style": self.api_style,
        }

    def __repr__(self) -> str:
        return f"APIConfig(base_url={self.base_url!r}, model={self.model!r}, api_key={redact_api_key(self.api_key)!r})"


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Process-safe-enough persistent budget ledger for sequential P3A scripts."""

    _lock = threading.Lock()

    def __init__(self, path: Path, config: APIConfig):
        self.path = Path(path)
        self.config = config

    def read(self) -> dict[str, int]:
        if not self.path.exists():
            return {"calls": 0, "sent_calls": 0, "successful_calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            reservations = int(raw.get("calls", 0))
            return {
                "calls": reservations,
                "sent_calls": int(raw.get("sent_calls", reservations)),
                "successful_calls": int(raw.get("successful_calls", raw.get("sent_calls", reservations))),
                "input_tokens": int(raw.get("input_tokens", 0)),
                "output_tokens": int(raw.get("output_tokens", 0)),
                "retries": int(raw.get("retries", 0)),
            }
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Budget ledger is invalid: {self.path}") from exc

    def reserve_call(self, *, is_retry: bool = False) -> int:
        with self._lock:
            state = self.read()
            if state["calls"] >= self.config.max_calls:
                raise BudgetExceeded(f"API call budget reached ({self.config.max_calls})")
            if state["input_tokens"] >= self.config.max_input_tokens:
                raise BudgetExceeded("API input token budget reached")
            if state["output_tokens"] >= self.config.max_output_tokens:
                raise BudgetExceeded("API output token budget reached")
            state["calls"] += 1
            if is_retry:
                state["retries"] += 1
            self._write(state)
            return state["calls"]

    def record_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        with self._lock:
            state = self.read()
            state["successful_calls"] += 1
            state["input_tokens"] += max(0, int(input_tokens or 0))
            state["output_tokens"] += max(0, int(output_tokens or 0))
            self._write(state)

    def mark_sent(self) -> None:
        with self._lock:
            state = self.read()
            state["sent_calls"] += 1
            self._write(state)

    def _write(self, state: dict[str, int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
