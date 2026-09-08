from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api_config import APIConfig, BudgetExceeded, BudgetLedger, sha256_text
from .event_validation import validate_and_deduplicate
from .llm_cache import LLMCache
from .llm_client import LLMRequestError, LLMResponse, OpenAICompatibleClient
from .schemas import EVENT_SCHEMA_VERSION, EventExtractionResult, EventRecord, TextVariantRecord


PROMPT_VERSION = "p_extract_v1"


@dataclass(frozen=True)
class ExtractionOutcome:
    status: str
    source_row_id: int
    source_text_hash: str
    cache_status: str
    api_success: bool
    json_parse_success: bool
    schema_success: bool
    grounding_success: bool
    temporal_success: bool
    usable_event_count: int
    result: EventExtractionResult | None
    raw_model_text: str | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float | None
    retries: int
    actual_cost: float | None
    failure_type: str | None = None
    failure_message: str | None = None

    def to_dict(self, *, include_result: bool = True, include_raw_text: bool = False) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "source_row_id": self.source_row_id,
            "source_text_hash": self.source_text_hash,
            "cache_status": self.cache_status,
            "api_success": self.api_success,
            "json_parse_success": self.json_parse_success,
            "schema_success": self.schema_success,
            "grounding_success": self.grounding_success,
            "temporal_success": self.temporal_success,
            "usable_event_count": self.usable_event_count,
            "request_id": self.request_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "retries": self.retries,
            "actual_cost": self.actual_cost,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }
        if include_result:
            payload["result"] = self.result.to_dict() if self.result else None
        if include_raw_text:
            payload["raw_model_text"] = self.raw_model_text
        return payload


class EventExtractor:
    """Grounded API extractor. No-arg construction retains the P2 fail-closed contract."""

    def __init__(
        self,
        api_config: APIConfig | None = None,
        *,
        prompt_path: Path | None = None,
        schema_path: Path | None = None,
        cache_directory: Path | None = None,
        ledger_path: Path | None = None,
        max_retries: int = 3,
        max_output_tokens: int = 1800,
    ):
        self.api_config = api_config
        if api_config is None:
            return
        if not prompt_path or not schema_path or not cache_directory or not ledger_path:
            raise ValueError("prompt_path, schema_path, cache_directory, and ledger_path are required")
        self.prompt_path = Path(prompt_path)
        self.schema_path = Path(schema_path)
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self.schema_text = self.schema_path.read_text(encoding="utf-8")
        json.loads(self.schema_text)
        self.prompt_hash = sha256_text(self.prompt_template)
        self.schema_hash = sha256_text(self.schema_text)
        self.cache = LLMCache(Path(cache_directory))
        self.ledger = BudgetLedger(Path(ledger_path), api_config)
        self.client = OpenAICompatibleClient(api_config, self.ledger, max_retries=max_retries)
        self.max_output_tokens = max_output_tokens

    def transform(self, source_text: str, timestamp: str, context: dict[str, Any]) -> TextVariantRecord:
        if self.api_config is None:
            raise NotImplementedError("Event extraction requires the P3A API configuration.")
        outcome = self.extract(
            source_row_id=int(context["source_row_id"]),
            source_text=source_text,
            report_time=timestamp,
            dataset_name=str(context.get("dataset_name", "Environment")),
            domain=str(context.get("domain", "environment")),
            target_description=str(context.get("target_description", "")),
            force_refresh=bool(context.get("force_refresh", False)),
        )
        if not outcome.result:
            raise RuntimeError(f"event extraction failed: {outcome.failure_type}")
        return TextVariantRecord(
            timestamp=timestamp,
            source_text=source_text,
            text=json.dumps(outcome.result.to_dict(), ensure_ascii=False, sort_keys=True),
            mode="event",
            text_column=str(context.get("text_column", "event_fact")),
            metadata={"cache_status": outcome.cache_status},
        )

    def extract(
        self,
        *,
        source_row_id: int,
        source_text: str,
        report_time: str,
        dataset_name: str,
        domain: str,
        target_description: str,
        force_refresh: bool = False,
    ) -> ExtractionOutcome:
        if self.api_config is None:
            raise RuntimeError("P3A API configuration is not initialized")
        normalized_source = " ".join(source_text.strip().split())
        source_hash = sha256_text(normalized_source)
        metadata = {
            "provider": "openai_compatible",
            "base_url_hash": self.api_config.base_url_hash,
            "model": self.api_config.model,
            "temperature": 0.0,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "schema_version": EVENT_SCHEMA_VERSION,
            "source_text_hash": source_hash,
            "report_time": report_time,
            "target_description_hash": sha256_text(target_description),
        }
        lookup = self.cache.get(metadata, force_refresh=force_refresh)
        if lookup.status == "hit" and lookup.record:
            return self._outcome_from_cached(lookup.record, source_row_id, source_hash, report_time, source_text)

        prompt = self._render_prompt(
            dataset_name=dataset_name,
            domain=domain,
            target_description=target_description or "unknown",
            report_time=report_time,
            source_row_id=source_row_id,
            source_text=source_text,
        )
        cache_warning = [lookup.warning] if lookup.warning else []
        try:
            response = self.client.complete(prompt, max_output_tokens=self.max_output_tokens, temperature=0.0)
        except BudgetExceeded as exc:
            return self._failure(source_row_id, source_hash, lookup.status, "budget_exceeded", exc)
        except LLMRequestError as exc:
            return self._failure(source_row_id, source_hash, lookup.status, "api_error", exc)
        except Exception as exc:
            return self._failure(source_row_id, source_hash, lookup.status, "internal_client_error", exc)

        parsed, parse_error = self._parse_json(response.text)
        json_parse_success = parsed is not None
        result: EventExtractionResult | None = None
        schema_error: Exception | None = None
        if parsed is not None:
            try:
                result = self._build_result(parsed, source_row_id, source_hash, report_time, source_text, response, cache_warning)
            except Exception as exc:
                schema_error = exc
        else:
            schema_error = parse_error

        repair_response: LLMResponse | None = None
        if result is None:
            try:
                repair_prompt = self._repair_prompt(response.text, str(schema_error))
                repair_response = self.client.complete(repair_prompt, max_output_tokens=self.max_output_tokens, temperature=0.0)
                repaired, repair_error = self._parse_json(repair_response.text)
                if repaired is None:
                    raise repair_error or ValueError("repair JSON parse failed")
                result = self._build_result(repaired, source_row_id, source_hash, report_time, source_text, repair_response, cache_warning + ["json_repair_used"])
                response = repair_response
                json_parse_success = True
            except Exception as exc:
                schema_error = exc

        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "safe_config": self.api_config.safe_manifest(),
            "source_text": source_text,
            "raw_response": response.raw_body,
            "model_text": response.text,
            "parsed_response": result.to_dict() if result else parsed,
            "token_usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            "latency_seconds": response.latency_seconds,
            "request_id": response.request_id,
            "response_hash": hashlib.sha256(response.raw_body.encode("utf-8")).hexdigest(),
            "parse_status": "valid" if result else "failed",
            "failure_type": type(schema_error).__name__ if schema_error else None,
        }
        self.cache.put(metadata, record)
        if result is None:
            return ExtractionOutcome(
                status="failed", source_row_id=source_row_id, source_text_hash=source_hash, cache_status=lookup.status,
                api_success=True, json_parse_success=json_parse_success, schema_success=False, grounding_success=False,
                temporal_success=False, usable_event_count=0, result=None, raw_model_text=response.text,
                request_id=response.request_id, input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                latency_seconds=response.latency_seconds, retries=response.retries + (repair_response.retries if repair_response else 0),
                actual_cost=response.actual_cost, failure_type="schema_error", failure_message=_safe_error(schema_error),
            )
        return self._success_outcome(result, lookup.status, response)

    def _render_prompt(self, **values: Any) -> str:
        rendered = self.prompt_template
        for name, value in values.items():
            rendered = rendered.replace("{{" + name + "}}", str(value))
        return rendered.replace("{{schema_json}}", self.schema_text)

    def _repair_prompt(self, model_text: str, error: str) -> str:
        return (
            "Repair the following model output so it conforms exactly to the supplied JSON schema. "
            "Preserve only claims already present in the output; do not add facts. Return JSON only.\n"
            f"Schema:\n{self.schema_text}\nValidation error: {error[:500]}\nOutput:\n{model_text}"
        )

    @staticmethod
    def _parse_json(text: str) -> tuple[dict[str, Any] | None, Exception | None]:
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError("top-level JSON must be an object")
            return parsed, None
        except Exception as exc:
            return None, exc

    def _build_result(
        self,
        parsed: dict[str, Any],
        row_id: int,
        source_hash: str,
        report_time: str,
        source_text: str,
        response: LLMResponse,
        warnings: list[str],
    ) -> EventExtractionResult:
        required_top = {"events", "no_event", "no_event_reason"}
        if set(parsed) != required_top:
            raise ValueError(f"top-level fields must be exactly {sorted(required_top)}")
        no_event = parsed.get("no_event")
        if not isinstance(no_event, bool):
            raise TypeError("no_event must be boolean")
        raw_events = parsed.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > 5:
            raise ValueError("events must be a list with at most 5 items")
        events = [
            EventRecord.from_dict(item, source_row_id=row_id, source_text_hash=source_hash, report_time=report_time, event_index=index)
            for index, item in enumerate(raw_events)
            if isinstance(item, dict)
        ]
        if len(events) != len(raw_events):
            raise TypeError("every event must be an object")
        events, validation_warnings = validate_and_deduplicate(events, source_text)
        no_event_reason = parsed.get("no_event_reason")
        if no_event_reason is not None:
            no_event_reason = str(no_event_reason).strip() or None
        return EventExtractionResult(
            source_row_id=row_id,
            source_text_hash=source_hash,
            report_time=report_time,
            events=events,
            no_event=no_event,
            no_event_reason=no_event_reason,
            schema_version=EVENT_SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION,
            model=self.api_config.model,
            request_id=response.request_id,
            raw_response_hash=hashlib.sha256(response.raw_body.encode("utf-8")).hexdigest(),
            warnings=sorted(set(warnings + validation_warnings)),
        )

    def _outcome_from_cached(self, record: dict[str, Any], row_id: int, source_hash: str, report_time: str, source_text: str) -> ExtractionOutcome:
        parsed = record.get("parsed_response")
        if record.get("parse_status") != "valid" or not isinstance(parsed, dict):
            return self._failure(row_id, source_hash, "hit", "cached_invalid", ValueError("cached response is not valid"))
        events = [
            EventRecord(**event) for event in parsed.get("events", [])
        ]
        result = EventExtractionResult(
            source_row_id=row_id, source_text_hash=source_hash, report_time=report_time, events=events,
            no_event=bool(parsed["no_event"]), no_event_reason=parsed.get("no_event_reason"),
            schema_version=parsed["schema_version"], prompt_version=parsed["prompt_version"], model=parsed["model"],
            request_id=parsed.get("request_id"), raw_response_hash=parsed["raw_response_hash"], warnings=parsed.get("warnings", []),
        )
        usage = record.get("token_usage") or {}
        response = LLMResponse(
            text=record.get("model_text", ""), request_id=record.get("request_id"),
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
            latency_seconds=float(record.get("latency_seconds") or 0.0), raw_body=record.get("raw_response", ""), retries=0,
        )
        return self._success_outcome(result, "hit", response)

    @staticmethod
    def _success_outcome(result: EventExtractionResult, cache_status: str, response: LLMResponse) -> ExtractionOutcome:
        events = result.events
        grounding_success = all(event.grounding_status in {"exact", "normalized_exact"} for event in events)
        temporal_success = all(event.temporal_status != "invalid" for event in events)
        usable = sum(event.grounding_status != "unsupported" and event.temporal_status != "invalid" for event in events)
        return ExtractionOutcome(
            status="success", source_row_id=result.source_row_id, source_text_hash=result.source_text_hash,
            cache_status=cache_status, api_success=True, json_parse_success=True, schema_success=True,
            grounding_success=grounding_success, temporal_success=temporal_success, usable_event_count=usable,
            result=result, raw_model_text=response.text, request_id=response.request_id,
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds, retries=response.retries, actual_cost=response.actual_cost,
        )

    @staticmethod
    def _failure(row_id: int, source_hash: str, cache_status: str, failure_type: str, exc: Exception) -> ExtractionOutcome:
        return ExtractionOutcome(
            status="failed", source_row_id=row_id, source_text_hash=source_hash, cache_status=cache_status,
            api_success=False, json_parse_success=False, schema_success=False, grounding_success=False,
            temporal_success=False, usable_event_count=0, result=None, raw_model_text=None, request_id=None,
            input_tokens=None, output_tokens=None, latency_seconds=None, retries=0, actual_cost=None,
            failure_type=failure_type, failure_message=_safe_error(exc),
        )


def _safe_error(exc: Exception | None) -> str | None:
    if exc is None:
        return None
    return f"{type(exc).__name__}:{str(exc)[:500]}"
