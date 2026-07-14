from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .api_config import APIConfig, BudgetLedger


@dataclass(frozen=True)
class LLMResponse:
    text: str
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    raw_body: str
    retries: int
    actual_cost: float | None = None


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, *, recoverable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.recoverable = recoverable
        self.status_code = status_code


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig, ledger: BudgetLedger, *, max_retries: int = 3):
        self.config = config
        self.ledger = ledger
        self.max_retries = max_retries

    def complete(self, prompt: str, *, max_output_tokens: int = 1800, temperature: float = 0.0) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.ledger.reserve_call(is_retry=attempt > 0)
            started = time.perf_counter()
            try:
                self.ledger.mark_sent()
                body, headers = self._post(prompt, max_output_tokens=max_output_tokens, temperature=temperature)
                latency = time.perf_counter() - started
                response = self._parse_response(body, headers, latency, attempt)
                self.ledger.record_usage(response.input_tokens, response.output_tokens)
                return response
            except LLMRequestError as exc:
                last_error = exc
                if not exc.recoverable or attempt >= self.max_retries:
                    raise
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMRequestError(f"network_error:{type(exc).__name__}", recoverable=True) from exc
            delay = min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.1)
            time.sleep(delay)
        raise LLMRequestError(f"request_failed:{type(last_error).__name__ if last_error else 'unknown'}")

    def _post(self, prompt: str, *, max_output_tokens: int, temperature: float) -> tuple[str, dict[str, str]]:
        if self.config.api_style == "responses":
            endpoint = self.config.base_url + "/responses"
            payload = {
                "model": self.config.model,
                "input": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "text": {"format": {"type": "json_object"}},
            }
        else:
            endpoint = self.config.base_url + "/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "response_format": {"type": "json_object"},
            }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                safe_headers = {key.lower(): value for key, value in response.headers.items() if key.lower() in {"x-request-id", "request-id"}}
                return body, safe_headers
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            recoverable = status in {408, 409, 429} or 500 <= status <= 599
            error_body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMRequestError(f"http_{status}:{error_body}", recoverable=recoverable, status_code=status) from exc

    def _parse_response(self, body: str, headers: dict[str, str], latency: float, retries: int) -> LLMResponse:
        try:
            payload = json.loads(body)
            if self.config.api_style == "responses":
                text = payload.get("output_text")
                if not text:
                    parts = []
                    for output in payload.get("output", []):
                        for content in output.get("content", []):
                            if isinstance(content.get("text"), str):
                                parts.append(content["text"])
                    text = "".join(parts)
            else:
                text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty model text")
            usage = payload.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
            request_id = payload.get("id") or headers.get("x-request-id") or headers.get("request-id")
            cost = usage.get("cost")
            return LLMResponse(
                text=text,
                request_id=str(request_id) if request_id else None,
                input_tokens=int(input_tokens) if input_tokens is not None else None,
                output_tokens=int(output_tokens) if output_tokens is not None else None,
                latency_seconds=latency,
                raw_body=body,
                retries=retries,
                actual_cost=float(cost) if cost is not None else None,
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMRequestError(f"invalid_api_response:{type(exc).__name__}", recoverable=False) from exc
