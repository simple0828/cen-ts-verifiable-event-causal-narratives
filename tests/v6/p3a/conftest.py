from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tats_cen"))

from cen_ts.api_config import APIConfig
from cen_ts.event_extractor import EventExtractor
from cen_ts.llm_client import LLMRequestError, LLMResponse


def valid_model_payload(evidence: str = "The agency announced a new rule.") -> dict:
    return {
        "events": [{
            "event_time_start": None, "event_time_end": None, "actor": "The agency", "action": "announced",
            "object": "a new rule", "location": None, "event_type_raw": "rule announcement",
            "event_type_canonical": None, "affected_target": None, "expected_direction": "unknown",
            "intensity": "unknown", "candidate_lag_min": None, "candidate_lag_max": None,
            "expected_duration": None, "factuality": "announced", "evidence_span": evidence,
            "source_name": None, "extraction_confidence": 0.9,
        }],
        "no_event": False,
        "no_event_reason": None,
    }


class FakeClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or valid_model_payload()
        self.error = error
        self.calls = 0

    def complete(self, prompt: str, *, max_output_tokens: int, temperature: float) -> LLMResponse:
        self.calls += 1
        if self.error:
            raise self.error
        text = json.dumps(self.payload)
        return LLMResponse(text=text, request_id=f"fake-{self.calls}", input_tokens=10, output_tokens=20, latency_seconds=0.01, raw_body=json.dumps({"id": f"fake-{self.calls}"}), retries=0)


@pytest.fixture
def extractor_factory(tmp_path):
    def factory(payload: dict | None = None, error: Exception | None = None):
        config = APIConfig(api_key="sk-super-secret-value", base_url="https://example.invalid/v1", model="fake-model", max_calls=20)
        extractor = EventExtractor(
            config,
            prompt_path=ROOT / "prompts/v6/event_extraction/p_extract_v1.txt",
            schema_path=ROOT / "prompts/v6/event_extraction/schema_v1.json",
            cache_directory=tmp_path / "cache",
            ledger_path=tmp_path / "ledger.json",
        )
        fake = FakeClient(payload, error)
        extractor.client = fake
        return extractor, fake
    return factory
