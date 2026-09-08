from __future__ import annotations

from dataclasses import dataclass, asdict

from cen_tats.io_utils import stable_hash


@dataclass(frozen=True)
class PromptProgram:
    P_extract: str
    P_inverse: str
    P_verify: str
    P_narrative: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["prompt_hash"] = stable_hash(payload)
        return payload


MANUAL_PROMPT = PromptProgram(
    P_extract="Extract only grounded, time-stamped factual events with evidence spans copied from input text.",
    P_inverse="Infer only abstract historical series events from the past numeric window; never use future values.",
    P_verify="Score evidence, causal lag relevance, cycle consistency, and lag observability separately.",
    P_narrative="Render only fields present in the event and verifier JSON using the fixed CEN-TaTS narrative template.",
)
