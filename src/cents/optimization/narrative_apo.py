from __future__ import annotations

from cents.optimization.prompt_program import DEFAULT_PROMPT_PROGRAM


def run_lightweight_apo(failures: list[dict] | None = None) -> dict:
    failures = failures or []
    candidates = [DEFAULT_PROMPT_PROGRAM]
    if failures:
        candidates.append({
            **DEFAULT_PROMPT_PROGRAM,
            "event_selection_rule": "Require verifier score >= 0.55 and drop generic trend-only statements.",
        })
    return {
        "status": "interface_ready",
        "llm_calls": 0,
        "selected_program": candidates[-1],
        "candidates": candidates,
        "failure_count": len(failures),
    }

