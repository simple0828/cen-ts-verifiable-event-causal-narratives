from __future__ import annotations


def llm_forecast_placeholder(narrative: str) -> dict:
    return {
        "available": False,
        "note": "Stage-1 prototype generates narratives but does not call an LLM forecaster by default.",
        "narrative_preview": narrative[:500],
    }

