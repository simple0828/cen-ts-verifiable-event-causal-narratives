from __future__ import annotations

from cen_tats.apo.beam_search import beam_search
from cen_tats.apo.prompt_program import MANUAL_PROMPT


def optimize_prompts(failures: list[dict], cfg: dict) -> dict:
    return beam_search(MANUAL_PROMPT, failures, cfg)
