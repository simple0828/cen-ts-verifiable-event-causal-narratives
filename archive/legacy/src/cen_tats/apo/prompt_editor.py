from __future__ import annotations

from cen_tats.apo.prompt_program import PromptProgram


def edit_prompt(program: PromptProgram, gradient: str, n: int = 3) -> list[PromptProgram]:
    suffixes = [
        " Reject forecasts and opinions unless factuality is explicitly marked.",
        " Preserve lag-aware observability: do not penalize cycle mismatch before lag_min is reached.",
        " Prefer target-relevant events with accepted graph edges and grounded evidence.",
    ]
    candidates = []
    for suffix in suffixes[:n]:
        candidates.append(
            PromptProgram(
                P_extract=program.P_extract + suffix,
                P_inverse=program.P_inverse + " Report numeric evidence as slope/change/change_point only.",
                P_verify=program.P_verify + " " + gradient.replace("\n", " "),
                P_narrative=program.P_narrative + " Do not add causes, entities, or numbers not in verifier JSON.",
            )
        )
    return candidates
