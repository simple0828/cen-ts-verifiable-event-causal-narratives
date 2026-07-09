from __future__ import annotations


DEFAULT_PROMPT_PROGRAM = {
    "event_extraction_instruction": "Extract conservative structured events only when text suggests a plausible temporal mechanism.",
    "event_selection_rule": "Prefer high confidence and verifier-consistent events.",
    "lag_rule": "Use 1-3 dataset steps unless domain evidence indicates otherwise.",
    "narrative_template": "Verified causal event narrative with time, target, effect, lag, score, and evidence.",
}

