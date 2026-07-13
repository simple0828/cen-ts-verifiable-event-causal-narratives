from __future__ import annotations

from cen_tats.apo.candidate_evaluator import evaluate_candidate
from cen_tats.apo.prompt_editor import edit_prompt
from cen_tats.apo.prompt_program import PromptProgram
from cen_tats.apo.textual_gradient import textual_gradient


def beam_search(initial: PromptProgram, failures: list[dict], cfg: dict) -> dict:
    beam = [initial]
    trajectory = []
    weights = cfg.get("weights", {})
    for depth in range(int(cfg.get("max_depth", 2))):
        gradient = textual_gradient(failures)
        candidates = []
        for parent in beam:
            candidates.extend(edit_prompt(parent, gradient, int(cfg.get("candidates_per_parent", 3))))
        scored = []
        for cand in candidates:
            proxy = {
                "schema_validity": 0.95,
                "grounding": 0.90 + 0.02 * depth,
                "cycle": 0.65 + 0.03 * depth,
                "forecast_mse": max(0.1, 1.0 - 0.05 * depth),
            }
            scored.append((evaluate_candidate(cand, proxy, weights), cand))
        scored.sort(key=lambda x: x[0]["score"])
        beam = [cand for _, cand in scored[: int(cfg.get("beam_width", 2))]]
        trajectory.append(
            {
                "round": depth,
                "gradient": gradient,
                "candidate_count": len(candidates),
                "best": scored[0][0] if scored else {},
                "beam_hashes": [p.to_dict()["prompt_hash"] for p in beam],
            }
        )
    return {"best_prompt": beam[0].to_dict(), "trajectory": trajectory}
