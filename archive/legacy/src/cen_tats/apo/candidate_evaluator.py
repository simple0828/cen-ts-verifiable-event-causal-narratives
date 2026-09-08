from __future__ import annotations

from cen_tats.apo.prompt_program import PromptProgram


def evaluate_candidate(program: PromptProgram, proxy_metrics: dict, weights: dict) -> dict:
    token_cost = sum(len(v.split()) for v in program.to_dict().values() if isinstance(v, str))
    schema_loss = 1.0 - float(proxy_metrics.get("schema_validity", 1.0))
    grounding_loss = 1.0 - float(proxy_metrics.get("grounding", 1.0))
    cycle_loss = 1.0 - float(proxy_metrics.get("cycle", 0.5))
    forecast_loss = float(proxy_metrics.get("forecast_mse", 1.0))
    score = (
        weights.get("forecast", 1.0) * forecast_loss
        + weights.get("schema", 0.2) * schema_loss
        + weights.get("grounding", 0.2) * grounding_loss
        + weights.get("cycle", 0.2) * cycle_loss
        + weights.get("token", 0.01) * token_cost
    )
    return {"score": float(score), "token_cost": int(token_cost), "prompt_hash": program.to_dict()["prompt_hash"]}
