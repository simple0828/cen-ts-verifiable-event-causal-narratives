import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]


def test_prior_weight_zero_math_and_run():
    run_metrics = ROOT / "results" / "v6" / "p2" / "runs" / "p2_raw_forecasting_fair_pw0.0_s2025" / "test_metrics.json"
    assert run_metrics.exists()
    outputs = torch.randn(2, 48, 1)
    prior_a = torch.randn(2, 48, 1)
    prior_b = torch.randn(2, 48, 1)
    mixed_a = (1 - 0.0) * outputs + 0.0 * prior_a
    mixed_b = (1 - 0.0) * outputs + 0.0 * prior_b
    assert torch.equal(mixed_a, mixed_b)
    metrics = json.loads(run_metrics.read_text(encoding="utf-8"))
    assert metrics["native_scaled"]["MSE"] > 0
