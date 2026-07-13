import json
from pathlib import Path


def test_p1_required_prediction_hashes_differ() -> None:
    status = json.loads(Path("results/v6/p1/p1_status.json").read_text(encoding="utf-8"))
    assert status["m0_m1_predictions_identical"] is False
    assert status["m1_z0_predictions_identical"] is False
    assert status["m1_s0_predictions_identical"] is False
    assert status["counterfactual_real_zero_max_diff"] > 1e-6
    assert status["counterfactual_real_shuffle_max_diff"] > 1e-6
