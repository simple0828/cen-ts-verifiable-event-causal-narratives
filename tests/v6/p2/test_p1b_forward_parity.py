import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_p1b_forward_parity():
    parity = json.loads((ROOT / "results" / "v6" / "p2" / "p1b_parity.json").read_text(encoding="utf-8"))
    assert parity["p2_parity"] == "PASS"
    assert parity["forward_max_abs_diff"] < 1e-6
    assert parity["loss_abs_diff"] < 1e-6
    assert parity["gradient_max_abs_diff"] < 1e-6
    assert parity["metric_relative_difference"] < 0.01
    assert parity["prediction_correlation"] > 0.99
