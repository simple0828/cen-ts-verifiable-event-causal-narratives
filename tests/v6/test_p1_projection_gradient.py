import json
from pathlib import Path


def test_p1_m1_projection_gradient_is_nonzero() -> None:
    status = json.loads(Path("results/v6/p1/p1_status.json").read_text(encoding="utf-8"))
    assert status["projection_gradient_norm_max"] > 1e-8
