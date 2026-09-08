import json
from pathlib import Path

import numpy as np

from cen_ts.evaluation.runner import evaluate_predictions


def test_latest_predictions_reproduce_published_metrics():
    root = Path(__file__).resolve().parents[1]
    actual = evaluate_predictions(root / "results/latest/predictions.csv")
    expected = json.loads((root / "results/latest/metrics.json").read_text(encoding="utf-8"))["normalized"]
    for name, value in expected.items():
        assert np.isclose(actual[name], value, rtol=1e-6, atol=1e-7)
