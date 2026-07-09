from __future__ import annotations

import numpy as np

from cents.evaluation.forecasting_metrics import forecasting_metrics


def test_forecasting_metrics_zero_error() -> None:
    y = np.array([[1.0, 2.0, 3.0]])
    metrics = forecasting_metrics(y, y)
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0

