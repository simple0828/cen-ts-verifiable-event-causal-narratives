from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cen_ts.utils.paths import project_path


def evaluate_predictions(path: Path) -> dict[str, float]:
    frame = pd.read_csv(project_path(path))
    true = frame["target_normalized"].to_numpy(dtype=np.float64)
    pred = frame["prediction_normalized"].to_numpy(dtype=np.float64)
    error = pred - true
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error / true))),
        "MSE": float(np.mean(error ** 2)),
        "MSPE": float(np.mean((error / true) ** 2)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved normalized forecasts with the published metric definitions.")
    parser.add_argument("--predictions", type=Path, default=Path("results/latest/predictions.csv"))
    parser.add_argument("--reference", type=Path, default=Path("results/latest/metrics.json"))
    args = parser.parse_args()
    metrics = evaluate_predictions(args.predictions)
    reference = json.loads(project_path(args.reference).read_text(encoding="utf-8"))["normalized"]
    for name, value in metrics.items():
        if not np.isclose(value, reference[name], rtol=1e-6, atol=1e-7):
            raise AssertionError(f"{name} differs from published result: {value} != {reference[name]}")
    print(json.dumps(metrics, indent=2))
    return 0
