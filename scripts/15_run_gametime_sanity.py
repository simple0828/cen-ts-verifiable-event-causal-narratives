from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from cents.utils.io import ensure_dir
from cents.verifier.event_consistency import score_event_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/GAMETime")
    parser.add_argument("--out", default="experiments/tables/gametime_sanity.csv")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    csv_like = list(raw_dir.rglob("*.csv")) + list(raw_dir.rglob("*.json")) + list(raw_dir.rglob("*.jsonl"))
    synthetic = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=16, freq="D"),
        "OT": [0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.8, 0.6, 0.4, 0.2, 0.2, 0.2, 0.2],
    })
    pos = score_event_consistency({"time": "2024-01-04", "polarity": "positive", "expected_lag_min": 1, "expected_lag_max": 3}, synthetic, "OT")
    neg = score_event_consistency({"time": "2024-01-09", "polarity": "negative", "expected_lag_min": 1, "expected_lag_max": 4}, synthetic, "OT")
    rows = [{
        "check": "repository_data_availability",
        "passed": bool(csv_like),
        "score": float(len(csv_like)),
        "note": "GAMETime repo clone does not include full benchmark data; README says to request/download data separately." if not csv_like else "Found candidate data files.",
    }, {
        "check": "synthetic_positive_event_verifier",
        "passed": pos["final_consistency_score"] > 0.6,
        "score": pos["final_consistency_score"],
        "note": "Positive event followed by target increase.",
    }, {
        "check": "synthetic_negative_event_verifier",
        "passed": neg["final_consistency_score"] > 0.6,
        "score": neg["final_consistency_score"],
        "note": "Negative event followed by target decrease.",
    }]
    out = Path(args.out)
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

