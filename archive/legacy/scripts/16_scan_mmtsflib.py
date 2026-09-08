from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from cents.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="external/MM-TSFlib")
    parser.add_argument("--out", default="experiments/tables/mmtsflib_status.csv")
    args = parser.parse_args()
    root = Path(args.root)
    models = sorted(p.stem for p in (root / "models").glob("*.py") if p.stem != "__init__")
    torch_available = importlib.util.find_spec("torch") is not None
    rows = []
    for model in models:
        rows.append({
            "model": model,
            "available_in_repo": True,
            "torch_available": torch_available,
            "stage1_status": "not_run_missing_torch" if not torch_available else "ready_for_wrapper",
            "note": "MM-TSFlib cloned and model file exists; full training requires torch/GPU environment." if not torch_available else "Torch is available; wrapper can run in next stage.",
        })
    out = Path(args.out)
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

