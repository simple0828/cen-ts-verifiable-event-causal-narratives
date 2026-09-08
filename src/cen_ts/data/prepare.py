from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from cen_ts.utils.paths import ROOT, project_path

REQUIRED_COLUMNS = {"date", "OT", "fact"}


def prepare_environment_data(source: Path, output: Path | None = None) -> dict:
    source = project_path(source)
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) != 15248:
        raise ValueError(f"Expected 15248 Environment rows, found {len(frame)}")
    if output is not None:
        output = project_path(output)
        if output != source:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"path": str(source.relative_to(ROOT)), "rows": len(frame), "columns": list(frame.columns), "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or install the Environment experiment dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/Environment.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_environment_data(args.input, args.output), indent=2))
    return 0
