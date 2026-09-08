"""Restore missing pristine TaTS models from a fixed local upstream commit.

Existing files, minimal adapters, business code, and experiment artifacts are
never overwritten. A normal checkout already contains all required models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cen_ts.paths import TATS_ROOT, UPSTREAM_ROOT, UPSTREAM_COMMIT


def prepare_vendor(destination: Path = TATS_ROOT, upstream: Path = UPSTREAM_ROOT) -> dict:
    manifest = json.loads((TATS_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    restored, preserved = [], []
    for relative, expected in manifest["model_blob_sha256"].items():
        path = destination / relative
        if path.exists():
            preserved.append(relative)
            continue
        data = subprocess.check_output([
            "git", "-c", f"safe.directory={upstream.resolve().as_posix()}",
            "-C", str(upstream), "show", f"{UPSTREAM_COMMIT}:{relative}",
        ])
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Pinned upstream content mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        restored.append(relative)
    return {"upstream_commit": UPSTREAM_COMMIT, "restored": restored, "preserved": preserved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare_vendor(upstream=args.upstream), indent=2))


if __name__ == "__main__":
    main()
