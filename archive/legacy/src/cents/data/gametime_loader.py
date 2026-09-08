from __future__ import annotations

from pathlib import Path

from cents.utils.io import ensure_dir, write_json


def prepare_gametime(raw_dir: str | Path, out_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = ensure_dir(out_dir)
    files = [str(p.relative_to(raw_dir)) for p in raw_dir.rglob("*") if p.is_file()]
    summary = {
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "sample_files": files[:50],
        "note": "GAMETime is registered for verifier sanity checks; stage-1 report records availability.",
    }
    write_json(out_dir / "summary.json", summary)
    return summary

