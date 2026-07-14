from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party" / "TaTS"
DEST = ROOT / "tats_cen"
OUT = ROOT / "results" / "v6" / "p2"
SAFE_TATS = "C:/Users/Administrator/Desktop/TS/cen-ts-verifiable-event-causal-narratives/third_party/TaTS"


EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    "checkpoints",
    "results",
    "result",
    "logs",
    "wandb",
    ".cache",
}
EXCLUDED_SUFFIXES = {".pth", ".pt", ".npy", ".npz", ".log"}


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(UPSTREAM), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_allowed(rel: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if rel.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def manifest_entry(root: Path, rel: Path) -> dict[str, Any]:
    path = root / rel
    return {
        "path": rel.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    upstream_commit = run_git(["rev-parse", "HEAD"]).strip()
    status = run_git(["status", "--short"])
    tracked = [Path(line) for line in run_git(["ls-files"]).splitlines() if line.strip()]
    files = [rel for rel in tracked if is_allowed(rel)]

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    upstream_entries = []
    dest_entries = []
    for rel in files:
        src = UPSTREAM / rel
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        upstream_entries.append(manifest_entry(UPSTREAM, rel))
        dest_entries.append(manifest_entry(DEST, rel))

    upstream_manifest = {
        "stage": "P2",
        "source": str(UPSTREAM),
        "upstream_commit": upstream_commit,
        "upstream_status_short": status,
        "tracked_file_count": len(tracked),
        "copied_file_count": len(files),
        "files": upstream_entries,
    }
    dest_manifest = {
        "stage": "P2",
        "destination": str(DEST),
        "upstream_commit": upstream_commit,
        "copied_file_count": len(files),
        "files": dest_entries,
        "content_matches_upstream": upstream_entries == dest_entries,
    }
    write_json(OUT / "upstream_manifest.json", upstream_manifest)
    write_json(OUT / "tats_cen_initial_manifest.json", dest_manifest)
    print(f"copied {len(files)} tracked upstream files from {upstream_commit}")


if __name__ == "__main__":
    main()
