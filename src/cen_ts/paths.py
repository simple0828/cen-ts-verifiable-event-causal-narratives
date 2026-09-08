"""Repository paths for editable installs and stage scripts."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TATS_ROOT = ROOT / "vendor" / "tats"
UPSTREAM_ROOT = ROOT / "third_party" / "TaTS"
UPSTREAM_COMMIT = "a053503674c61c54d101d01d47c9d680288a7c9a"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def gpt2_path(value: str | Path | None = None) -> Path:
    return project_path(value or os.environ.get("CEN_TS_GPT2_PATH") or "models/gpt2")


DEFAULT_GPT2_PATH = str(gpt2_path())
