from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cen_tats.runtime.p1_official import diagnose_inputs


if __name__ == "__main__":
    diagnose_inputs()
