from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cen_tats.runtime.p1_official import refresh_status_after_tests


if __name__ == "__main__":
    refresh_status_after_tests()
