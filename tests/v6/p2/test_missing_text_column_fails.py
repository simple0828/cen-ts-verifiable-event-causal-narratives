import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.text_modes import require_column


def test_missing_text_column_fails():
    with pytest.raises(ValueError, match="No fallback to fact"):
        require_column(["date", "fact", "OT"], "event_fact")
