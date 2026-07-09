from __future__ import annotations

from pathlib import Path

import pandas as pd

from cents.data.timemmd_loader import discover_domains


def test_discover_domains(tmp_path: Path) -> None:
    (tmp_path / "numerical" / "Energy").mkdir(parents=True)
    (tmp_path / "textual" / "Energy").mkdir(parents=True)
    assert discover_domains(tmp_path) == ["Energy"]

