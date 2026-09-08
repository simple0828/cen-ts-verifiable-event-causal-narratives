from __future__ import annotations

import sys
from pathlib import Path


def run(stage: str) -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from _bootstrap import bootstrap

    bootstrap()
    from cen_tats.cli import main

    main(stage)
