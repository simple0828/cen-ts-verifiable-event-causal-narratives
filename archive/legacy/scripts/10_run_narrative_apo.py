from __future__ import annotations

import json

from _bootstrap import bootstrap

bootstrap()

from cents.optimization.narrative_apo import run_lightweight_apo


if __name__ == "__main__":
    print(json.dumps(run_lightweight_apo([]), indent=2))

