from __future__ import annotations

import json

from _bootstrap import bootstrap

bootstrap()

from cents.experiment_runner import collect_results


if __name__ == "__main__":
    print(json.dumps(collect_results(), indent=2))

