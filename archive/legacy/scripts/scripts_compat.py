from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cents.experiment_runner import run_experiment


def run_main(default_config: str, mode: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args()
    print(run_experiment(args.config, mode=mode))

