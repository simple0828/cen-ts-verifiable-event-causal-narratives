from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cents.experiment_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    args = parser.parse_args()
    print(run_experiment(args.config, mode="main"))


if __name__ == "__main__":
    main()

