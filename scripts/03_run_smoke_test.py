from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cents.experiment_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/smoke_test.yaml")
    args = parser.parse_args()
    run_dir = run_experiment(args.config, mode="smoke")
    print(f"smoke run complete: {run_dir}")


if __name__ == "__main__":
    main()

