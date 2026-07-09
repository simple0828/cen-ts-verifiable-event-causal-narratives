from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from cents.data.timemmd_loader import prepare_timemmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/Time-MMD")
    parser.add_argument("--out_dir", default="data/processed/TimeMMD")
    parser.add_argument("--target_variable", default="OT")
    args = parser.parse_args()
    summary = prepare_timemmd(args.raw_dir, args.out_dir, args.target_variable)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

