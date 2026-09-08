from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from cents.data.gametime_loader import prepare_gametime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/GAMETime")
    parser.add_argument("--out_dir", default="data/processed/GAMETime")
    args = parser.parse_args()
    print(json.dumps(prepare_gametime(args.raw_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()

