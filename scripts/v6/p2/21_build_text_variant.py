from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TATS_CEN = ROOT / "tats_cen"
if str(TATS_CEN) not in sys.path:
    sys.path.insert(0, str(TATS_CEN))

from cen_ts.variant_builder import build_text_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P2 CEN-TaTS text variants.")
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--text_column", default="fact")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--manifest_path", required=True)
    args = parser.parse_args()
    manifest = build_text_variant(
        source_csv=Path(args.source_csv),
        output_csv=Path(args.output_csv),
        mode=args.mode,
        text_column=args.text_column,
        seed=args.seed,
        manifest_path=Path(args.manifest_path),
        input_jsonl=Path(args.input_jsonl) if args.input_jsonl else None,
    )
    print(json.dumps({k: manifest[k] for k in ["mode", "output_csv", "output_sha256", "non_text_columns_identical"]}, indent=2))


if __name__ == "__main__":
    main()
