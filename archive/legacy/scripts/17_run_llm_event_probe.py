from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.text.llm_event_extractor import run_llm_event_probe
from cents.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/smoke_test.yaml")
    parser.add_argument("--max_rows", type=int, default=3)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    domain = rank_domains(cfg["processed_dir"], 1)[0]
    df = load_processed_domain(cfg["processed_dir"], domain, cfg.get("max_rows"))
    status = run_llm_event_probe(df, domain, cfg.get("target_variable", "OT"), max_rows=args.max_rows)
    print(status)


if __name__ == "__main__":
    main()

