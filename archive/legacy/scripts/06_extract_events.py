from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.text.event_extractor import extract_events_for_domain
from cents.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    domains = rank_domains(cfg["processed_dir"], cfg.get("max_domains", 3)) if cfg.get("domains") == "auto" else cfg["domains"]
    for domain in domains:
        df = load_processed_domain(cfg["processed_dir"], domain, cfg.get("max_rows"))
        path = Path("data/cache/llm_events") / f"{domain}_events.jsonl"
        events = extract_events_for_domain(df, domain, cfg.get("target_variable", "OT"), path, cfg.get("max_rows"))
        print(f"{domain}: {len(events)} events -> {path}")


if __name__ == "__main__":
    main()

