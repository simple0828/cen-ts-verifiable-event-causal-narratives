from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.graph.graph_io import save_graph
from cents.graph.lagged_causal_graph import build_event_variable_graph
from cents.text.event_extractor import extract_events_for_domain
from cents.utils.io import read_yaml
from cents.verifier.event_consistency import verify_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    domains = rank_domains(cfg["processed_dir"], cfg.get("max_domains", 3)) if cfg.get("domains") == "auto" else cfg["domains"]
    for domain in domains:
        df = load_processed_domain(cfg["processed_dir"], domain, cfg.get("max_rows"))
        events = extract_events_for_domain(df, domain, cfg.get("target_variable", "OT"), Path("data/cache/llm_events") / f"{domain}_events.jsonl", cfg.get("max_rows"))
        events = verify_events(events, df, cfg.get("target_variable", "OT"))
        graph = build_event_variable_graph(df, events, cfg.get("target_variable", "OT"))
        out = Path("data/cache/causal_graphs") / f"{domain}_graph.json"
        save_graph(graph, out)
        print(f"{domain}: {len(graph['edges'])} edges -> {out}")


if __name__ == "__main__":
    main()

