from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.graph.granger_lite import granger_lite_edges
from cents.utils.io import ensure_dir, read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    parser.add_argument("--out", default="experiments/tables/granger_edges.csv")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    domains = rank_domains(cfg["processed_dir"], cfg.get("max_domains", 3)) if cfg.get("domains") == "auto" else cfg["domains"]
    rows = []
    for domain in domains:
        df = load_processed_domain(cfg["processed_dir"], domain, cfg.get("max_rows"))
        edges = granger_lite_edges(df, cfg.get("target_variable", "OT"), max_lag=3)
        edges["domain"] = domain
        rows.append(edges.head(50))
        print(f"{domain}: {int(edges['significant'].sum())}/{len(edges)} significant Granger-lite edges")
    out = Path(args.out)
    ensure_dir(out.parent)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

