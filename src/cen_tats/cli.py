from __future__ import annotations

import argparse

from cen_tats import pipeline


def main(stage: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v5/cen_tats_v5.yaml")
    parser.add_argument("--include-ablations", action="store_true", default=False)
    args = parser.parse_args()
    stage = stage or "run_all"
    mapping = {
        "audit_repo": pipeline.audit_repo,
        "prepare_data": pipeline.prepare_data,
        "extract_events": pipeline.extract_events,
        "build_causal_graph": pipeline.build_causal_graphs,
        "infer_events_from_series": pipeline.infer_inverse_events,
        "run_verifier": pipeline.run_verifier,
        "render_narratives": pipeline.render_narratives_and_optimize,
        "optimize_prompts": pipeline.render_narratives_and_optimize,
        "cache_tats_embeddings": lambda config: pipeline.run_forecasting(config, include_ablations=False),
        "run_forecasting": lambda config: pipeline.run_forecasting(config, include_ablations=False),
        "run_ablations": lambda config: pipeline.run_forecasting(config, include_ablations=True),
        "aggregate_results": pipeline.aggregate_results,
        "generate_report_assets": pipeline.generate_report,
        "run_all": lambda config: pipeline.run_all(config, include_ablations=args.include_ablations),
    }
    if stage not in mapping:
        raise ValueError(f"Unknown v5 stage: {stage}")
    mapping[stage](args.config)
