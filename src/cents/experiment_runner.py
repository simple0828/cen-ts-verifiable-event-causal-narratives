from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.evaluation.explanation_metrics import explanation_summary
from cents.evaluation.forecasting_metrics import forecasting_metrics
from cents.evaluation.robustness_metrics import performance_drop
from cents.graph.graph_io import save_graph
from cents.graph.lagged_causal_graph import build_event_variable_graph
from cents.models.event_feature_models import event_feature_forecast
from cents.models.fusion_baselines import text_fusion_forecast
from cents.models.numeric_baselines import numeric_columns, persistence_forecast, ridge_window_forecast
from cents.optimization.narrative_apo import run_lightweight_apo
from cents.text.event_extractor import extract_events_for_domain
from cents.text.narrative_builder import build_causal_narrative
from cents.utils.io import ensure_dir, read_yaml, write_json, write_jsonl
from cents.utils.seed import set_seed
from cents.verifier.event_consistency import verify_events


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _select_domains(config: dict) -> list[str]:
    if config.get("domains") == "auto":
        return rank_domains(config["processed_dir"], int(config.get("max_domains", 3)))
    return list(config.get("domains", []))


def _record_predictions(domain: str, method: str, horizon: int, y_true: np.ndarray, y_pred: np.ndarray, anchors: list[int]) -> pd.DataFrame:
    rows = []
    for i, anchor in enumerate(anchors):
        for step in range(y_true.shape[1]):
            rows.append({
                "domain": domain,
                "method": method,
                "horizon": horizon,
                "anchor": anchor,
                "step": step + 1,
                "y_true": float(y_true[i, step]),
                "y_pred": float(y_pred[i, step]),
            })
    return pd.DataFrame(rows)


def run_experiment(config_path: str | Path, mode: str = "smoke") -> Path:
    config = read_yaml(config_path)
    exp_name = config.get("exp_name", mode)
    run_dir = ensure_dir(Path(config.get("run_dir", "experiments/runs")) / f"{_timestamp()}_{exp_name}")
    write_json(run_dir / "config_snapshot.json", config)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    target = config.get("target_variable", "OT")
    history = int(config.get("history_length", 24))
    horizons = list(config.get("horizons", [6]))
    max_rows = config.get("max_rows")
    verifier_threshold = float(config.get("verifier_threshold", 0.5))
    seeds = list(config.get("seeds", [2026]))
    domains = _select_domains(config)
    all_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    selected_events: list[dict] = []
    case_studies: list[dict] = []
    robustness_rows: list[dict] = []

    for seed in seeds:
        set_seed(int(seed))
        for domain in domains:
            df = load_processed_domain(config["processed_dir"], domain, max_rows=max_rows)
            if target not in df:
                target = "OT" if "OT" in df else numeric_columns(df)[0]
            base_cols = numeric_columns(df)
            events_cache = Path("data/cache/llm_events") / f"{domain}_events.jsonl"
            events = extract_events_for_domain(df, domain, target, events_cache, max_rows=max_rows)
            verified_events = verify_events(events, df, target, seed=int(seed))
            graph = build_event_variable_graph(df, verified_events, target)
            save_graph(graph, run_dir / f"{domain}_causal_graph.json")
            narrative = build_causal_narrative(verified_events, graph, top_k=int(config.get("top_k_events", 5)))
            selected_events.extend(verified_events)
            if len(case_studies) < 2:
                top = sorted(verified_events, key=lambda e: e.get("final_consistency_score", 0), reverse=True)[:2]
                for event in top:
                    case_studies.append({
                        "domain": domain,
                        "raw_text": event.get("event_phrase", ""),
                        "event": event,
                        "graph_edges": [edge for edge in graph["edges"] if edge.get("source") == event.get("event_id")][:3],
                        "narrative": narrative,
                    })

            for horizon in horizons:
                runs = []
                y_true, y_pred, anchors = persistence_forecast(df, target, history, int(horizon))
                runs.append(("persistence", y_true, y_pred, anchors, []))
                y_true, y_pred, anchors = ridge_window_forecast(df, base_cols, target, history, int(horizon))
                runs.append(("numerical_only", y_true, y_pred, anchors, []))
                y_true, y_pred, anchors = text_fusion_forecast(
                    df, base_cols, target, history, int(horizon),
                    str(Path("data/cache/text_embeddings") / f"{domain}_{len(df)}.npy"),
                )
                runs.append(("raw_text_embedding", y_true, y_pred, anchors, []))
                y_true, y_pred, anchors = event_feature_forecast(df, base_cols, target, verified_events, history, int(horizon), None)
                runs.append(("event_no_verifier", y_true, y_pred, anchors, verified_events))
                y_true, y_pred, anchors = event_feature_forecast(df, base_cols, target, verified_events, history, int(horizon), verifier_threshold)
                runs.append(("event_with_verifier", y_true, y_pred, anchors, verified_events))
                y_true, y_pred, anchors = event_feature_forecast(df, base_cols, target, verified_events, history, int(horizon), verifier_threshold)
                runs.append(("full_cents_feature", y_true, y_pred, anchors, verified_events))

                original_mse = {}
                for method, truth, pred, anchors, method_events in runs:
                    metrics = forecasting_metrics(truth, pred)
                    expl = explanation_summary(method_events, graph)
                    row = {
                        "seed": seed,
                        "domain": domain,
                        "method": method,
                        "horizon": int(horizon),
                        **metrics,
                        **expl,
                        "train_time_sec": 0.0,
                        "inference_time_sec": 0.0,
                        "llm_calls": 0,
                        "token_cost_estimate": 0.0,
                    }
                    all_metrics.append(row)
                    all_predictions.append(_record_predictions(domain, method, int(horizon), truth, pred, anchors))
                    original_mse[method] = metrics["mse"]

                if mode in {"robustness", "smoke", "main"}:
                    shuffled = df.copy()
                    shuffled["text"] = shuffled["text"].sample(frac=1.0, random_state=int(seed)).to_numpy()
                    sh_events = extract_events_for_domain(shuffled, domain, target, cache_path=None, max_rows=max_rows)
                    sh_events = verify_events(sh_events, shuffled, target, seed=int(seed))
                    truth, pred, _ = event_feature_forecast(shuffled, base_cols, target, sh_events, history, int(horizon), verifier_threshold)
                    sh_mse = forecasting_metrics(truth, pred)["mse"]
                    noisy = df.copy()
                    noisy["text"] = noisy["text"].fillna("") + " unrelated random background sentence"
                    no_events = verify_events(extract_events_for_domain(noisy, domain, target, None, max_rows=max_rows), noisy, target, seed=int(seed))
                    truth, pred, _ = event_feature_forecast(noisy, base_cols, target, no_events, history, int(horizon), verifier_threshold)
                    no_mse = forecasting_metrics(truth, pred)["mse"]
                    robustness_rows.append({
                        "seed": seed,
                        "domain": domain,
                        "method": "event_with_verifier",
                        "horizon": int(horizon),
                        "original_mse": original_mse.get("event_with_verifier", float("nan")),
                        "shuffled_text_mse": sh_mse,
                        "random_text_mse": no_mse,
                        "drop": performance_drop(original_mse.get("event_with_verifier", 0.0), sh_mse),
                    })

    metrics_df = pd.DataFrame(all_metrics)
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "metrics.json", all_metrics)
    pred_df.to_csv(run_dir / "per_window_predictions.csv", index=False)
    write_jsonl(run_dir / "selected_events.jsonl", selected_events)
    write_jsonl(run_dir / "narratives.jsonl", [{"text": build_causal_narrative(selected_events[:10])}])
    pd.DataFrame(robustness_rows).to_csv(run_dir / "robustness.csv", index=False)
    write_json(run_dir / "case_studies.json", case_studies)
    write_json(run_dir / "narrative_apo.json", run_lightweight_apo([]))
    (run_dir / "log.txt").write_text("Experiment completed with stage-1 lightweight baselines.\n", encoding="utf-8")
    return run_dir


def collect_results(runs_dir: str | Path = "experiments/runs", tables_dir: str | Path = "experiments/tables") -> dict:
    runs_dir = Path(runs_dir)
    tables_dir = ensure_dir(tables_dir)
    metric_files = list(runs_dir.glob("*/metrics.csv"))
    robust_files = list(runs_dir.glob("*/robustness.csv"))
    metrics = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    robustness = pd.concat([pd.read_csv(f) for f in robust_files], ignore_index=True) if robust_files else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(tables_dir / "all_metrics.csv", index=False)
        summary = metrics.groupby(["method", "domain", "horizon"], as_index=False).agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            trend_f1=("trend_f1", "mean"),
            event_consistency=("event_consistency", "mean"),
            avg_tokens=("avg_tokens", "mean"),
        )
        summary.to_csv(tables_dir / "main_results.csv", index=False)
        ablation = metrics.groupby("method", as_index=False).agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            event_consistency=("event_consistency", "mean"),
        )
        ablation["observation"] = ablation["method"].map(lambda m: "Stage-1 lightweight closed-loop result")
        ablation.to_csv(tables_dir / "ablation.csv", index=False)
    if not robustness.empty:
        robustness.to_csv(tables_dir / "robustness.csv", index=False)
    return {"metric_files": len(metric_files), "robustness_files": len(robust_files)}

