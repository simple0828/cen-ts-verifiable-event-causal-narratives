from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from _bootstrap import bootstrap

bootstrap()

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cents.data.splits import temporal_split_indices
from cents.data.timemmd_loader import load_processed_domain, rank_domains
from cents.evaluation.forecasting_metrics import forecasting_metrics
from cents.models.event_feature_models import EVENT_FEATURE_COLS, add_event_features
from cents.models.fusion_baselines import add_text_embeddings
from cents.models.numeric_baselines import numeric_columns, persistence_forecast
from cents.text.event_extractor import extract_events_for_domain
from cents.utils.io import ensure_dir, read_yaml
from cents.utils.seed import set_seed
from cents.verifier.event_consistency import verify_events


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    anchors: list[int]
    feature_cols: list[str]
    target_index: int


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _select_domains(config: dict) -> list[str]:
    if config.get("domains") == "auto":
        return rank_domains(config["processed_dir"], int(config.get("max_domains", 3)))
    return list(config.get("domains", []))


def _make_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str, history: int, horizon: int) -> WindowData:
    values = df[feature_cols].astype(float).to_numpy()
    target = df[target_col].astype(float).to_numpy()
    xs, ys, anchors = [], [], []
    for end in range(history, len(df) - horizon + 1):
        xs.append(values[end - history:end])
        ys.append(target[end:end + horizon])
        anchors.append(end)
    return WindowData(np.asarray(xs), np.asarray(ys), anchors, feature_cols, feature_cols.index(target_col))


def _load_mmts_model(model_name: str):
    root = Path("external/MM-TSFlib").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"models.{model_name}").Model


def _model_config(model_name: str, seq_len: int, pred_len: int, n_features: int) -> SimpleNamespace:
    down_sampling_layers = 1 if model_name == "TimeMixer" else 0
    down_sampling_window = 2 if model_name == "TimeMixer" else 1
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=max(1, seq_len // 2),
        pred_len=pred_len,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        moving_avg=3,
        factor=1,
        dropout=0.05,
        embed="fixed",
        freq="d",
        activation="gelu",
        output_attention=False,
        top_k=2,
        num_kernels=3,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=down_sampling_layers,
        down_sampling_window=down_sampling_window,
        down_sampling_method="avg",
        model=model_name,
    )


def _predict(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device), None, None, None)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _fit_one(
    model_name: str,
    windows: WindowData,
    split,
    history: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, list[int]]:
    set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = windows.x.astype(np.float32)
    y = windows.y.astype(np.float32)
    train_x = x[split.train]
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean) / std
    y_scaled = (y - mean[windows.target_index]) / std[windows.target_index]

    x_train = torch.tensor(x_scaled[split.train], dtype=torch.float32)
    y_train = torch.tensor(y_scaled[split.train], dtype=torch.float32)
    x_val = torch.tensor(x_scaled[split.val], dtype=torch.float32)
    y_val = torch.tensor(y_scaled[split.val], dtype=torch.float32)
    x_test = torch.tensor(x_scaled[split.test], dtype=torch.float32)

    model_cls = _load_mmts_model(model_name)
    model = model_cls(_model_config(model_name, history, horizon, x.shape[-1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    start = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb, None, None, None)[:, :, windows.target_index]
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        if len(split.val):
            with torch.no_grad():
                val_out = model(x_val.to(device), None, None, None)[:, :, windows.target_index]
                val_loss = float(criterion(val_out, y_val.to(device)).detach().cpu())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    scaled_pred = _predict(model, x_test, batch_size, device)[:, :, windows.target_index]
    pred = scaled_pred * std[windows.target_index] + mean[windows.target_index]
    truth = y[split.test]
    metrics = forecasting_metrics(truth, pred)
    metrics["train_time_sec"] = float(time.time() - start)
    metrics["torch_device"] = str(device)
    metrics["best_val_scaled_mse"] = float(best_val)
    return metrics, truth, pred, [windows.anchors[i] for i in split.test]


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 1e-8
    if not np.any(valid):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]))


def _normalized_metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence_mse: float) -> dict:
    base = forecasting_metrics(y_true, y_pred)
    variance = float(np.var(y_true))
    nmse = float(base["mse"] / variance) if variance > 1e-12 else float("nan")
    rel = float((persistence_mse - base["mse"]) / persistence_mse) if persistence_mse > 1e-12 else float("nan")
    return {**base, "nmse": nmse, "smape": _smape(y_true, y_pred), "relative_improvement_over_persistence": rel}


def _event_mask(anchors: list[int], event_signal: np.ndarray, history: int, horizon: int) -> np.ndarray:
    mask = []
    for anchor in anchors:
        hist_start = max(0, anchor - history)
        forecast_end = min(len(event_signal), anchor + horizon)
        mask.append(float(np.sum(event_signal[hist_start:forecast_end])) > 0.0)
    return np.asarray(mask, dtype=bool)


def _volatility_mask(df: pd.DataFrame, target_col: str, anchors: list[int], quantile: float = 0.75) -> np.ndarray:
    values = df[target_col].astype(float).to_numpy()
    diff = np.abs(np.diff(values, prepend=values[0]))
    threshold = float(np.nanquantile(diff, quantile))
    return np.asarray([diff[min(anchor, len(diff) - 1)] >= threshold for anchor in anchors], dtype=bool)


def _subset_row(label: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"subset": label, "n_windows": 0}
    metrics = forecasting_metrics(y_true[mask], y_pred[mask])
    return {"subset": label, "n_windows": int(mask.sum()), **metrics}


def _enrich_frame(
    df: pd.DataFrame,
    method: str,
    domain: str,
    events: list[dict],
    target: str,
    verifier_threshold: float,
    text_cache: Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    base_cols = numeric_columns(df)
    if method == "strong_numerical_only":
        return df.copy(), base_cols, np.zeros(len(df), dtype=float)
    if method == "strong_raw_text_fusion":
        fused, text_cols = add_text_embeddings(df, str(text_cache / f"{domain}_{len(df)}.npy"))
        return fused, base_cols + text_cols, np.zeros(len(df), dtype=float)
    if method == "strong_event_no_verifier":
        enriched = add_event_features(df, events, verifier_threshold=None)
        return enriched, base_cols + EVENT_FEATURE_COLS, enriched["event_count"].to_numpy(dtype=float)
    if method == "strong_event_with_verifier":
        enriched = add_event_features(df, events, verifier_threshold=verifier_threshold)
        return enriched, base_cols + EVENT_FEATURE_COLS, enriched["event_count"].to_numpy(dtype=float)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    parser.add_argument("--models", nargs="+", default=["TimesNet", "PatchTST"])
    parser.add_argument("--methods", nargs="+", default=[
        "strong_numerical_only",
        "strong_raw_text_fusion",
        "strong_event_no_verifier",
        "strong_event_with_verifier",
    ])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="experiments/runs")
    parser.add_argument("--energy-robustness", action="store_true", default=True)
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_dir = ensure_dir(Path(args.out_dir) / f"{_timestamp()}_torch_event_fusion")
    tables_dir = ensure_dir("experiments/tables")
    text_cache = ensure_dir("data/cache/text_embeddings")
    target = config.get("target_variable", "OT")
    history = int(config.get("history_length", 24))
    horizons = list(config.get("horizons", [3, 6]))
    seeds = list(config.get("seeds", [2026]))
    verifier_threshold = float(config.get("verifier_threshold", 0.5))
    max_rows = config.get("max_rows")
    domains = _select_domains(config)

    rows: list[dict] = []
    subset_rows: list[dict] = []
    robustness_rows: list[dict] = []
    failures: list[dict] = []

    for seed in seeds:
        set_seed(int(seed))
        for domain in domains:
            df = load_processed_domain(config["processed_dir"], domain, max_rows=max_rows)
            target_col = target if target in df else ("OT" if "OT" in df else numeric_columns(df)[0])
            base_events = extract_events_for_domain(df, domain, target_col, cache_path=None, max_rows=max_rows)
            verified_events = verify_events(base_events, df, target_col, seed=int(seed))
            reference_events = add_event_features(df, verified_events, verifier_threshold=verifier_threshold)
            reference_event_signal = reference_events["event_count"].to_numpy(dtype=float)
            for horizon in horizons:
                p_truth, p_pred, p_anchors = persistence_forecast(df, target_col, history, int(horizon))
                split_for_persistence = temporal_split_indices(len(p_truth))
                p_truth = p_truth[split_for_persistence.test]
                p_pred = p_pred[split_for_persistence.test]
                persistence_metrics = forecasting_metrics(p_truth, p_pred)
                persistence_mse = persistence_metrics["mse"]
                rows.append({
                    "seed": int(seed),
                    "domain": domain,
                    "method": "persistence",
                    "source_model": "persistence",
                    "horizon": int(horizon),
                    "history_length": history,
                    "epochs": 0,
                    "batch_size": 0,
                    "learning_rate": 0.0,
                    **_normalized_metrics(p_truth, p_pred, persistence_mse),
                })

                for method in args.methods:
                    try:
                        enriched, feature_cols, event_signal = _enrich_frame(
                            df, method, domain, verified_events, target_col, verifier_threshold, text_cache
                        )
                        windows = _make_windows(enriched, feature_cols, target_col, history, int(horizon))
                        split = temporal_split_indices(len(windows.x))
                        for model_name in args.models:
                            metrics, truth, pred, anchors = _fit_one(
                                model_name=model_name,
                                windows=windows,
                                split=split,
                                history=history,
                                horizon=int(horizon),
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                seed=int(seed),
                            )
                            norm = _normalized_metrics(truth, pred, persistence_mse)
                            rows.append({
                                "seed": int(seed),
                                "domain": domain,
                                "method": method,
                                "source_model": model_name,
                                "horizon": int(horizon),
                                "history_length": history,
                                "epochs": args.epochs,
                                "batch_size": args.batch_size,
                                "learning_rate": args.learning_rate,
                                "n_features": len(feature_cols),
                                **norm,
                                "train_time_sec": metrics["train_time_sec"],
                                "torch_device": metrics["torch_device"],
                                "best_val_scaled_mse": metrics["best_val_scaled_mse"],
                            })
                            event_m = _event_mask(anchors, reference_event_signal, history, int(horizon))
                            vol_m = _volatility_mask(df, target_col, anchors)
                            for subset in [
                                _subset_row("event_window", truth, pred, event_m),
                                _subset_row("high_volatility_window", truth, pred, vol_m),
                            ]:
                                subset_rows.append({
                                    "seed": int(seed),
                                    "domain": domain,
                                    "method": method,
                                    "source_model": model_name,
                                    "horizon": int(horizon),
                                    **subset,
                                })

                            if args.energy_robustness and domain == "Energy" and method == "strong_event_with_verifier":
                                shuffled = df.copy()
                                shuffled["text"] = shuffled["text"].sample(frac=1.0, random_state=int(seed)).to_numpy()
                                sh_events = verify_events(
                                    extract_events_for_domain(shuffled, domain, target_col, None, max_rows=max_rows),
                                    shuffled,
                                    target_col,
                                    seed=int(seed),
                                )
                                sh_enriched, sh_cols, _ = _enrich_frame(
                                    shuffled, method, domain, sh_events, target_col, verifier_threshold, text_cache
                                )
                                sh_windows = _make_windows(sh_enriched, sh_cols, target_col, history, int(horizon))
                                sh_metrics, _, _, _ = _fit_one(
                                    model_name=model_name,
                                    windows=sh_windows,
                                    split=temporal_split_indices(len(sh_windows.x)),
                                    history=history,
                                    horizon=int(horizon),
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    learning_rate=args.learning_rate,
                                    seed=int(seed),
                                )
                                robustness_rows.append({
                                    "seed": int(seed),
                                    "domain": domain,
                                    "method": method,
                                    "source_model": model_name,
                                    "horizon": int(horizon),
                                    "original_mse": norm["mse"],
                                    "shuffled_text_mse": sh_metrics["mse"],
                                    "drop": sh_metrics["mse"] - norm["mse"],
                                })
                    except Exception as exc:
                        failures.append({
                            "seed": int(seed),
                            "domain": domain,
                            "method": method,
                            "horizon": int(horizon),
                            "error": repr(exc),
                        })

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        non_persistence = metrics_df[metrics_df["method"] != "persistence"].copy()
        non_persistence["rank_mse_in_domain_horizon"] = non_persistence.groupby(
            ["domain", "horizon"]
        )["mse"].rank(method="min")
        metrics_df = pd.concat([metrics_df[metrics_df["method"] == "persistence"], non_persistence], ignore_index=True)

    summary = pd.DataFrame()
    if not metrics_df.empty:
        summary = metrics_df.groupby(["method", "source_model"], as_index=False).agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            nmse=("nmse", "mean"),
            smape=("smape", "mean"),
            trend_f1=("trend_f1", "mean"),
            relative_improvement_over_persistence=("relative_improvement_over_persistence", "mean"),
        ).sort_values(["mse", "nmse"])

    metrics_df.to_csv(run_dir / "torch_event_fusion.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(run_dir / "torch_event_window_metrics.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(run_dir / "torch_energy_robustness.csv", index=False)
    pd.DataFrame(failures).to_csv(run_dir / "torch_event_fusion_failures.csv", index=False)
    summary.to_csv(run_dir / "torch_event_fusion_summary.csv", index=False)

    metrics_df.to_csv(tables_dir / "torch_event_fusion.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(tables_dir / "torch_event_window_metrics.csv", index=False)
    pd.DataFrame(robustness_rows).to_csv(tables_dir / "torch_energy_robustness.csv", index=False)
    pd.DataFrame(failures).to_csv(tables_dir / "torch_event_fusion_failures.csv", index=False)
    summary.to_csv(tables_dir / "torch_event_fusion_summary.csv", index=False)

    print(run_dir / "torch_event_fusion.csv")
    print(run_dir / "torch_event_fusion_summary.csv")


if __name__ == "__main__":
    main()
