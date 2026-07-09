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
from cents.models.numeric_baselines import numeric_columns
from cents.utils.io import ensure_dir, read_yaml
from cents.utils.seed import set_seed


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
            xb = xb.to(device)
            out = model(xb, None, None, None)
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
) -> dict:
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
    scaled_pred_all = _predict(model, x_test, batch_size, device)[:, :, windows.target_index]
    pred = scaled_pred_all * std[windows.target_index] + mean[windows.target_index]
    truth = y[split.test]
    metrics = forecasting_metrics(truth, pred)
    metrics["train_time_sec"] = float(time.time() - start)
    metrics["torch_device"] = str(device)
    metrics["best_val_scaled_mse"] = float(best_val)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/main_timemmd.yaml")
    parser.add_argument("--models", nargs="+", default=["DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="experiments/runs")
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_dir = ensure_dir(Path(args.out_dir) / f"{_timestamp()}_torch_strong_baselines")
    rows = []
    failures = []
    domains = _select_domains(config)
    target = config.get("target_variable", "OT")
    history = int(config.get("history_length", 24))
    horizons = list(config.get("horizons", [3, 6]))
    seeds = list(config.get("seeds", [2026]))

    for seed in seeds:
        for domain in domains:
            df = load_processed_domain(config["processed_dir"], domain, max_rows=config.get("max_rows"))
            target_col = target if target in df else ("OT" if "OT" in df else numeric_columns(df)[0])
            feature_cols = numeric_columns(df)
            if target_col not in feature_cols:
                feature_cols.append(target_col)
            for horizon in horizons:
                windows = _make_windows(df, feature_cols, target_col, history, int(horizon))
                split = temporal_split_indices(len(windows.x))
                for model_name in args.models:
                    try:
                        metrics = _fit_one(
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
                        rows.append({
                            "seed": int(seed),
                            "domain": domain,
                            "method": f"mmts_{model_name.lower()}",
                            "source_model": model_name,
                            "horizon": int(horizon),
                            "history_length": history,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            **metrics,
                        })
                    except Exception as exc:  # Keep failed strong baselines auditable instead of hiding them.
                        failures.append({
                            "seed": int(seed),
                            "domain": domain,
                            "model": model_name,
                            "horizon": int(horizon),
                            "error": repr(exc),
                        })

    metrics_path = run_dir / "torch_strong_baselines.csv"
    failure_path = run_dir / "torch_strong_baseline_failures.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    tables_dir = ensure_dir("experiments/tables")
    pd.DataFrame(rows).to_csv(tables_dir / "torch_strong_baselines.csv", index=False)
    pd.DataFrame(failures).to_csv(tables_dir / "torch_strong_baseline_failures.csv", index=False)
    print(metrics_path)
    print(failure_path)


if __name__ == "__main__":
    main()
