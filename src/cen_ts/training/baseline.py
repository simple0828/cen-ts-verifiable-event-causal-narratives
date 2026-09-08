from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


from cen_ts.utils.paths import DEFAULT_GPT2_PATH, ROOT, TATS_ROOT
PYTHON = sys.executable
WORK_ROOT = ROOT / ".cache" / "cen_ts" / "training"
PUBLISHED_ROOT = ROOT / "results" / "latest"
TATS_RUN = TATS_ROOT / "run.py"


DATASETS = {
    "raw": ROOT / "data" / "processed" / "Environment.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def parse_epoch_metrics(log_text: str) -> list[dict[str, float]]:
    pattern = re.compile(
        r"Epoch: (?P<epoch>\d+), Steps: (?P<steps>\d+) \| Train Loss: (?P<train>[0-9.]+) Vali Loss: (?P<vali>[0-9.]+) Test Loss \(MSE\): (?P<test>[0-9.]+) Test MAE: (?P<mae>[0-9.]+)"
    )
    rows = []
    for match in pattern.finditer(log_text):
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "steps": int(match.group("steps")),
                "train_loss": float(match.group("train")),
                "vali_loss": float(match.group("vali")),
                "test_mse": float(match.group("test")),
                "test_mae": float(match.group("mae")),
            }
        )
    return rows


def write_epoch_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "steps", "train_loss", "vali_loss", "test_mse", "test_mae"])
        writer.writeheader()
        writer.writerows(rows)


def metrics_unscaled(pred: np.ndarray, true: np.ndarray, csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    train = df.iloc[: int(len(df) * 0.7)]["OT"]
    mean = float(train.mean())
    std = float(train.std(ddof=0))
    pred_u = pred * std + mean
    true_u = true * std + mean
    diff = pred_u - true_u
    return {
        "MSE": float(np.mean(diff**2)),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
    }


def find_single(directory: Path, name: str) -> Path:
    matches = list(directory.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {directory}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CEN-TaTS baseline with the published protocol.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--mode", choices=sorted(DATASETS), default="raw")
    parser.add_argument("--text_column", default="fact")
    parser.add_argument("--prior_weight", type=float)
    parser.add_argument("--train_epochs", type=int)
    parser.add_argument("--run_id", default="baseline", help="Name of the ignored training workspace")
    parser.add_argument("--python", default=PYTHON, help="Python interpreter for the child process")
    parser.add_argument("--llm_path", default=DEFAULT_GPT2_PATH, help="Local GPT-2 directory")
    parser.add_argument("--source_csv", type=Path, help="Override the mode's dataset path")
    parser.add_argument("--smoke-test", action="store_true", help="Run one CPU iTransformer forward pass and exit")
    args = parser.parse_args()

    from cen_ts.utils.paths import gpt2_path, project_path
    if args.smoke_test:
        from cen_ts.models import forward_smoke
        print(json.dumps({"status": "ok", "output_shape": forward_smoke()}))
        return 0
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    csv_path = project_path(args.source_csv or config["dataset"]["path"])
    prior_weight = args.prior_weight if args.prior_weight is not None else float(training["prior_weight"])
    train_epochs = args.train_epochs if args.train_epochs is not None else int(training["train_epochs"])
    run_id = args.run_id
    run_dir = (WORK_ROOT / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    command = [
        str(args.python),
        str(TATS_RUN),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", run_id,
        "--model", "iTransformer",
        "--data", "custom",
        "--root_path", str(csv_path.parent.resolve()),
        "--data_path", csv_path.name,
        "--features", "S",
        "--target", "OT",
        "--freq", "d",
        "--seq_len", "24",
        "--label_len", "12",
        "--pred_len", "48",
        "--text_emb", "12",
        "--seed", "2025",
        "--train_epochs", str(train_epochs),
        "--patience", str(training["patience"]),
        "--pool_type", "avg",
        "--llm_model", "GPT2",
        "--llm_path", str(gpt2_path(args.llm_path)),
        "--strict_local_llm", "True",
        "--d_model", "512",
        "--n_heads", "8",
        "--e_layers", "2",
        "--d_ff", "2048",
        "--dropout", "0.1",
        "--num_workers", "0",
        "--batch_size", str(training["batch_size"]),
        "--prior_weight", str(prior_weight),
        "--learning_rate", str(training["learning_rate"]),
        "--learning_rate2", "0.01",
        "--learning_rate3", "0.001",
        "--learning_rate_weight", "0.0001",
        "--text_mode", args.mode,
        "--text_column", args.text_column,
        "--save_root", str(WORK_ROOT.resolve()),
        "--run_name", run_id,
        "--prompt_version", "raw_text_baseline_v1",
        "--fail_on_missing_text", "True",
        "--record_input_hashes", "True",
    ]
    write_json(run_dir / "command.json", {"command": command, "cwd": str(run_dir)})
    started = time.time()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    runtime = {"seconds": time.time() - started, "returncode": completed.returncode}
    write_json(run_dir / "runtime.json", runtime)
    if completed.returncode != 0:
        raise RuntimeError(f"CEN-TaTS run failed; see {log_path}")

    result_metrics = find_single(run_dir / "results", "metrics.npy")
    result_pred = result_metrics.parent / "pred.npy"
    result_true = result_metrics.parent / "true.npy"
    pred = np.load(result_pred)
    true = np.load(result_true)
    metrics_arr = np.load(result_metrics)
    if pred.ndim != 3 or pred.shape[-1] != 1 or pred.shape != true.shape:
        raise AssertionError(f"Unexpected prediction/target shapes: {pred.shape}, {true.shape}")
    test_metrics = {
        "native_scaled": {
            "MAE": float(metrics_arr[0]),
            "MSE": float(metrics_arr[1]),
            "RMSE": float(metrics_arr[2]),
            "MAPE": float(metrics_arr[3]),
            "MSPE": float(metrics_arr[4]),
        },
        "unscaled": metrics_unscaled(pred, true, csv_path),
        "prediction_shape": list(pred.shape),
        "target_shape": list(true.shape),
    }
    write_json(run_dir / "test_metrics.json", test_metrics)
    rows = parse_epoch_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
    write_epoch_csv(run_dir / "epoch_metrics.csv", rows)
    checkpoint = find_single(run_dir / "checkpoints", "checkpoint.pth")
    (run_dir / "checkpoint_path.txt").write_text(str(checkpoint), encoding="utf-8")
    (run_dir / "checkpoint_sha256.txt").write_text(sha256_file(checkpoint) + "\n", encoding="utf-8")
    write_json(
        run_dir / "dataset_manifest.json",
        {
            "dataset_csv": str(csv_path),
            "dataset_sha256": sha256_file(csv_path),
            "text_mode": args.mode,
            "text_column": args.text_column,
            "prior_weight": prior_weight,
            "train_epochs": train_epochs,
        },
    )
    PUBLISHED_ROOT.mkdir(parents=True, exist_ok=True)
    flat_predictions = pd.DataFrame(
        {
            "window": np.repeat(np.arange(pred.shape[0]), pred.shape[1]),
            "horizon": np.tile(np.arange(pred.shape[1]), pred.shape[0]),
            "target_normalized": true.reshape(-1),
            "prediction_normalized": pred.reshape(-1),
        }
    )
    flat_predictions.to_csv(PUBLISHED_ROOT / "predictions.csv", index=False)
    published_metrics = {
        "experiment": "cen_tats_environment_baseline",
        "status": "PASS",
        "dataset": "Environment",
        "target": "OT",
        "seed": int(config["seed"]),
        "prediction_shape": list(pred.shape),
        "normalized": test_metrics["native_scaled"],
        "original_scale": test_metrics["unscaled"],
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    write_json(PUBLISHED_ROOT / "metrics.json", published_metrics)
    published_config = dict(config)
    published_config["training"] = dict(training)
    published_config["training"]["prior_weight"] = prior_weight
    published_config["training"]["train_epochs"] = train_epochs
    (PUBLISHED_ROOT / "config.yaml").write_text(
        yaml.safe_dump(published_config, sort_keys=False), encoding="utf-8"
    )
    log_lines = [
        "experiment=cen_tats_environment_baseline",
        f"dataset=Environment target=OT seed={config['seed']}",
        *(
            f"epoch={row['epoch']} train_loss={row['train_loss']:.7f} "
            f"val_loss={row['vali_loss']:.7f} test_mse={row['test_mse']:.7f} test_mae={row['test_mae']:.7f}"
            for row in rows
        ),
        "status=PASS paid_llm_calls=0",
    ]
    (PUBLISHED_ROOT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "metrics": test_metrics["native_scaled"], "runtime": runtime}, indent=2))
    return 0


if __name__ == "__main__":
    main()
