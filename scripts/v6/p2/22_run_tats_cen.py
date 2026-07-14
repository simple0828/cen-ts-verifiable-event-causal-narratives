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


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("D:/Miniconda/envs/tats/python.exe")
RUNS_ROOT = ROOT / "results" / "v6" / "p2" / "runs"
TATS_CEN_RUN = ROOT / "tats_cen" / "run.py"


DATASETS = {
    "raw": ROOT / "third_party" / "TaTS" / "data" / "Environment.csv",
    "constant": ROOT / "data" / "v6" / "p2" / "Environment_constant.csv",
    "shuffled": ROOT / "data" / "v6" / "p2" / "Environment_shuffled.csv",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated CEN-TaTS P2 experiments.")
    parser.add_argument("--mode", choices=sorted(DATASETS), default="raw")
    parser.add_argument("--text_column", default="fact")
    parser.add_argument("--prior_weight", type=float, default=0.5)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    csv_path = DATASETS[args.mode]
    run_id = args.run_id or f"p2_{args.mode}_pw{args.prior_weight}_e{args.train_epochs}_s2025"
    run_dir = (RUNS_ROOT / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    command = [
        str(PYTHON),
        str(TATS_CEN_RUN),
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
        "--train_epochs", str(args.train_epochs),
        "--patience", "5",
        "--pool_type", "avg",
        "--llm_model", "GPT2",
        "--llm_path", "D:/models/gpt2",
        "--strict_local_llm", "True",
        "--d_model", "512",
        "--n_heads", "8",
        "--e_layers", "2",
        "--d_ff", "2048",
        "--dropout", "0.1",
        "--num_workers", "0",
        "--batch_size", "32",
        "--prior_weight", str(args.prior_weight),
        "--learning_rate", "0.0001",
        "--learning_rate2", "0.01",
        "--learning_rate3", "0.001",
        "--learning_rate_weight", "0.0001",
        "--text_mode", args.mode,
        "--text_column", args.text_column,
        "--save_root", str(RUNS_ROOT.resolve()),
        "--run_name", run_id,
        "--prompt_version", "p2_minimal_fork_v1",
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
    np.save(run_dir / "predictions.npy", pred)
    np.save(run_dir / "targets.npy", true)
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
            "prior_weight": args.prior_weight,
            "train_epochs": args.train_epochs,
        },
    )
    print(json.dumps({"run_id": run_id, "metrics": test_metrics["native_scaled"], "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
