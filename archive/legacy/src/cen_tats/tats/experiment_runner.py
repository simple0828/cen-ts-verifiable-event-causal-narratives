from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cen_tats.evaluation.forecast_metrics import forecast_metrics
from cen_tats.io_utils import stable_hash, write_json
from cen_tats.tats.official_adapter import RandomInitGPT2InputEmbedding, TaTSForecastModel
from cen_tats.tats.text_cache import cache_key, load_embedding_cache, save_embedding_cache


@dataclass
class WindowBundle:
    x: np.ndarray
    y: np.ndarray
    text: np.ndarray
    anchors: np.ndarray
    split_names: list[str]
    scaler_mean: float
    scaler_std: float
    train_diff_std: float


def make_windows(values: np.ndarray, text_embeddings: np.ndarray, split, history: int, horizon: int, max_by_split: dict[str, int] | None = None) -> WindowBundle:
    values = np.asarray(values, dtype="float32")
    train_values = values[split.train_core]
    mean = float(np.mean(train_values))
    std = float(np.std(train_values) + 1e-8)
    scaled = (values - mean) / std
    train_diff_std = float(np.std(np.diff(train_values)) + 1e-8)
    buckets = {"train_core": [], "prompt_dev": [], "model_val": [], "test": []}
    for anchor in range(history - 1, len(values) - horizon):
        label_end = anchor + horizon
        assigned = None
        for name in buckets:
            arr = getattr(split, name)
            if len(arr) and int(arr[0]) <= anchor <= int(arr[-1]) and label_end <= int(arr[-1]):
                assigned = name
                break
        if assigned:
            buckets[assigned].append(anchor)
    rng = np.random.default_rng(2026)
    if max_by_split:
        for name, limit in max_by_split.items():
            if limit and len(buckets.get(name, [])) > limit:
                buckets[name] = sorted(rng.choice(buckets[name], size=limit, replace=False).tolist())
    anchors: list[int] = []
    split_names: list[str] = []
    for name in ["train_core", "prompt_dev", "model_val", "test"]:
        anchors.extend(buckets[name])
        split_names.extend([name] * len(buckets[name]))
    x = np.stack([scaled[a - history + 1 : a + 1] for a in anchors]).reshape(len(anchors), history, 1)
    y = np.stack([scaled[a + 1 : a + horizon + 1] for a in anchors]).reshape(len(anchors), horizon, 1)
    txt = np.stack([text_embeddings[a - history + 1 : a + 1] for a in anchors])
    return WindowBundle(x=x, y=y, text=txt, anchors=np.asarray(anchors), split_names=split_names, scaler_mean=mean, scaler_std=std, train_diff_std=train_diff_std)


def build_text_embeddings(dataset: str, texts: list[str], method: str, prompt_hash: str, cfg: dict) -> tuple[np.ndarray, dict]:
    params = {"max_length": cfg.get("max_token_length", 256), "pool_type": cfg.get("pooling", "avg"), "mode": cfg.get("lm_mode", "random_init")}
    key = cache_key(texts, prompt_hash, cfg.get("llm_model", "openai-community/gpt2"), method, dataset, params)
    cache_path = Path("artifacts/v5/text_embeddings") / f"{dataset}_{method}_{key}.npz"
    cached = load_embedding_cache(cache_path)
    if cached is not None:
        arr, meta = cached
        meta["cache_hit"] = True
        return arr, meta
    encoder = RandomInitGPT2InputEmbedding(seed=int(cfg.get("embedding_seed", 2026)), max_length=int(cfg.get("max_token_length", 256)))
    arr = encoder.encode(texts, batch_size=int(cfg.get("embedding_batch_size", 64)), pool_type=cfg.get("pooling", "avg"))
    meta = {
        "cache_key": key,
        "cache_hit": False,
        "cache_path": str(cache_path),
        "model_name": encoder.model_name,
        "lm_mode": encoder.mode,
        "prompt_hash": prompt_hash,
        "embedding_shape": list(arr.shape),
    }
    save_embedding_cache(cache_path, arr, meta)
    return arr, meta


def _indices(bundle: WindowBundle, split_name: str) -> np.ndarray:
    return np.asarray([i for i, s in enumerate(bundle.split_names) if s == split_name], dtype=int)


def train_eval_tats(bundle: WindowBundle, method: str, backbone: str, seed: int, cfg: dict, run_dir: str | Path, use_text: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", False) else "cpu")
    history = bundle.x.shape[1]
    horizon = bundle.y.shape[1]
    model = TaTSForecastModel(
        backbone=backbone,
        seq_len=history,
        label_len=int(cfg.get("label_len", max(1, history // 2))),
        pred_len=horizon,
        text_dim=int(cfg.get("text_emb_dim", 12)),
        llm_dim=bundle.text.shape[-1],
        use_text=use_text,
        d_model=int(cfg.get("d_model", 32)),
        n_heads=int(cfg.get("n_heads", 4)),
        e_layers=int(cfg.get("e_layers", 1)),
        d_ff=int(cfg.get("d_ff", 64)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    train_idx = _indices(bundle, "train_core")
    val_idx = _indices(bundle, "model_val")
    test_idx = _indices(bundle, "test")
    train_ds = TensorDataset(torch.tensor(bundle.x[train_idx]), torch.tensor(bundle.text[train_idx]), torch.tensor(bundle.y[train_idx]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 32)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    loss_fn = torch.nn.MSELoss()
    train_log: list[dict] = []
    start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for epoch in range(int(cfg.get("epochs", 1))):
        model.train()
        losses = []
        for xb, tb, yb in train_loader:
            xb = xb.float().to(device)
            tb = tb.float().to(device)
            yb = yb.float().to(device)
            opt.zero_grad()
            pred = model(xb, tb if use_text else None)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_log.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else float("nan")})
    model.eval()

    def predict(indices: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for start_i in range(0, len(indices), int(cfg.get("batch_size", 32))):
                idx = indices[start_i : start_i + int(cfg.get("batch_size", 32))]
                xb = torch.tensor(bundle.x[idx]).float().to(device)
                tb = torch.tensor(bundle.text[idx]).float().to(device)
                preds.append(model(xb, tb if use_text else None).detach().cpu().numpy())
        return np.vstack(preds) if preds else np.empty((0, horizon, 1))

    val_pred = predict(val_idx)
    test_pred = predict(test_idx)
    y_val = bundle.y[val_idx]
    y_test = bundle.y[test_idx]
    metrics = forecast_metrics(y_test.squeeze(-1), test_pred.squeeze(-1), train_diff_std=bundle.train_diff_std / bundle.scaler_std)
    val_metrics = forecast_metrics(y_val.squeeze(-1), val_pred.squeeze(-1), train_diff_std=bundle.train_diff_std / bundle.scaler_std) if len(val_idx) else {}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", y_true=y_test, y_pred=test_pred, anchors=bundle.anchors[test_idx])
    torch.save(model.state_dict(), run_dir / "checkpoint.pth")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    with open(run_dir / "train.log", "w", encoding="utf-8") as f:
        for row in train_log:
            f.write(json.dumps(row) + "\n")
    (run_dir / "selected_events.jsonl").write_text(json.dumps({"method": method, "note": "Selected events are stored in artifacts/v5/verifier and artifacts/v5/events; this run consumes the method text variant."}) + "\n", encoding="utf-8")
    (run_dir / "narratives.jsonl").write_text(json.dumps({"method": method, "note": "Narrative source artifacts are stored under artifacts/v5/narratives."}) + "\n", encoding="utf-8")
    write_json(run_dir / "prompt_versions.json", {"method": method, "prompt_source": "prompts/v5/manual or prompts/v5/apo depending on method", "prompt_hash": stable_hash(method)})
    write_json(run_dir / "causal_graph_version.json", {"graph_source": "artifacts/v5/causal_graph", "graph_hash": "recorded_in_artifact_json"})
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    runtime = time.time() - start
    git_commit = _cmd(["git", "rev-parse", "HEAD"])
    tats_commit = _cmd(["git", "-c", f"safe.directory={Path('third_party/TaTS').resolve().as_posix()}", "-C", "third_party/TaTS", "rev-parse", "HEAD"])
    manifest = {
        "git_commit": git_commit,
        "tats_commit": tats_commit,
        "run_id": run_dir.name,
        "method": method,
        "backbone": backbone,
        "seed": seed,
        "history": history,
        "horizon": horizon,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(0),
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_sec": runtime,
        "text_channels_used": bool(use_text),
        "official_backbone_imported_from": "third_party/TaTS/models",
        "checkpoint_path": str(run_dir / "checkpoint.pth"),
        "prediction_path": str(run_dir / "predictions.npz"),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", {"test": metrics, "model_val": val_metrics})
    return {
        **metrics,
        "model_val_MSE": val_metrics.get("MSE", float("nan")),
        "method": method,
        "backbone": backbone,
        "seed": seed,
        "run_id": run_dir.name,
        "runtime_sec": manifest["runtime_sec"],
        "n_test_windows": int(len(test_idx)),
        "prediction_path": str(run_dir / "predictions.npz"),
        "error_vector": ((test_pred.squeeze(-1) - y_test.squeeze(-1)) ** 2).mean(axis=1).tolist() if len(test_idx) else [],
    }


def run_id_for(dataset: str, method: str, backbone: str, horizon: int, seed: int) -> str:
    return f"{dataset}_{method}_{backbone}_H{horizon}_S{seed}_{stable_hash({'d': dataset, 'm': method, 'b': backbone, 'h': horizon, 's': seed}, 8)}"


def _cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"unavailable:{exc}"
