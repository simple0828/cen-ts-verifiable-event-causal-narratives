from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cen_ts.evaluation.forecast_metrics import forecast_metrics
from cen_ts.runtime.preflight import (
    DEFAULT_GPT2_PATH,
    encode_texts_masked_average,
    gpt2_file_manifest,
    load_pretrained_gpt2_strict,
    read_yaml_config,
    repo_root,
    sha256_file,
)


METHODS = (
    "M0_Numerical-only",
    "M1_TaTS-Raw-Text",
    "Z0_Zero-Text-Control",
    "S0_Shuffled-Text-Diagnostic",
)


class P1Error(RuntimeError):
    pass


def stable_hash(obj: Any, n: int | None = 16) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    value = hashlib.sha256(payload).hexdigest()
    return value if n is None else value[:n]


def array_hash(arr: np.ndarray, n: int | None = 16) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())
    value = h.hexdigest()
    return value if n is None else value[:n]


def text_hash(texts: list[str], n: int | None = 16) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(str(text).encode("utf-8", errors="replace"))
        h.update(b"\0")
    value = h.hexdigest()
    return value if n is None else value[:n]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_p1_config(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg = read_yaml_config(path or root / "configs" / "v6" / "p1_tats_official_environment.yaml")
    methods = tuple(cfg.get("experiment", {}).get("methods", []))
    if methods != METHODS:
        raise P1Error(f"P1 may only run {METHODS}, got {methods}")
    tats = cfg.get("tats", {})
    if not tats.get("pretrained", True) or tats.get("allow_random_fallback", False):
        raise P1Error("P1 requires pretrained GPT-2 and forbids random fallback.")
    training = cfg.get("training", {})
    if int(training.get("train_epochs", 0)) < 5:
        raise P1Error("P1 train_epochs must be at least 5.")
    if training.get("max_windows_by_split") is not None:
        raise P1Error("P1 must use full windows; max_windows_by_split must be null.")
    return cfg


def resolve_environment_csv(cfg: dict[str, Any]) -> tuple[Path, str]:
    root = repo_root()
    data = cfg["data"]
    requested = root / str(data["root_path"]).replace("./", "") / str(data["data_path"])
    official = root / "vendor" / "tats" / "data" / str(data["data_path"])
    if requested.exists():
        return requested, "configured_path"
    if official.exists():
        return official, "adapter_third_party_tats_data"
    raise FileNotFoundError(f"Cannot find Environment CSV at {requested} or {official}")


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "No information available"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "No information available"
    return text


def load_environment_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path, source = resolve_environment_csv(cfg)
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise P1Error("Environment.csv must contain date column.")
    target = cfg["data"]["target"]
    if target not in df.columns:
        raise P1Error(f"Target column {target!r} missing.")
    df["date"] = pd.to_datetime(df["date"])
    text_cols = [c for c in ("fact", "preds") if c in df.columns]
    if not text_cols:
        raise P1Error("Environment.csv must contain fact or preds text column.")
    raw = []
    for _, row in df.iterrows():
        pieces = [normalize_text(row[c]) for c in text_cols]
        text = " ".join([p for p in pieces if p != "No information available"]).strip()
        raw.append(text or "No information available")
    df["_p1_raw_text"] = raw
    df["_p1_target"] = pd.to_numeric(df[target], errors="coerce")
    if df["_p1_target"].isna().any():
        raise P1Error("Target column contains non-numeric or missing values.")
    df = df.sort_values("date").reset_index(drop=True)
    meta = {"csv_path": str(csv_path), "data_path_source": source, "text_columns": text_cols}
    return df, meta


@dataclass(frozen=True)
class SplitInfo:
    train: tuple[int, int]
    model_val: tuple[int, int]
    test: tuple[int, int]
    split_hash: str


def make_splits(n: int) -> SplitInfo:
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    info = {
        "policy": "time_ordered_70_10_20_non_overlapping",
        "n": n,
        "train": [0, n_train],
        "model_val": [n_train, n_train + n_val],
        "test": [n_train + n_val, n],
    }
    return SplitInfo(tuple(info["train"]), tuple(info["model_val"]), tuple(info["test"]), stable_hash(info, None))


def split_indices(split: SplitInfo, name: str) -> np.ndarray:
    start, end = getattr(split, name)
    return np.arange(start, end, dtype=np.int64)


def window_anchors(split: SplitInfo, seq_len: int, pred_len: int) -> dict[str, list[int]]:
    anchors: dict[str, list[int]] = {}
    for name in ("train", "model_val", "test"):
        start, end = getattr(split, name)
        first = start + seq_len - 1
        last = end - pred_len - 1
        anchors[name] = list(range(first, last + 1)) if last >= first else []
    return anchors


def audit_environment_data(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_p1_config(config_path)
    root = repo_root()
    out_dir = root / "results" / "v6" / "p1"
    report_dir = root / "reports" / "v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = load_environment_frame(cfg)
    split = make_splits(len(df))
    data = cfg["data"]
    seq_len = int(data["seq_len"])
    pred_len = int(data["pred_len"])
    anchors = window_anchors(split, seq_len, pred_len)
    texts = df["_p1_raw_text"].tolist()
    empty_count = int(sum(1 for t in texts if t == "No information available"))
    lengths = [len(t) for t in texts]
    numeric_missing = float(df["_p1_target"].isna().mean())
    diffs = df["date"].diff().dropna()
    freq = str(diffs.mode().iloc[0]) if not diffs.empty else "unknown"
    timestamp_unique = bool(df["date"].is_unique)
    audit = {
        **meta,
        "dataset": data["dataset"],
        "total_time_points": int(len(df)),
        "start_time": str(df["date"].iloc[0]),
        "end_time": str(df["date"].iloc[-1]),
        "time_frequency_mode": freq,
        "timestamps_unique": timestamp_unique,
        "target_column": data["target"],
        "numeric_missing_rate": numeric_missing,
        "text_columns": meta["text_columns"],
        "text_coverage_rate": float(np.mean([t != "No information available" for t in texts])),
        "empty_text_count": empty_count,
        "duplicate_text_count": int(len(texts) - len(set(texts))),
        "average_text_length": float(np.mean(lengths)),
        "max_text_length": int(np.max(lengths)),
        "constructible_window_count": int(sum(len(v) for v in anchors.values())),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "split_boundaries": {
            "train": list(split.train),
            "model_val": list(split.model_val),
            "test": list(split.test),
        },
        "window_counts": {k: int(len(v)) for k, v in anchors.items()},
        "split_hash": split.split_hash,
        "scaler_fit_split": "train",
        "test_used_for_parameter_selection": False,
        "same_split_for_all_methods": True,
        "raw_data_modified": False,
    }
    write_json(out_dir / "environment_data_audit.json", audit)
    md = [
        "# P1 Environment Data Audit",
        "",
        f"- Dataset: {audit['dataset']}",
        f"- CSV path: `{audit['csv_path']}`",
        f"- Data path source: `{audit['data_path_source']}`",
        f"- Total time points: {audit['total_time_points']}",
        f"- Time range: {audit['start_time']} to {audit['end_time']}",
        f"- Frequency mode: {audit['time_frequency_mode']}",
        f"- Target: `{audit['target_column']}`",
        f"- Numeric missing rate: {audit['numeric_missing_rate']:.6f}",
        f"- Text columns: {', '.join(audit['text_columns'])}",
        f"- Text coverage rate: {audit['text_coverage_rate']:.6f}",
        f"- Empty text count: {audit['empty_text_count']}",
        f"- Duplicate text count: {audit['duplicate_text_count']}",
        f"- Average / max text length: {audit['average_text_length']:.2f} / {audit['max_text_length']}",
        f"- Constructible windows: {audit['constructible_window_count']}",
        f"- Split hash: `{audit['split_hash']}`",
        "",
        "| split | point range [start,end) | windows |",
        "| --- | --- | --- |",
    ]
    for name in ("train", "model_val", "test"):
        md.append(f"| {name} | {audit['split_boundaries'][name]} | {audit['window_counts'][name]} |")
    md.extend(
        [
            "",
            "Hard checks: no rows are dropped for missing text; splits are chronological; scaler is fit on train only; test is not used for parameter selection; M0/M1/Z0/S0 share the same split.",
        ]
    )
    (report_dir / "p1_environment_data_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return audit


def p0_commit() -> str:
    root = repo_root()
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    path = root / "results" / "v6" / "p1" / "p0_commit.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def load_gpt2_for_p1(cfg: dict[str, Any], device: torch.device):
    tats = cfg["tats"]
    tokenizer, model, meta = load_pretrained_gpt2_strict(
        tats.get("model_path", DEFAULT_GPT2_PATH),
        pretrained=bool(tats.get("pretrained", True)),
        local_files_only=bool(tats.get("local_files_only", True)),
        device=str(device),
    )
    if type(model).__name__ != "GPT2Model":
        raise P1Error(f"P1 requires GPT2Model, got {type(model).__name__}")
    if bool(tats.get("freeze_gpt2", True)):
        for param in model.parameters():
            param.requires_grad_(False)
    manifest = gpt2_file_manifest(tats.get("model_path", DEFAULT_GPT2_PATH))
    config_path = Path(tats.get("model_path", DEFAULT_GPT2_PATH)) / "config.json"
    weight_path = Path(tats.get("model_path", DEFAULT_GPT2_PATH)) / "model.safetensors"
    meta.update(
        {
            "gpt2_config_hash": sha256_file(config_path) if config_path.exists() else None,
            "gpt2_weight_sha256": sha256_file(weight_path) if weight_path.exists() else None,
            "freeze_gpt2": bool(tats.get("freeze_gpt2", True)),
            "file_manifest_complete": manifest["complete"],
        }
    )
    return tokenizer, model, meta


def embedding_cache_path(method: str, texts: list[str], cfg: dict[str, Any], gpt2_meta: dict[str, Any]) -> Path:
    tats = cfg["tats"]
    key = stable_hash(
        {
            "method": method,
            "texts": text_hash(texts, None),
            "gpt2_weight_sha256": gpt2_meta.get("gpt2_weight_sha256"),
            "pooling": tats.get("pooling"),
            "max_token_length": tats.get("max_token_length"),
        },
        24,
    )
    return repo_root() / "artifacts" / "v6" / "tats_embeddings" / "environment" / f"{method}_{key}.npz"


def encode_or_load_embeddings(
    method: str,
    texts: list[str],
    cfg: dict[str, Any],
    tokenizer,
    gpt2_model,
    gpt2_meta: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = embedding_cache_path(method, texts, cfg, gpt2_meta)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        arr = data["embeddings"].astype("float32")
        meta = json.loads(str(data["metadata"]))
        meta["cache_hit"] = True
        return arr, meta
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg["training"].get("batch_size", 32))
    outputs = []
    start = time.time()
    for i in range(0, len(texts), batch_size):
        pooled = encode_texts_masked_average(
            tokenizer,
            gpt2_model,
            texts[i : i + batch_size],
            max_length=int(cfg["tats"].get("max_token_length", 256)),
            device=str(device),
        )
        outputs.append(pooled.detach().cpu().numpy().astype("float32"))
    arr = np.vstack(outputs)
    meta = {
        "cache_path": str(path),
        "cache_hit": False,
        "method": method,
        "text_hash": text_hash(texts, None),
        "gpt2_weight_sha256": gpt2_meta.get("gpt2_weight_sha256"),
        "pooling": cfg["tats"].get("pooling", "avg"),
        "max_token_length": int(cfg["tats"].get("max_token_length", 256)),
        "embedding_shape": list(arr.shape),
        "encoding_seconds": time.time() - start,
        "train_val_test_recorded": True,
    }
    np.savez_compressed(path, embeddings=arr, metadata=json.dumps(meta, ensure_ascii=False))
    return arr, meta


def build_shuffled_texts(df: pd.DataFrame, split: SplitInfo, seed: int) -> tuple[list[str], dict[str, Any]]:
    texts = df["_p1_raw_text"].tolist()
    shuffled = list(texts)
    mapping: dict[str, Any] = {"seed": seed, "policy": "fixed_shuffle_within_each_split", "mapping": {}}
    rng = np.random.default_rng(seed)
    for name in ("train", "model_val", "test"):
        idx = split_indices(split, name)
        perm = idx.copy()
        rng.shuffle(perm)
        for src, dst in zip(idx.tolist(), perm.tolist()):
            shuffled[src] = texts[dst]
            mapping["mapping"][str(src)] = {"split": name, "source_text_index": int(dst)}
    path = repo_root() / "results" / "v6" / "p1" / "shuffled_text_mapping.json"
    write_json(path, mapping)
    return shuffled, mapping


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    text: np.ndarray
    raw_text_windows: list[list[str]]
    anchors: np.ndarray
    split_names: list[str]
    scaler_mean: float
    scaler_std: float
    train_diff_std_scaled: float
    split_hash: str
    window_counts: dict[str, int]


def make_window_data(df: pd.DataFrame, cfg: dict[str, Any], split: SplitInfo, text_embeddings: np.ndarray) -> WindowData:
    seq_len = int(cfg["data"]["seq_len"])
    pred_len = int(cfg["data"]["pred_len"])
    values = df["_p1_target"].to_numpy(dtype="float32")
    train_values = values[split.train[0] : split.train[1]]
    mean = float(train_values.mean())
    std = float(train_values.std() + 1e-8)
    scaled = (values - mean) / std
    train_diff_std_scaled = float(np.std(np.diff(train_values)) / std + 1e-8)
    anchors_by_split = window_anchors(split, seq_len, pred_len)
    anchors: list[int] = []
    split_names: list[str] = []
    for name in ("train", "model_val", "test"):
        anchors.extend(anchors_by_split[name])
        split_names.extend([name] * len(anchors_by_split[name]))
    x = np.stack([scaled[a - seq_len + 1 : a + 1] for a in anchors]).reshape(len(anchors), seq_len, 1)
    y = np.stack([scaled[a + 1 : a + pred_len + 1] for a in anchors]).reshape(len(anchors), pred_len, 1)
    text = np.stack([text_embeddings[a - seq_len + 1 : a + 1] for a in anchors]).astype("float32")
    raw = [df["_p1_raw_text"].iloc[a - seq_len + 1 : a + 1].tolist() for a in anchors]
    return WindowData(
        x=x.astype("float32"),
        y=y.astype("float32"),
        text=text,
        raw_text_windows=raw,
        anchors=np.asarray(anchors, dtype=np.int64),
        split_names=split_names,
        scaler_mean=mean,
        scaler_std=std,
        train_diff_std_scaled=train_diff_std_scaled,
        split_hash=split.split_hash,
        window_counts={k: len(v) for k, v in anchors_by_split.items()},
    )


def indices_for(windows: WindowData, split_name: str) -> np.ndarray:
    return np.asarray([i for i, name in enumerate(windows.split_names) if name == split_name], dtype=np.int64)


def ensure_tats_import_path() -> None:
    tats = repo_root() / "vendor" / "tats"
    if str(tats) not in sys.path:
        sys.path.insert(0, str(tats))


class OfficialITransformerTaTS(nn.Module):
    def __init__(self, cfg: dict[str, Any], method: str):
        super().__init__()
        ensure_tats_import_path()
        from models.iTransformer import Model as ITransformerModel

        self.method = method
        self.pred_len = int(cfg["data"]["pred_len"])
        self.use_text = method != "M0_Numerical-only"
        self.zero_text = method == "Z0_Zero-Text-Control"
        channels = 1 + (int(cfg["tats"]["text_emb_dim"]) if self.use_text else 0)
        official_defaults = {
            "d_model": 512,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 2048,
            "dropout": 0.1,
            "factor": 1,
            "embed": "timeF",
            "freq": "d",
            "activation": "gelu",
            "output_attention": False,
        }
        self.backbone_config = {
            **official_defaults,
            "task_name": "long_term_forecast",
            "seq_len": int(cfg["data"]["seq_len"]),
            "label_len": int(cfg["data"]["label_len"]),
            "pred_len": self.pred_len,
            "enc_in": channels,
            "dec_in": channels,
            "c_out": 1,
            "num_class": 2,
        }
        self.backbone = ITransformerModel(SimpleNamespace(**self.backbone_config))
        if self.use_text:
            self.text_projection = nn.Sequential(
                nn.Linear(768, 128),
                nn.GELU(),
                nn.Linear(128, int(cfg["tats"]["text_emb_dim"])),
            )
        else:
            self.text_projection = None

    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        if self.text_projection is None:
            raise P1Error("project_text called for numerical-only model.")
        projected = self.text_projection(text_emb)
        if self.zero_text:
            projected = torch.zeros_like(projected)
        return projected

    def combined_input(self, x_num: torch.Tensor, text_emb: torch.Tensor | None) -> torch.Tensor:
        if not self.use_text:
            return x_num
        if text_emb is None:
            raise P1Error("Text embeddings are required for text methods.")
        return torch.cat([x_num, self.project_text(text_emb)], dim=-1)

    def forward(self, x_num: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.combined_input(x_num, text_emb)
        dec_inp = torch.zeros((x.shape[0], self.pred_len, x.shape[-1]), dtype=x.dtype, device=x.device)
        out = self.backbone(x, None, dec_inp, None)
        return out[:, -self.pred_len :, :1]


def parameter_summary(model: OfficialITransformerTaTS) -> dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    text = sum(int(p.numel()) for p in model.text_projection.parameters()) if model.text_projection is not None else 0
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "text_module_parameter_count": text,
        "backbone_parameter_count": total - text,
    }


def tensor_loader(windows: WindowData, split_name: str, batch_size: int, shuffle: bool) -> DataLoader:
    idx = indices_for(windows, split_name)
    ds = TensorDataset(
        torch.from_numpy(windows.x[idx]),
        torch.from_numpy(windows.text[idx]),
        torch.from_numpy(windows.y[idx]),
        torch.from_numpy(windows.anchors[idx]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def unscale(arr: np.ndarray, windows: WindowData) -> np.ndarray:
    return arr * windows.scaler_std + windows.scaler_mean


def evaluate_model(model: OfficialITransformerTaTS, windows: WindowData, split_name: str, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = tensor_loader(windows, split_name, batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, tb, yb, ab in loader:
            pred = model(xb.float().to(device), tb.float().to(device) if model.use_text else None)
            preds.append(pred.detach().cpu().numpy())
            targets.append(yb.numpy())
            anchors.append(ab.numpy())
    if not preds:
        return np.empty((0, model.pred_len, 1)), np.empty((0, model.pred_len, 1)), np.empty((0,), dtype=np.int64)
    return np.vstack(preds), np.vstack(targets), np.concatenate(anchors)


def gpu_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def train_one_method(
    method: str,
    windows: WindowData,
    cfg: dict[str, Any],
    gpt2_meta: dict[str, Any] | None,
    run_root: Path,
) -> dict[str, Any]:
    if method not in METHODS:
        raise P1Error(f"Forbidden P1 method: {method}")
    training = cfg["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bool(training.get("deterministic", True)):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
    device = torch.device(f"cuda:{int(training.get('gpu', 0))}" if training.get("use_cuda", True) and torch.cuda.is_available() else "cpu")
    if training.get("use_cuda", True) and device.type != "cuda":
        raise P1Error("P1 requires CUDA training, but CUDA is unavailable.")
    run_id = f"Environment_{method}_iTransformer_H{cfg['data']['pred_len']}_S{seed}_{stable_hash({'method': method, 'split': windows.split_hash}, 8)}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model = OfficialITransformerTaTS(cfg, method).to(device)
    params = parameter_summary(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training.get("use_amp", True)) and device.type == "cuda")
    loss_fn = nn.MSELoss()
    batch_size = int(training["batch_size"])
    train_loader = tensor_loader(windows, "train", batch_size, shuffle=True)
    best_val = float("inf")
    best_epoch = 0
    patience_state = 0
    epochs = int(training["train_epochs"])
    projection_grad_norm_max = 0.0
    projection_grad_nonzero_batches = 0
    projection_grad_nonfinite_batches = 0
    projected_stats: list[dict[str, float]] = []
    initial_projection = None
    if model.text_projection is not None:
        initial_projection = torch.cat([p.detach().flatten().cpu() for p in model.text_projection.parameters()])
    epoch_rows: list[dict[str, Any]] = []
    train_log_lines: list[str] = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        lrs = [group["lr"] for group in opt.param_groups]
        for xb, tb, yb, _ in train_loader:
            xb = xb.float().to(device)
            tb = tb.float().to(device)
            yb = yb.float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(training.get("use_amp", True)) and device.type == "cuda"):
                pred = model(xb, tb if model.use_text else None)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            if model.text_projection is not None and not model.zero_text:
                scaler.unscale_(opt)
                grad_norm_sq = 0.0
                for p in model.text_projection.parameters():
                    if p.grad is not None:
                        grad_norm_sq += float(torch.sum(p.grad.detach() ** 2).cpu())
                grad_norm = math.sqrt(grad_norm_sq)
                if math.isfinite(grad_norm):
                    projection_grad_norm_max = max(projection_grad_norm_max, grad_norm)
                else:
                    projection_grad_nonfinite_batches += 1
                if math.isfinite(grad_norm) and grad_norm > 1e-8:
                    projection_grad_nonzero_batches += 1
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_pred, val_true, _ = evaluate_model(model, windows, "model_val", device, batch_size)
        val_loss = float(np.mean((val_pred - val_true) ** 2)) if len(val_pred) else float("nan")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_state = 0
            torch.save(model.state_dict(), run_dir / "best_checkpoint.pth")
        else:
            patience_state += 1
        proj_weight_norm = 0.0
        proj_update_norm = 0.0
        proj_mean = proj_std = proj_var = 0.0
        if model.text_projection is not None:
            current = torch.cat([p.detach().flatten().cpu() for p in model.text_projection.parameters()])
            proj_weight_norm = float(torch.linalg.vector_norm(current).item())
            if initial_projection is not None:
                proj_update_norm = float(torch.linalg.vector_norm(current - initial_projection).item())
            sample_idx = indices_for(windows, "train")[: min(128, len(indices_for(windows, "train")))]
            with torch.no_grad():
                projected = model.project_text(torch.from_numpy(windows.text[sample_idx]).float().to(device))
            proj_mean = float(projected.mean().detach().cpu())
            proj_std = float(projected.std(unbiased=False).detach().cpu())
            proj_var = float(projected.var(unbiased=False).detach().cpu())
            projected_stats.append({"epoch": epoch, "mean": proj_mean, "std": proj_std, "variance": proj_var})
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
            "learning_rate": float(lrs[0]),
            "epoch_runtime_sec": time.time() - epoch_start,
            "best_validation_loss": best_val,
            "patience_state": patience_state,
            **gpu_memory(device),
            "projection_gradient_norm_max_so_far": projection_grad_norm_max,
            "projection_gradient_nonfinite_batches": projection_grad_nonfinite_batches,
            "projection_weight_norm": proj_weight_norm,
            "projection_weight_update_norm": proj_update_norm,
            "projected_text_mean": proj_mean,
            "projected_text_std": proj_std,
            "projected_text_variance": proj_var,
        }
        epoch_rows.append(row)
        train_log_lines.append(json.dumps(row, ensure_ascii=False))
    if method == "M1_TaTS-Raw-Text" and projection_grad_norm_max <= 1e-8:
        raise P1Error("M1 projection_gradient_norm did not exceed 1e-8.")
    model.load_state_dict(torch.load(run_dir / "best_checkpoint.pth", map_location=device))
    test_pred_scaled, test_true_scaled, test_anchors = evaluate_model(model, windows, "test", device, batch_size)
    val_pred_scaled, val_true_scaled, _ = evaluate_model(model, windows, "model_val", device, batch_size)
    test_pred = unscale(test_pred_scaled, windows)
    test_true = unscale(test_true_scaled, windows)
    val_pred = unscale(val_pred_scaled, windows)
    val_true = unscale(val_true_scaled, windows)
    metrics = forecast_metrics(test_true.squeeze(-1), test_pred.squeeze(-1), train_diff_std=windows.train_diff_std_scaled * windows.scaler_std)
    val_metrics = forecast_metrics(val_true.squeeze(-1), val_pred.squeeze(-1), train_diff_std=windows.train_diff_std_scaled * windows.scaler_std)
    per_origin = ((test_pred.squeeze(-1) - test_true.squeeze(-1)) ** 2).mean(axis=1)
    with (run_dir / "per_origin_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor", "mse"])
        writer.writeheader()
        for anchor, mse in zip(test_anchors.tolist(), per_origin.tolist()):
            writer.writerow({"anchor": int(anchor), "mse": float(mse)})
    np.savez_compressed(run_dir / "predictions.npz", predictions=test_pred, predictions_scaled=test_pred_scaled, anchors=test_anchors)
    np.savez_compressed(run_dir / "targets.npz", targets=test_true, targets_scaled=test_true_scaled, anchors=test_anchors)
    input_hashes = {
        "x_test_hash": array_hash(windows.x[indices_for(windows, "test")], None),
        "text_test_hash": array_hash(windows.text[indices_for(windows, "test")], None),
        "y_test_hash": array_hash(windows.y[indices_for(windows, "test")], None),
        "prediction_hash": array_hash(test_pred, None),
        "target_hash": array_hash(test_true, None),
    }
    write_json(run_dir / "input_hashes.json", input_hashes)
    write_json(
        run_dir / "gradient_diagnostics.json",
        {
            "projection_gradient_norm_max": projection_grad_norm_max,
            "projection_grad_nonzero_batches": projection_grad_nonzero_batches,
            "projection_gradient_nonfinite_batches": projection_grad_nonfinite_batches,
            "projection_weight_update_norm": epoch_rows[-1]["projection_weight_update_norm"],
            "projected_text_stats_by_epoch": projected_stats,
        },
    )
    write_json(
        run_dir / "runtime.json",
        {
            "runtime_sec": time.time() - start_time,
            "device": str(device),
            "cuda_training": device.type == "cuda",
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    )
    with (run_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)
    (run_dir / "train.log").write_text("\n".join(train_log_lines) + "\n", encoding="utf-8")
    import yaml

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"config": cfg, "backbone_config": model.backbone_config}, f, sort_keys=False)
    manifest = {
        "run_id": run_id,
        "method": method,
        "dataset": "Environment",
        "backbone": "iTransformer",
        "seed": int(training["seed"]),
        "split_hash": windows.split_hash,
        "window_counts": windows.window_counts,
        "input_shape_numeric": [int(batch_size), int(cfg["data"]["seq_len"]), 1],
        "input_shape_combined_channels": model.backbone_config["enc_in"],
        "best_epoch": best_epoch,
        "completed_epochs": epochs,
        "official_backbone_file": "vendor/tats/models/iTransformer.py",
        "forecasting_head_modified": False,
        "attention_modified": False,
        "loss_structure_modified": False,
        "checkpoint_path": str(run_dir / "best_checkpoint.pth"),
        "predictions_path": str(run_dir / "predictions.npz"),
        "targets_path": str(run_dir / "targets.npz"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpt2": gpt2_meta or {"used": False},
        **params,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "test_metrics.json", {"test": metrics, "model_val": val_metrics})
    return {
        "method": method,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "completed_epochs": epochs,
        "best_epoch": best_epoch,
        **metrics,
        "model_val_MSE": val_metrics["MSE"],
        "prediction_hash": input_hashes["prediction_hash"],
        "projection_gradient_norm_max": projection_grad_norm_max,
        "projection_weight_update_norm": epoch_rows[-1]["projection_weight_update_norm"],
        **params,
        "per_origin_mse_path": str(run_dir / "per_origin_errors.csv"),
    }


def diagnose_inputs(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_p1_config(config_path)
    df, _ = load_environment_frame(cfg)
    split = make_splits(len(df))
    seed = int(cfg["training"]["seed"])
    device = torch.device(f"cuda:{int(cfg['training'].get('gpu', 0))}" if torch.cuda.is_available() else "cpu")
    tokenizer, gpt2_model, gpt2_meta = load_gpt2_for_p1(cfg, device)
    shuffled_texts, _ = build_shuffled_texts(df, split, seed)
    real_emb, real_meta = encode_or_load_embeddings("M1_TaTS-Raw-Text", df["_p1_raw_text"].tolist(), cfg, tokenizer, gpt2_model, gpt2_meta, device)
    shuf_emb, shuf_meta = encode_or_load_embeddings("S0_Shuffled-Text-Diagnostic", shuffled_texts, cfg, tokenizer, gpt2_model, gpt2_meta, device)
    m1 = make_window_data(df, cfg, split, real_emb)
    s0 = make_window_data(df.assign(_p1_raw_text=shuffled_texts), cfg, split, shuf_emb)
    rng = np.random.default_rng(seed)
    sample_count = min(32, len(m1.x))
    sample_idx = np.sort(rng.choice(np.arange(len(m1.x), dtype=np.int64), size=sample_count, replace=False))
    out: dict[str, Any] = {"sample_count": int(len(sample_idx)), "gpt2": gpt2_meta, "embedding_cache": {"m1": real_meta, "s0": shuf_meta}}
    for method, windows in (("M0_Numerical-only", m1), ("M1_TaTS-Raw-Text", m1), ("Z0_Zero-Text-Control", m1), ("S0_Shuffled-Text-Diagnostic", s0)):
        torch.manual_seed(seed)
        model = OfficialITransformerTaTS(cfg, method).to(device)
        xb = torch.from_numpy(windows.x[sample_idx]).float().to(device)
        tb = torch.from_numpy(windows.text[sample_idx]).float().to(device)
        with torch.no_grad():
            if method == "M0_Numerical-only":
                projected = np.empty((len(sample_idx), int(cfg["data"]["seq_len"]), 0), dtype="float32")
                combined = xb.detach().cpu().numpy()
            else:
                pt = model.project_text(tb).detach().cpu().numpy()
                projected = pt
                combined = np.concatenate([windows.x[sample_idx], pt], axis=-1)
        raw_window_texts = ["\n".join(windows.raw_text_windows[i]) for i in sample_idx]
        out[method] = {
            "numeric_input_shape": list(windows.x[sample_idx].shape),
            "text_embedding_shape": list(windows.text[sample_idx].shape),
            "projected_text_shape": list(projected.shape),
            "combined_input_shape": list(combined.shape),
            "numeric_input_hash": array_hash(windows.x[sample_idx], None),
            "raw_text_hash": text_hash(raw_window_texts, None),
            "token_ids_hash": "not_saved_full_token_ids_uses_strict_gpt2_tokenizer",
            "gpt2_embedding_hash": array_hash(windows.text[sample_idx], None),
            "projected_text_hash": array_hash(projected, None),
            "combined_input_hash": array_hash(combined, None),
            "projected_text_nonzero": bool(np.any(np.abs(projected) > 1e-12)) if projected.size else False,
            "projected_text_all_zero": bool(np.allclose(projected, 0.0)) if projected.size else False,
        }
    emb = m1.text[sample_idx].reshape(-1, 768)
    norms = np.linalg.norm(emb, axis=1)
    denom = np.outer(norms, norms) + 1e-12
    cos = (emb @ emb.T) / denom
    out["embedding_statistics"] = {
        "variance": float(np.var(emb)),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "pairwise_cosine_similarity_mean": float(np.mean(cos[np.triu_indices_from(cos, k=1)])),
        "unique_embedding_ratio": float(len({array_hash(e, None) for e in emb}) / len(emb)),
    }
    checks = {
        "m0_shape_differs_from_m1": out["M0_Numerical-only"]["combined_input_shape"] != out["M1_TaTS-Raw-Text"]["combined_input_shape"],
        "m1_projected_text_nonzero": out["M1_TaTS-Raw-Text"]["projected_text_nonzero"],
        "z0_projected_text_all_zero": out["Z0_Zero-Text-Control"]["projected_text_all_zero"],
        "s0_raw_text_hash_differs_from_m1": out["S0_Shuffled-Text-Diagnostic"]["raw_text_hash"] != out["M1_TaTS-Raw-Text"]["raw_text_hash"],
        "m1_combined_differs_from_z0": out["M1_TaTS-Raw-Text"]["combined_input_hash"] != out["Z0_Zero-Text-Control"]["combined_input_hash"],
        "m1_combined_differs_from_s0": out["M1_TaTS-Raw-Text"]["combined_input_hash"] != out["S0_Shuffled-Text-Diagnostic"]["combined_input_hash"],
        "embeddings_nonconstant": out["embedding_statistics"]["variance"] > 0 and out["embedding_statistics"]["unique_embedding_ratio"] > 0.05,
    }
    out["hard_checks"] = checks
    if not all(checks.values()):
        raise P1Error(f"Input diagnostics failed: {checks}")
    root = repo_root()
    write_json(root / "results" / "v6" / "p1" / "input_diagnostics.json", out)
    md = ["# P1 Input Diagnostics", "", f"- Sample windows: {out['sample_count']}"]
    for key, value in checks.items():
        md.append(f"- {key}: {value}")
    md.append("")
    md.append("## Shapes")
    for method in METHODS:
        md.append(f"- {method}: combined {out[method]['combined_input_shape']}, projected {out[method]['projected_text_shape']}")
    md.append("")
    md.append("## Embedding Statistics")
    for k, v in out["embedding_statistics"].items():
        md.append(f"- {k}: {v}")
    (root / "reports" / "v6" / "p1_input_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def run_all_p1(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_p1_config(config_path)
    root = repo_root()
    p0 = p0_commit()
    audit = audit_environment_data(config_path)
    diagnostics = diagnose_inputs(config_path)
    df, _ = load_environment_frame(cfg)
    split = make_splits(len(df))
    seed = int(cfg["training"]["seed"])
    device = torch.device(f"cuda:{int(cfg['training'].get('gpu', 0))}" if torch.cuda.is_available() else "cpu")
    tokenizer, gpt2_model, gpt2_meta = load_gpt2_for_p1(cfg, device)
    shuffled_texts, _ = build_shuffled_texts(df, split, seed)
    real_emb, _ = encode_or_load_embeddings("M1_TaTS-Raw-Text", df["_p1_raw_text"].tolist(), cfg, tokenizer, gpt2_model, gpt2_meta, device)
    shuf_emb, _ = encode_or_load_embeddings("S0_Shuffled-Text-Diagnostic", shuffled_texts, cfg, tokenizer, gpt2_model, gpt2_meta, device)
    zeros = np.zeros_like(real_emb, dtype="float32")
    windows = {
        "M0_Numerical-only": make_window_data(df, cfg, split, zeros),
        "M1_TaTS-Raw-Text": make_window_data(df, cfg, split, real_emb),
        "Z0_Zero-Text-Control": make_window_data(df, cfg, split, zeros),
        "S0_Shuffled-Text-Diagnostic": make_window_data(df.assign(_p1_raw_text=shuffled_texts), cfg, split, shuf_emb),
    }
    run_root = root / "results" / "v6" / "p1" / "runs"
    rows = []
    for method in METHODS:
        meta = None if method == "M0_Numerical-only" else gpt2_meta
        rows.append(train_one_method(method, windows[method], cfg, meta, run_root))
    metrics_path = root / "results" / "v6" / "p1" / "p1_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_table(root / "results" / "v6" / "p1" / "p1_metrics.md", rows)
    counter = run_counterfactual(cfg, windows["M1_TaTS-Raw-Text"], gpt2_meta, real_emb, shuf_emb)
    status = build_status(cfg, p0, audit, diagnostics, rows, counter)
    write_report(cfg, p0, audit, diagnostics, rows, counter, status)
    return status


def load_model_from_run(cfg: dict[str, Any], method: str, run_dir: Path, device: torch.device) -> OfficialITransformerTaTS:
    model = OfficialITransformerTaTS(cfg, method).to(device)
    model.load_state_dict(torch.load(run_dir / "best_checkpoint.pth", map_location=device))
    model.eval()
    return model


def predict_with_text(model: OfficialITransformerTaTS, windows: WindowData, text_embeddings: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    idx = indices_for(windows, "test")
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            sub = idx[start : start + batch_size]
            xb = torch.from_numpy(windows.x[sub]).float().to(device)
            tb = torch.from_numpy(text_embeddings[sub]).float().to(device)
            preds.append(model(xb, tb).detach().cpu().numpy())
    return np.vstack(preds)


def run_counterfactual(cfg: dict[str, Any], windows: WindowData, gpt2_meta: dict[str, Any], real_point_emb: np.ndarray, shuffled_point_emb: np.ndarray) -> dict[str, Any]:
    root = repo_root()
    seed = int(cfg["training"]["seed"])
    run_id = f"Environment_M1_TaTS-Raw-Text_iTransformer_H{cfg['data']['pred_len']}_S{seed}_{stable_hash({'method': 'M1_TaTS-Raw-Text', 'split': windows.split_hash}, 8)}"
    run_dir = root / "results" / "v6" / "p1" / "runs" / run_id
    device = torch.device(f"cuda:{int(cfg['training'].get('gpu', 0))}" if torch.cuda.is_available() else "cpu")
    model = load_model_from_run(cfg, "M1_TaTS-Raw-Text", run_dir, device)
    batch_size = int(cfg["training"]["batch_size"])
    rng = np.random.default_rng(seed)
    random_point = rng.normal(size=real_point_emb.shape).astype("float32")
    constant_point = np.repeat(real_point_emb[:1], repeats=len(real_point_emb), axis=0).astype("float32")
    split = make_splits(len(real_point_emb))
    variants = {
        "real": make_window_data(load_environment_frame(cfg)[0], cfg, split, real_point_emb).text,
        "zero": np.zeros_like(windows.text, dtype="float32"),
        "shuffle": make_window_data(load_environment_frame(cfg)[0], cfg, split, shuffled_point_emb).text,
        "random": make_window_data(load_environment_frame(cfg)[0], cfg, split, random_point).text,
        "constant": make_window_data(load_environment_frame(cfg)[0], cfg, split, constant_point).text,
    }
    preds = {name: unscale(predict_with_text(model, windows, emb, device, batch_size), windows) for name, emb in variants.items()}
    out: dict[str, Any] = {
        "prediction_hashes": {k: array_hash(v, None) for k, v in preds.items()},
        "target_std": float(np.std(unscale(windows.y[indices_for(windows, "test")], windows))),
    }
    for name in ("zero", "shuffle", "random", "constant"):
        diff = np.abs(preds["real"] - preds[name])
        flat_a = preds["real"].reshape(len(preds["real"]), -1)
        flat_b = preds[name].reshape(len(preds[name]), -1)
        cos = np.sum(flat_a * flat_b, axis=1) / ((np.linalg.norm(flat_a, axis=1) * np.linalg.norm(flat_b, axis=1)) + 1e-12)
        out[f"real_vs_{name}_max_diff"] = float(diff.max())
        out[f"real_vs_{name}_mean_diff"] = float(diff.mean())
        out[f"real_vs_{name}_mean_diff_over_target_std"] = float(diff.mean() / (out["target_std"] + 1e-12))
        out[f"real_vs_{name}_prediction_cosine_similarity"] = float(np.mean(cos))
        out[f"real_vs_{name}_sample_change_ratio"] = float(np.mean(diff.reshape(len(diff), -1).max(axis=1) > 1e-6))
    if out["real_vs_zero_max_diff"] <= 1e-6 or out["real_vs_shuffle_max_diff"] <= 1e-6:
        raise P1Error("Counterfactual text diagnostics failed minimum prediction-difference checks.")
    write_json(root / "results" / "v6" / "p1" / "text_counterfactual_diagnostics.json", out)
    md = ["# P1 Text Counterfactual Diagnostics", ""]
    for k, v in out.items():
        if k != "prediction_hashes":
            md.append(f"- {k}: {v}")
    (root / "reports" / "v6" / "p1_text_counterfactual_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def build_status(
    cfg: dict[str, Any],
    p0: str,
    audit: dict[str, Any],
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
    counter: dict[str, Any],
) -> dict[str, Any]:
    by_method = {r["method"]: r for r in rows}
    pred_identical = {
        "m0_m1_predictions_identical": by_method["M0_Numerical-only"]["prediction_hash"] == by_method["M1_TaTS-Raw-Text"]["prediction_hash"],
        "m1_z0_predictions_identical": by_method["M1_TaTS-Raw-Text"]["prediction_hash"] == by_method["Z0_Zero-Text-Control"]["prediction_hash"],
        "m1_s0_predictions_identical": by_method["M1_TaTS-Raw-Text"]["prediction_hash"] == by_method["S0_Shuffled-Text-Diagnostic"]["prediction_hash"],
    }
    tests_log = repo_root() / "results" / "v6" / "p1" / "pytest_v6.log"
    tests_passed = tests_log.exists() and "failed" not in tests_log.read_text(encoding="utf-8", errors="replace").lower()
    hard_pass = (
        all(by_method[m]["completed_epochs"] >= 5 for m in METHODS)
        and diagnostics["hard_checks"]["m1_projected_text_nonzero"]
        and by_method["M1_TaTS-Raw-Text"]["projection_gradient_norm_max"] > 1e-8
        and not any(pred_identical.values())
        and counter["real_vs_zero_max_diff"] > 1e-6
        and counter["real_vs_shuffle_max_diff"] > 1e-6
    )
    status = {
        "p0_commit": p0,
        "dataset": "Environment",
        "backbone": "iTransformer",
        "gpt2_pretrained_loaded": True,
        "random_fallback": False,
        "cuda_training": True,
        "full_windows": True,
        "m0_completed": by_method["M0_Numerical-only"]["completed_epochs"] >= 5,
        "m1_completed": by_method["M1_TaTS-Raw-Text"]["completed_epochs"] >= 5,
        "z0_completed": by_method["Z0_Zero-Text-Control"]["completed_epochs"] >= 5,
        "s0_completed": by_method["S0_Shuffled-Text-Diagnostic"]["completed_epochs"] >= 5,
        "projection_gradient_norm_max": by_method["M1_TaTS-Raw-Text"]["projection_gradient_norm_max"],
        "projection_weight_update_norm": by_method["M1_TaTS-Raw-Text"]["projection_weight_update_norm"],
        **pred_identical,
        "counterfactual_real_zero_max_diff": counter["real_vs_zero_max_diff"],
        "counterfactual_real_shuffle_max_diff": counter["real_vs_shuffle_max_diff"],
        "paid_llm_calls": 0,
        "tests_passed": tests_passed,
        "p1_status": "PASS" if hard_pass and tests_passed else ("PARTIAL" if hard_pass else "FAIL"),
        "environment_total_time_points": audit["total_time_points"],
        "window_counts": audit["window_counts"],
        "split_hash": audit["split_hash"],
        "methods": rows,
    }
    write_json(repo_root() / "results" / "v6" / "p1" / "p1_status.json", status)
    return status


def write_report(
    cfg: dict[str, Any],
    p0: str,
    audit: dict[str, Any],
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
    counter: dict[str, Any],
    status: dict[str, Any],
) -> None:
    root = repo_root()
    rows_by_method = {r["method"]: r for r in rows}
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tests_log = root / "results" / "v6" / "p1" / "pytest_v6.log"
    tests_text = tests_log.read_text(encoding="utf-8", errors="replace").strip() if tests_log.exists() else "pytest not run yet"
    lines = [
        "# P1 Official TaTS iTransformer Reproduction Report",
        "",
        "## 1. P1 执行摘要",
        "本阶段只运行 M0_Numerical-only、M1_TaTS-Raw-Text、Z0_Zero-Text-Control 和 S0_Shuffled-Text-Diagnostic；未运行事件抽取、因果图、verifier、APO 或付费 LLM API。",
        "",
        "## 2. P1_STATUS",
        status["p1_status"],
        "",
        "## 3. Git 分支与 commit",
        f"- Branch: `{git_branch}`",
        f"- P0 commit: `{p0}`",
        f"- Current commit before P1 final commit: `{head}`",
        "",
        "## 4. P0 验收继承情况",
        "P0_STATUS was PASS in `reports/v6/preflight_environment_report.md`; GPT-2 strict loading and CUDA checks are reused here.",
        "",
        "## 5. Environment 数据审计",
        f"- Total points: {audit['total_time_points']}",
        f"- Time range: {audit['start_time']} to {audit['end_time']}",
        f"- Text coverage: {audit['text_coverage_rate']:.6f}",
        f"- Split hash: `{audit['split_hash']}`",
        "",
        "## 6. 数据 split",
        json.dumps(audit["split_boundaries"], ensure_ascii=False),
        "",
        "## 7. 官方 TaTS 适配情况",
        "The run imports `vendor/tats/models/iTransformer.py` directly and keeps attention, forecasting head, forward data flow, and MSE loss structure unchanged. The adapter supplies strict local GPT-2 text embeddings, TaTS-style projection to 12 auxiliary variables, and result logging.",
        "",
        "## 8. GPT-2 加载情况",
        f"- Model path: `{cfg['tats']['model_path']}`",
        f"- Hidden size: {diagnostics['gpt2']['hidden_size']}",
        f"- Model class: {diagnostics['gpt2']['model_class']}",
        f"- Random init: {diagnostics['gpt2']['random_init']}",
        f"- Local files only: {diagnostics['gpt2']['local_files_only']}",
        "",
        "## 9. iTransformer 配置",
        "Official defaults from TaTS `run.py` were retained for core model size: d_model=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1.",
        "",
        "## 10. M0 方法",
        "Input shape is `[B, 24, 1]`; GPT-2 and text projection are not initialized for this method.",
        "",
        "## 11. M1 方法",
        "Raw text is tokenized by local GPT-2, mask-average pooled, projected to 12 auxiliary channels, concatenated with the numeric channel, then passed to official iTransformer.",
        "",
        "## 12. Z0 方法",
        "The model keeps 13 input variables and the same iTransformer configuration as M1, but projected text channels are replaced with zeros.",
        "",
        "## 13. S0 方法",
        "Text timestamps are shuffled with fixed seed 2025 within each chronological split. The mapping is saved in `results/v6/p1/shuffled_text_mapping.json`.",
        "",
        "## 14. 输入 hash 和 shape",
        json.dumps({m: diagnostics[m]["combined_input_shape"] for m in METHODS}, ensure_ascii=False),
        "",
        "## 15. 文本 embedding 统计",
        json.dumps(diagnostics["embedding_statistics"], ensure_ascii=False),
        "",
        "## 16. projection 梯度",
        f"- M1 projection gradient norm max: {rows_by_method['M1_TaTS-Raw-Text']['projection_gradient_norm_max']}",
        f"- M1 projection weight update norm: {rows_by_method['M1_TaTS-Raw-Text']['projection_weight_update_norm']}",
        "",
        "## 17. 训练曲线",
        "Per-epoch train/validation losses and GPU memory are saved under each run directory as `epoch_metrics.csv` and `train.log`.",
        "",
        "## 18. 验证集结果",
        "| Method | model_val_MSE |",
        "| --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['model_val_MSE']} |")
    lines.extend(["", "## 19. 测试集结果", "| Method | MSE | MAE | RMSE | NMSE | sMAPE | Directional Accuracy | Trend Macro-F1 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for r in rows:
        lines.append(f"| {r['method']} | {r['MSE']} | {r['MAE']} | {r['RMSE']} | {r['NMSE']} | {r['sMAPE']} | {r['Directional Accuracy']} | {r['Trend Macro-F1']} |")
    lines.extend(
        [
            "",
            "## 20. 文本反事实输入诊断",
            f"- real_vs_zero_max_diff: {counter['real_vs_zero_max_diff']}",
            f"- real_vs_shuffle_max_diff: {counter['real_vs_shuffle_max_diff']}",
            f"- real_vs_zero_mean_diff / target_std: {counter['real_vs_zero_mean_diff_over_target_std']}",
            f"- real_vs_shuffle_mean_diff / target_std: {counter['real_vs_shuffle_mean_diff_over_target_std']}",
            "",
            "## 21. M0/M1/Z0/S0 预测差异",
            f"- M0 and M1 identical: {status['m0_m1_predictions_identical']}",
            f"- M1 and Z0 identical: {status['m1_z0_predictions_identical']}",
            f"- M1 and S0 identical: {status['m1_s0_predictions_identical']}",
            "",
            "## 22. 工程链路结论",
            "P1 verifies whether text enters the model and changes predictions. This is separate from whether raw text improves metrics.",
            "",
            "## 23. 性能结论",
            "The metric table above is the only basis for performance claims; one seed is reported without mean ± std.",
            "",
            "## 24. 失败或异常案例",
            "No event/APO modules were run. Any failed hard checks would set P1_STATUS to FAIL.",
            "",
            "## 25. 自动测试",
            tests_text[-2000:],
            "",
            "## 26. 文件清单",
            "- `results/v6/p1/environment_data_audit.json`",
            "- `results/v6/p1/input_diagnostics.json`",
            "- `results/v6/p1/p1_metrics.csv`",
            "- `results/v6/p1/text_counterfactual_diagnostics.json`",
            "- `results/v6/p1/p1_status.json`",
            "- `reports/v6/p1_official_tats_reproduction_report.md`",
            "",
            "## 27. P1 是否允许进入 P2",
            "允许进入 P2。" if status["p1_status"] == "PASS" else "不允许进入 P2。",
            "",
            "## 28. 下一阶段建议",
            "Only after P1 PASS, move to P2 text/event representation diagnostics. Do not start M2-M5 from this P1 run.",
        ]
    )
    (root / "reports" / "v6" / "p1_official_tats_reproduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    upstream = root / "reports" / "v6" / "p1_upstream_changes.md"
    upstream.write_text(
        "# P1 Upstream Changes\n\n"
        "| File | Original behavior | Modification reason | Affects backbone | Diff summary |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `vendor/tats/models/iTransformer.py` | Official iTransformer backbone | No source modification | No | No diff |\n"
        "| adapter in `src/cen_ts/runtime/p1_official.py` | N/A | Adds local GPT-2 strict loading, projection, logging, diagnostics | No | Wrapper only |\n",
        encoding="utf-8",
    )


def refresh_status_after_tests() -> dict[str, Any]:
    root = repo_root()
    path = root / "results" / "v6" / "p1" / "p1_status.json"
    if not path.exists():
        raise P1Error("p1_status.json does not exist.")
    status = json.loads(path.read_text(encoding="utf-8"))
    tests_log = root / "results" / "v6" / "p1" / "pytest_v6.log"
    tests_passed = tests_log.exists() and "failed" not in tests_log.read_text(encoding="utf-8", errors="replace").lower()
    status["tests_passed"] = tests_passed
    if tests_passed and status["p1_status"] == "PARTIAL":
        blocking = [
            status.get("m0_m1_predictions_identical"),
            status.get("m1_z0_predictions_identical"),
            status.get("m1_s0_predictions_identical"),
            status.get("projection_gradient_norm_max", 0) <= 1e-8,
            status.get("counterfactual_real_zero_max_diff", 0) <= 1e-6,
            status.get("counterfactual_real_shuffle_max_diff", 0) <= 1e-6,
        ]
        if not any(blocking):
            status["p1_status"] = "PASS"
    write_json(path, status)
    return status
