from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TATS = ROOT / "third_party" / "TaTS"
OUT = ROOT / "results" / "v6" / "p1b_official_tats"
REPORT = ROOT / "reports" / "v6" / "p1b_exact_official_tats_report.md"
GPT2_PATH = Path("D:/models/gpt2")
SAFE_TATS = "C:/Users/Administrator/Desktop/TS/cen-ts-verifiable-event-causal-narratives/third_party/TaTS"


def _cmd(cmd: list[str], cwd: Path = ROOT, check: bool = False) -> str:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout + completed.stderr


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def strict_patch_gpt2() -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

    original_config = GPT2Config.from_pretrained
    original_model = GPT2Model.from_pretrained
    original_tokenizer = GPT2Tokenizer.from_pretrained

    def resolve_path(name: Any) -> Any:
        return str(GPT2_PATH) if str(name) == "openai-community/gpt2" else name

    def patched_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs["local_files_only"] = True
        return original_config(resolve_path(pretrained_model_name_or_path), *args, **kwargs)

    def patched_model(cls, pretrained_model_name_or_path, *args, **kwargs):
        if kwargs.get("local_files_only") is False:
            raise RuntimeError("P1b forbids GPT-2 network fallback.")
        kwargs["local_files_only"] = True
        return original_model(resolve_path(pretrained_model_name_or_path), *args, **kwargs)

    def patched_tokenizer(cls, pretrained_model_name_or_path, *args, **kwargs):
        if kwargs.get("local_files_only") is False:
            raise RuntimeError("P1b forbids GPT-2 tokenizer network fallback.")
        kwargs["local_files_only"] = True
        return original_tokenizer(resolve_path(pretrained_model_name_or_path), *args, **kwargs)

    GPT2Config.from_pretrained = classmethod(patched_config)
    GPT2Model.from_pretrained = classmethod(patched_model)
    GPT2Tokenizer.from_pretrained = classmethod(patched_tokenizer)
    return {"patched_openai_community_gpt2_to": str(GPT2_PATH), "local_files_only": True, "network_fallback_forbidden": True}


def official_args() -> SimpleNamespace:
    model_id = "data_2025_24_48_fullLLM_p1b"
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=model_id,
        model="iTransformer",
        data="custom",
        root_path=str((TATS / "data").resolve()),
        data_path="Environment.csv",
        features="S",
        target="OT",
        freq="d",
        checkpoints=str((OUT / "checkpoints").resolve()),
        seq_len=24,
        label_len=12,
        pred_len=48,
        seasonal_patterns="Monthly",
        inverse=False,
        text_emb=12,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=13,
        dec_in=13,
        c_out=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,
        num_workers=0,
        itr=1,
        train_epochs=5,
        batch_size=32,
        patience=5,
        learning_rate=0.0001,
        des="Exp",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        llm_model="GPT2",
        llm_dim=768,
        llm_layers=6,
        text_path="None",
        type_tag="#F#",
        text_len=3,
        learning_rate2=1e-2,
        learning_rate3=1e-3,
        prompt_weight=0.5,
        prior_weight=0.5,
        pool_type="avg",
        date_name="end_date",
        addHisRate=0.5,
        init_method="normal",
        learning_rate_weight=0.0001,
        seed=2025,
        save_name=str((OUT / "result_environment_itransformer_gpt2.txt").resolve()),
        use_fullmodel=0,
        use_closedllm=0,
        huggingface_token="NA",
        text_dim=12,
    )


def setting_for(args: SimpleNamespace) -> str:
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0,
    )


def preflight_checks(args: SimpleNamespace) -> dict[str, Any]:
    from transformers import GPT2Model, GPT2Tokenizer

    csv_path = Path(args.root_path) / args.data_path
    df = pd.read_csv(csv_path, nrows=8)
    model = GPT2Model.from_pretrained(str(GPT2_PATH), local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_PATH), local_files_only=True)
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(["one short text", "another longer Environment fact"], return_tensors="pt", padding=True, truncation=True, max_length=256)
    emb = model.get_input_embeddings()(encoded["input_ids"])
    mask = encoded["attention_mask"].unsqueeze(-1).expand_as(emb)
    masked = emb * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return {
        "gpt2_model_class": type(model).__name__,
        "gpt2_model_path": str(GPT2_PATH),
        "random_init": False,
        "cuda": bool(torch.cuda.is_available()),
        "model": args.model,
        "environment_csv_exists": csv_path.exists(),
        "fact_column_exists": "fact" in df.columns,
        "uses_get_input_embeddings_precheck": list(emb.shape),
        "pool_type": args.pool_type,
        "padding_mask_shape": list(mask.shape),
        "padding_mask_valid_count_min": float(mask.sum(dim=1).min().item()),
        "pooled_shape": list(pooled.shape),
    }


def parse_epoch_metrics(log_text: str) -> list[dict[str, float]]:
    rows = []
    pattern = re.compile(
        r"Epoch: (?P<epoch>\d+), Steps: (?P<steps>\d+) \| Train Loss: (?P<train>[0-9.]+) Vali Loss: (?P<vali>[0-9.]+) Test Loss \(MSE\): (?P<test>[0-9.]+) Test MAE: (?P<mae>[0-9.]+)"
    )
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


def split_counts(args: SimpleNamespace) -> dict[str, Any]:
    n = len(pd.read_csv(Path(args.root_path) / args.data_path))
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test
    border1s = [0, num_train - args.seq_len, n - num_test - args.seq_len]
    border2s = [num_train, num_train + num_val, n]
    lengths = {
        "train": border2s[0] - border1s[0] - args.seq_len - args.pred_len + 1,
        "val": border2s[1] - border1s[1] - args.seq_len - args.pred_len + 1,
        "test": border2s[2] - border1s[2] - args.seq_len - args.pred_len + 1,
    }
    used = {k: (v // args.batch_size) * args.batch_size for k, v in lengths.items()}
    return {
        "n": n,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "border1s": border1s,
        "border2s": border2s,
        "dataset_window_counts": lengths,
        "drop_last_used_window_counts": used,
    }


def compare_with_p1_adapter(official_pred: np.ndarray, official_true: np.ndarray) -> dict[str, Any]:
    status_path = ROOT / "results" / "v6" / "p1" / "p1_status.json"
    if not status_path.exists():
        return {"available": False, "reason": "P1 status file not found"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    m1 = next(row for row in status["methods"] if row["method"] == "M1_TaTS-Raw-Text")
    p1_npz = np.load(Path(m1["run_dir"]) / "predictions.npz")
    p1_tgt = np.load(Path(m1["run_dir"]) / "targets.npz")
    p1_pred_scaled = p1_npz["predictions_scaled"]
    p1_true_scaled = p1_tgt["targets_scaled"]
    offset = 24
    count = min(len(official_pred) - offset, len(p1_pred_scaled))
    off_aligned = official_pred[offset : offset + count]
    p1_aligned = p1_pred_scaled[:count]
    off_true_aligned = official_true[offset : offset + count]
    p1_true_aligned = p1_true_scaled[:count]
    corr = float(np.corrcoef(off_aligned.reshape(-1), p1_aligned.reshape(-1))[0, 1])
    mad = float(np.mean(np.abs(off_aligned - p1_aligned)))
    target_mad = float(np.mean(np.abs(off_true_aligned - p1_true_aligned)))
    return {
        "available": True,
        "p1_adapter_run_id": m1["run_id"],
        "p1_adapter_test_mse_unscaled": m1["MSE"],
        "official_to_p1_alignment": {
            "official_offset": offset,
            "aligned_windows": int(count),
            "reason": "Official TaTS test split starts seq_len points before the test border; P1 adapter starts inside the test split.",
        },
        "prediction_correlation_scaled_aligned": corr,
        "prediction_mean_absolute_difference_scaled_aligned": mad,
        "target_mean_absolute_difference_scaled_aligned": target_mad,
        "adapter_parity": "FAIL",
        "adapter_parity_reasons": [
            "P1b official data_loader uses llm_model.get_input_embeddings()(input_ids); P1 adapter pooled GPT-2 model hidden states.",
            "P1b official projection is Linear-ReLU-Linear-ReLU-Dropout; P1 adapter uses a different projection block.",
            "P1b official train/val/test windows follow TaTS overlapping border1s and DataLoader drop_last=True; P1 adapter uses non-overlapping split windows and drop_last=False.",
            "P1b official forecasting loop mixes outputs with prior_history_avg using prior_weight=0.5; P1 adapter did not use this official prior mix.",
            "P1b official decoder input includes label_len target history and text decoder channels; P1 adapter used a simplified zero decoder input.",
        ],
    }


def write_report(
    args: SimpleNamespace,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    epoch_metrics: list[dict[str, Any]],
) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    p1_m1 = comparison.get("p1_adapter_test_mse_unscaled", "unavailable")
    lines = [
        "# P1b Exact Official TaTS Reproduction Report",
        "",
        "## 1. Scope",
        "This run directly uses `third_party/TaTS` official `data_provider`, `exp`, `models`, and training/test loops. It does not run event extraction, causal graph construction, verifier, APO, or paid LLM APIs.",
        "",
        "## 2. Upstream State",
        f"- upstream_commit: `{manifest['upstream_commit']}`",
        f"- upstream_status: `{manifest['upstream_status_summary']}`",
        f"- upstream_diff_patch: `results/v6/p1b_official_tats/upstream_diff.patch`",
        "",
        "## 3. Command",
        "`D:/Miniconda/envs/tats/python.exe scripts/v6/14_run_p1b_exact_official_tats.py`",
        "",
        "## 4. Preflight",
        f"- GPT-2 path: `{manifest['preflight']['gpt2_model_path']}`",
        f"- GPT-2 class: `{manifest['preflight']['gpt2_model_class']}`",
        f"- random_init: `{manifest['preflight']['random_init']}`",
        f"- CUDA: `{manifest['preflight']['cuda']}`",
        f"- model: `{manifest['preflight']['model']}`",
        f"- fact column exists: `{manifest['preflight']['fact_column_exists']}`",
        f"- pool_type: `{manifest['preflight']['pool_type']}`",
        "",
        "## 5. Official Resolved Args",
        f"- dataset: Environment",
        f"- model: {args.model}",
        f"- seq_len/label_len/pred_len: {args.seq_len}/{args.label_len}/{args.pred_len}",
        f"- text_emb: {args.text_emb}",
        f"- seed: {args.seed}",
        f"- train_epochs/patience: {args.train_epochs}/{args.patience}",
        f"- prior_weight: {args.prior_weight}",
        f"- use_amp: {args.use_amp}",
        f"- llm_layers: {args.llm_layers} (official script setting; GPT-2 input embedding table is still loaded from the local pretrained checkpoint)",
        "",
        "## 6. P1 Adapter M1 vs P1b Official M1",
        "| Item | P1 adapter M1 | P1b official M1 | Same? |",
        "| --- | --- | --- | --- |",
        "| Text encoding | GPT-2 forward hidden states, mask-average pooled | `llm_model.get_input_embeddings()(input_ids)`, mask-average pooled | No |",
        "| GPT-2 path | `D:/models/gpt2` | `D:/models/gpt2` via strict monkeypatch wrapper | Yes |",
        "| Pooling | avg | avg | Yes |",
        "| Projection MLP | custom P1 adapter projection | official `Linear-ReLU-Linear-ReLU-Dropout(0.3)` | No |",
        "| iTransformer params | d_model=512, n_heads=8, e_layers=2, d_ff=2048 | d_model=512, n_heads=8, e_layers=2, d_ff=2048 | Yes |",
        "| Data split | non-overlap P1 split windows | official TaTS 70/10/20 border windows with val/test history overlap | No |",
        "| Train/test windows | train 10602, val 1455, test 2978 | train 10602/used 10592, val 1479/used 1472, test 3002/used 2976 | No |",
        "| Training loop | P1 custom adapter loop | official `Exp_Long_Term_Forecast.train/test` | No |",
        "",
        "## 7. P1b Training Metrics",
        "| epoch | train_loss | vali_loss | test_mse | test_mae |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in epoch_metrics:
        lines.append(f"| {row['epoch']} | {row['train_loss']} | {row['vali_loss']} | {row['test_mse']} | {row['test_mae']} |")
    lines.extend(
        [
            "",
            "## 8. Test Metrics",
            f"- P1 adapter M1 unscaled MSE: `{p1_m1}`",
            f"- P1b official native scaled MSE: `{metrics['native_scaled']['MSE']}`",
            f"- P1b official native scaled MAE: `{metrics['native_scaled']['MAE']}`",
            f"- P1b official unscaled MSE: `{metrics['unscaled']['MSE']}`",
            f"- P1b official unscaled MAE: `{metrics['unscaled']['MAE']}`",
            "",
            "## 9. Prediction Comparison",
            f"- aligned prediction correlation: `{comparison.get('prediction_correlation_scaled_aligned')}`",
            f"- aligned prediction mean absolute difference: `{comparison.get('prediction_mean_absolute_difference_scaled_aligned')}`",
            f"- aligned target mean absolute difference: `{comparison.get('target_mean_absolute_difference_scaled_aligned')}`",
            "",
            "## 10. ADAPTER_PARITY",
            f"ADAPTER_PARITY={comparison.get('adapter_parity')}",
            "",
            "Reasons:",
        ]
    )
    for reason in comparison.get("adapter_parity_reasons", []):
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "Conclusion: the current P1 adapter is useful as an engineering-chain check, but it is not an exact implementation of the official TaTS M1 baseline.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "command.txt").write_text(
        "PowerShell:\n"
        "D:/Miniconda/envs/tats/python.exe scripts/v6/14_run_p1b_exact_official_tats.py\n\n"
        "Official Bash source converted from third_party/TaTS/scripts/main_forecast.sh; GPT-2 loading is redirected strictly to D:/models/gpt2 by wrapper monkeypatch.\n",
        encoding="utf-8",
    )
    upstream_status = _cmd(["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(TATS), "status"])
    upstream_commit = _cmd(["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(TATS), "rev-parse", "HEAD"], check=True).strip()
    upstream_diff = _cmd(["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(TATS), "diff"])
    (OUT / "upstream_commit.txt").write_text(upstream_commit + "\n", encoding="utf-8")
    (OUT / "upstream_diff.patch").write_text(upstream_diff, encoding="utf-8")
    patch_meta = strict_patch_gpt2()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(TATS) not in sys.path:
        sys.path.insert(0, str(TATS))
    if "patoolib" not in sys.modules:
        patoolib_stub = types.ModuleType("patoolib")

        def _unused_extract_archive(*_args, **_kwargs):
            raise RuntimeError("patoolib stub is only present for non-M4 P1b imports.")

        patoolib_stub.extract_archive = _unused_extract_archive
        sys.modules["patoolib"] = patoolib_stub
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.metrics import metric

    args = official_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("P1b requires CUDA=True.")
    preflight = preflight_checks(args)
    counts = split_counts(args)
    setting = setting_for(args)
    write_json(OUT / "resolved_args.json", vars(args) | {"setting": setting})

    cwd = Path.cwd()
    os.chdir(OUT)
    get_input_embedding_calls = {"count": 0}
    started = time.time()
    try:
        with (OUT / "train.log").open("w", encoding="utf-8", newline="\n") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print("P1b exact official TaTS run")
            print("upstream_commit:", upstream_commit)
            exp = Exp_Long_Term_Forecast(args)
            original_get = exp.llm_model.get_input_embeddings

            def counted_get_input_embeddings(self):
                get_input_embedding_calls["count"] += 1
                return original_get()

            exp.llm_model.get_input_embeddings = MethodType(counted_get_input_embeddings, exp.llm_model)
            print(">>>>>>>start training official loop>>>>>>>>")
            exp.train(setting)
            print(">>>>>>>testing official loop<<<<<<<<")
            exp.test(setting, test=0)
    finally:
        os.chdir(cwd)

    official_result_dir = OUT / "results" / setting
    pred = np.load(official_result_dir / "pred.npy")
    true = np.load(official_result_dir / "true.npy")
    native_metrics_arr = np.load(official_result_dir / "metrics.npy")
    shutil.copy2(official_result_dir / "pred.npy", OUT / "predictions.npy")
    shutil.copy2(official_result_dir / "true.npy", OUT / "targets.npy")
    shutil.copy2(official_result_dir / "metrics.npy", OUT / "official_metrics.npy")

    train_df = pd.read_csv(Path(args.root_path) / args.data_path).iloc[: int(counts["num_train"])]
    mean = float(train_df[args.target].mean())
    std = float(train_df[args.target].std(ddof=0))
    pred_unscaled = pred * std + mean
    true_unscaled = true * std + mean
    mae_u, mse_u, rmse_u, mape_u, mspe_u = metric(pred_unscaled, true_unscaled)
    metrics = {
        "native_scaled": {
            "MAE": float(native_metrics_arr[0]),
            "MSE": float(native_metrics_arr[1]),
            "RMSE": float(native_metrics_arr[2]),
            "MAPE": float(native_metrics_arr[3]),
            "MSPE": float(native_metrics_arr[4]),
        },
        "unscaled": {
            "MAE": float(mae_u),
            "MSE": float(mse_u),
            "RMSE": float(rmse_u),
            "MAPE": float(mape_u),
            "MSPE": float(mspe_u),
        },
    }
    comparison = compare_with_p1_adapter(pred, true)
    log_text = (OUT / "train.log").read_text(encoding="utf-8", errors="replace")
    epoch_metrics = parse_epoch_metrics(log_text)
    manifest = {
        "stage": "P1b",
        "status": "PASS",
        "upstream_commit": upstream_commit,
        "upstream_status": upstream_status,
        "upstream_status_summary": "clean" if "working tree clean" in upstream_status else "not_clean",
        "upstream_diff_empty": upstream_diff.strip() == "",
        "wrapper_patch": patch_meta,
        "preflight": preflight,
        "split_counts": counts,
        "setting": setting,
        "runtime_sec": time.time() - started,
        "get_input_embeddings_call_count": get_input_embedding_calls["count"],
        "official_result_dir": str(official_result_dir),
        "prediction_shape": list(pred.shape),
        "target_shape": list(true.shape),
        "adapter_comparison": comparison,
        "outputs": {
            "command": str(OUT / "command.txt"),
            "resolved_args": str(OUT / "resolved_args.json"),
            "upstream_commit": str(OUT / "upstream_commit.txt"),
            "upstream_diff": str(OUT / "upstream_diff.patch"),
            "train_log": str(OUT / "train.log"),
            "metrics": str(OUT / "metrics.json"),
            "predictions": str(OUT / "predictions.npy"),
            "manifest": str(OUT / "manifest.json"),
        },
    }
    write_json(OUT / "metrics.json", {"metrics": metrics, "epoch_metrics": epoch_metrics, "comparison": comparison})
    write_json(OUT / "manifest.json", manifest)
    write_report(args, manifest, metrics, comparison, epoch_metrics)


if __name__ == "__main__":
    main()
