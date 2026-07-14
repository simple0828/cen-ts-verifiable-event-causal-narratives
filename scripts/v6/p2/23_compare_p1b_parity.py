from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v6" / "p2"
P1B = ROOT / "results" / "v6" / "p1b_official_tats"
RAW_RUN = OUT / "runs" / "p2_raw_official_reproduction_pw0.5_s2025"
GPT2_PATH = Path("D:/models/gpt2")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def purge_modules() -> None:
    prefixes = ("data_provider", "exp", "layers", "models", "utils")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            sys.modules.pop(name, None)


def install_patoolib_stub() -> None:
    if "patoolib" not in sys.modules:
        stub = types.ModuleType("patoolib")

        def extract_archive(*_args, **_kwargs):
            raise RuntimeError("patoolib is not needed for Environment parity.")

        stub.extract_archive = extract_archive
        sys.modules["patoolib"] = stub


def patch_gpt2() -> None:
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

    original_config = GPT2Config.from_pretrained
    original_model = GPT2Model.from_pretrained
    original_tokenizer = GPT2Tokenizer.from_pretrained

    def resolve(name: Any) -> Any:
        return str(GPT2_PATH) if str(name) == "openai-community/gpt2" else name

    def patched_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs["local_files_only"] = True
        return original_config(resolve(pretrained_model_name_or_path), *args, **kwargs)

    def patched_model(cls, pretrained_model_name_or_path, *args, **kwargs):
        if kwargs.get("local_files_only") is False:
            raise RuntimeError("network fallback is forbidden")
        kwargs["local_files_only"] = True
        return original_model(resolve(pretrained_model_name_or_path), *args, **kwargs)

    def patched_tokenizer(cls, pretrained_model_name_or_path, *args, **kwargs):
        if kwargs.get("local_files_only") is False:
            raise RuntimeError("network fallback is forbidden")
        kwargs["local_files_only"] = True
        return original_tokenizer(resolve(pretrained_model_name_or_path), *args, **kwargs)

    GPT2Config.from_pretrained = classmethod(patched_config)
    GPT2Model.from_pretrained = classmethod(patched_model)
    GPT2Tokenizer.from_pretrained = classmethod(patched_tokenizer)


def base_args(root_path: Path, data_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id="p2_parity",
        model="iTransformer",
        data="custom",
        root_path=str(root_path.resolve()),
        data_path=data_path,
        features="S",
        target="OT",
        freq="d",
        checkpoints=str((OUT / "parity_tmp" / "checkpoints").resolve()),
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
        save_name=str((OUT / "parity_tmp" / "result.txt").resolve()),
        use_fullmodel=0,
        use_closedllm=0,
        huggingface_token="NA",
        text_dim=12,
    )


def run_first_batch(kind: str) -> dict[str, Any]:
    purge_modules()
    install_patoolib_stub()
    if kind == "official":
        sys.path.insert(0, str((ROOT / "third_party" / "TaTS").resolve()))
        patch_gpt2()
        args = base_args(ROOT / "third_party" / "TaTS" / "data", "Environment.csv")
    else:
        sys.path.insert(0, str((ROOT / "tats_cen").resolve()))
        args = base_args(ROOT / "third_party" / "TaTS" / "data", "Environment.csv")
        args.text_mode = "raw"
        args.text_column = "fact"
        args.llm_path = str(GPT2_PATH)
        args.strict_local_llm = True
        args.fail_on_missing_text = True
        args.record_input_hashes = True
        args.run_dir = str((OUT / "parity_tmp" / "tats_cen").resolve())
        args.gpt2_manifest_path = str(Path(args.run_dir) / "gpt2_manifest.json")
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    exp = Exp_Long_Term_Forecast(args)
    model_hash = state_hash(exp.model)
    mlp_shapes = {key: list(value.shape) for key, value in exp.mlp.state_dict().items()}
    train_data, train_loader = exp._get_data("train")
    torch.manual_seed(args.seed)
    batch_x, batch_y, batch_x_mark, batch_y_mark, index = next(iter(train_loader))
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    prior_y = torch.from_numpy(train_data.get_prior_y(index)).float().to(exp.device)
    batch_text_embeddings = train_data.get_text_embeddings(index)
    prompt_emb = exp.mlp(batch_text_embeddings)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    combined_x = torch.cat([batch_x, prompt_emb], dim=-1)
    text_dec_inp = torch.zeros((args.batch_size, args.pred_len, exp.text_embedding_dim)).to(exp.device)
    text_dec_inp = torch.cat([prompt_emb[:, : args.label_len, :], text_dec_inp], dim=1).float().to(exp.device)
    dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1)
    outputs = exp.model(combined_x, batch_x_mark, dec_inp, batch_y_mark)
    outputs = outputs[:, -args.pred_len :, 0:].contiguous()
    outputs = outputs[:, :, 0].unsqueeze(-1)
    outputs = (1 - exp.prompt_weight) * outputs + exp.prompt_weight * prior_y
    target = batch_y[:, -args.pred_len :, 0:].to(exp.device)
    loss = torch.nn.MSELoss()(outputs, target)
    loss.backward()
    grad_tensors = [p.grad.detach().flatten().cpu() for p in list(exp.model.parameters()) + list(exp.mlp.parameters()) if p.grad is not None]
    grad_flat = torch.cat(grad_tensors)
    row_ids = []
    for item in index.detach().cpu().numpy():
        s_begin = int(item % train_data.tot_len)
        row_ids.extend(range(s_begin, s_begin + args.seq_len))
    token_ids = train_data.input_ids[row_ids].view(args.batch_size, args.seq_len, -1)
    return {
        "model_state_hash": model_hash,
        "projection_shapes": mlp_shapes,
        "train_window_count": len(train_data),
        "first_batch_numeric_hash": tensor_hash(batch_x),
        "first_batch_token_ids_hash": tensor_hash(token_ids),
        "first_batch_pooled_embedding_hash": tensor_hash(batch_text_embeddings),
        "first_batch_projected_text_hash": tensor_hash(prompt_emb),
        "first_batch_combined_input_hash": tensor_hash(combined_x),
        "first_forward_output": outputs.detach().cpu().numpy(),
        "first_loss": float(loss.detach().cpu().item()),
        "first_gradient": grad_flat.numpy(),
    }


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> None:
    official = run_first_batch("official")
    tats_cen = run_first_batch("tats_cen")
    p1b_metrics = json.loads((P1B / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    raw_metrics = json.loads((RAW_RUN / "test_metrics.json").read_text(encoding="utf-8"))
    p1b_pred = np.load(P1B / "predictions.npy")
    raw_pred = np.load(RAW_RUN / "predictions.npy")
    corr = float(np.corrcoef(p1b_pred.reshape(-1), raw_pred.reshape(-1))[0, 1])
    p1b_scaled_mse = float(p1b_metrics["native_scaled"]["MSE"])
    raw_scaled_mse = float(raw_metrics["native_scaled"]["MSE"])
    metric_rel = abs(raw_scaled_mse - p1b_scaled_mse) / p1b_scaled_mse
    forward_diff = max_abs(official["first_forward_output"], tats_cen["first_forward_output"])
    loss_diff = abs(official["first_loss"] - tats_cen["first_loss"])
    grad_diff = max_abs(official["first_gradient"], tats_cen["first_gradient"])
    result = {
        "upstream_itransformer_sha256": sha256_file(ROOT / "third_party" / "TaTS" / "models" / "iTransformer.py"),
        "tats_cen_itransformer_sha256": sha256_file(ROOT / "tats_cen" / "models" / "iTransformer.py"),
        "itransformer_unchanged": sha256_file(ROOT / "third_party" / "TaTS" / "models" / "iTransformer.py") == sha256_file(ROOT / "tats_cen" / "models" / "iTransformer.py"),
        "projection_shapes_equal": official["projection_shapes"] == tats_cen["projection_shapes"],
        "train_window_count_equal": official["train_window_count"] == tats_cen["train_window_count"],
        "first_batch_numeric_parity": official["first_batch_numeric_hash"] == tats_cen["first_batch_numeric_hash"],
        "first_batch_token_parity": official["first_batch_token_ids_hash"] == tats_cen["first_batch_token_ids_hash"],
        "pooled_embedding_hash_equal": official["first_batch_pooled_embedding_hash"] == tats_cen["first_batch_pooled_embedding_hash"],
        "projected_text_hash_equal": official["first_batch_projected_text_hash"] == tats_cen["first_batch_projected_text_hash"],
        "combined_input_hash_equal": official["first_batch_combined_input_hash"] == tats_cen["first_batch_combined_input_hash"],
        "initial_model_state_equal": official["model_state_hash"] == tats_cen["model_state_hash"],
        "forward_max_abs_diff": forward_diff,
        "loss_abs_diff": loss_diff,
        "gradient_max_abs_diff": grad_diff,
        "p1b_scaled_mse": p1b_scaled_mse,
        "tats_cen_scaled_mse": raw_scaled_mse,
        "metric_relative_difference": metric_rel,
        "prediction_correlation": corr,
        "p2_parity": "PASS"
        if (
            official["first_batch_numeric_hash"] == tats_cen["first_batch_numeric_hash"]
            and official["first_batch_token_ids_hash"] == tats_cen["first_batch_token_ids_hash"]
            and forward_diff < 1e-6
            and loss_diff < 1e-6
            and grad_diff < 1e-6
            and metric_rel < 0.01
            and corr > 0.99
        )
        else "FAIL",
    }
    write_json(OUT / "p1b_parity.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
