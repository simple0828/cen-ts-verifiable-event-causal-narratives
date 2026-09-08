from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from cen_ts.runtime.preflight import DEFAULT_GPT2_PATH, PreflightError, encode_texts_masked_average, load_pretrained_gpt2_strict, repo_root


def ensure_tats_import_path() -> None:
    root = repo_root()
    tats = root / "vendor" / "tats"
    if str(tats) not in sys.path:
        sys.path.insert(0, str(tats))


def make_itransformer(seq_len: int, pred_len: int, channels: int, *, d_model: int = 32) -> nn.Module:
    ensure_tats_import_path()
    module = importlib.import_module("models.iTransformer")
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=max(1, seq_len // 2),
        pred_len=pred_len,
        enc_in=channels,
        dec_in=channels,
        c_out=1,
        d_model=d_model,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        factor=1,
        dropout=0.0,
        embed="timeF",
        freq="w",
        activation="gelu",
        output_attention=False,
        num_class=2,
    )
    return module.Model(cfg)


class TextProjection(nn.Module):
    def __init__(self, llm_dim: int = 768, text_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_dim, max(4, llm_dim // 8)),
            nn.ReLU(),
            nn.Linear(max(4, llm_dim // 8), text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_texts(seq_len: int) -> tuple[list[str], list[str]]:
    real = []
    shuffled = []
    a = "A major oil producer announced a production cut."
    b = "A severe weather event disrupted regional energy supply."
    for i in range(seq_len):
        real.append(f"{a} Window step {i}.")
        shuffled.append(f"{b} Window step {i}.")
    return real, shuffled


def _forward(backbone: nn.Module, projector: TextProjection, x_num: torch.Tensor, raw_text: torch.Tensor, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    projected = projector(raw_text)
    x = torch.cat([x_num, projected], dim=-1)
    dec_inp = torch.zeros((x.shape[0], pred_len, x.shape[-1]), dtype=x.dtype, device=x.device)
    output = backbone(x, None, dec_inp, None)[:, -pred_len:, :1]
    return output, projected


def run_itransformer_smoke(
    *,
    model_path: str | Path = DEFAULT_GPT2_PATH,
    device: str | None = None,
    seq_len: int = 8,
    pred_len: int = 2,
    text_dim: int = 12,
    seed: int = 2025,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device is None:
        if not torch.cuda.is_available():
            raise PreflightError("CUDA is required for the formal TaTS iTransformer smoke test.")
        device = "cuda:0"
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise PreflightError(f"Expected a CUDA device for smoke test, got {device}")

    tokenizer, gpt2, gpt2_meta = load_pretrained_gpt2_strict(model_path, device=device)
    real_texts, other_texts = _make_texts(seq_len)
    all_texts = real_texts + other_texts
    raw = encode_texts_masked_average(tokenizer, gpt2, all_texts, max_length=256, device=device)
    raw = raw.reshape(2, seq_len, -1).detach()
    raw_real = raw.clone()
    raw_shuffle = raw.flip(0).clone()
    raw_zero = torch.zeros_like(raw_real)

    x_num = torch.linspace(-1.0, 1.0, steps=2 * seq_len, device=torch_device, dtype=raw.dtype).reshape(2, seq_len, 1)
    target = torch.zeros((2, pred_len, 1), device=torch_device, dtype=raw.dtype)

    backbone = make_itransformer(seq_len, pred_len, 1 + text_dim).to(torch_device)
    projector = TextProjection(llm_dim=768, text_dim=text_dim).to(torch_device)

    backbone.train()
    projector.train()
    output, projected = _forward(backbone, projector, x_num, raw_real, pred_len)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    grad_norm = 0.0
    for param in projector.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().item())

    backbone.eval()
    projector.eval()
    with torch.no_grad():
        output_real, projected_real = _forward(backbone, projector, x_num, raw_real, pred_len)
        output_zero, projected_zero = _forward(backbone, projector, x_num, raw_zero, pred_len)
        output_shuffle, projected_shuffle = _forward(backbone, projector, x_num, raw_shuffle, pred_len)

    real_zero_diff = float((output_real - output_zero).abs().max().item())
    real_shuffle_diff = float((output_real - output_shuffle).abs().max().item())
    raw_var = float(raw_real.var(unbiased=False).item())
    projected_var = float(projected_real.var(unbiased=False).item())

    tensors_on_device = all(
        tensor.device.type == "cuda"
        for tensor in (raw_real, raw_zero, raw_shuffle, x_num, output_real, output_zero, output_shuffle, projected_real, projected_zero, projected_shuffle)
    )
    result: dict[str, Any] = {
        "success": bool(grad_norm > 1e-8 and real_zero_diff > 1e-6 and real_shuffle_diff > 1e-6 and tensors_on_device),
        "backbone": "iTransformer",
        "gpt2_model_class": gpt2_meta["model_class"],
        "no_random_gpt2_fallback": not bool(gpt2_meta["random_init"]),
        "raw_text_embedding_variance": raw_var,
        "projected_text_variance": projected_var,
        "projection_gradient_norm": grad_norm,
        "prediction_real_vs_zero_max_diff": real_zero_diff,
        "prediction_real_vs_shuffle_max_diff": real_shuffle_diff,
        "output_shape": list(output_real.shape),
        "device": str(torch_device),
        "all_tensors_on_expected_gpu": tensors_on_device,
    }
    if not result["success"]:
        raise PreflightError(f"TaTS iTransformer smoke thresholds failed: {result}")
    return result
