from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def _ensure_tats_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    tats = root / "third_party" / "TaTS"
    if str(tats) not in sys.path:
        sys.path.insert(0, str(tats))
    return tats


def load_official_backbone(name: str, cfg: SimpleNamespace) -> nn.Module:
    _ensure_tats_path()
    if name not in {"PatchTST", "iTransformer"}:
        raise ValueError(f"Unsupported v5 official TaTS backbone: {name}")
    module = importlib.import_module(f"models.{name}")
    return module.Model(cfg)


class RandomInitGPT2InputEmbedding:
    """Official-TaTS-compatible GPT2 tokenizer plus random input embedding.

    The official TaTS code uses tokenizer -> input embedding -> pooling by default.
    In this run the GPT2 tokenizer/config are cached, but pretrained weights timed
    out. This class is deliberately marked as random-init and is not reported as a
    completed pretrained TaTS baseline.
    """

    def __init__(self, cache_dir: str = "artifacts/v5/hf_cache", seed: int = 2026, max_length: int = 256):
        from transformers import GPT2Tokenizer

        self.cache_dir = cache_dir
        self.max_length = int(max_length)
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", cache_dir=cache_dir, local_files_only=True)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer)))
        n_embd = 768
        gen = torch.Generator().manual_seed(seed)
        self.embedding = nn.Embedding(vocab_size, n_embd)
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.02, generator=gen)
        self.embedding.weight.requires_grad_(False)
        self.dim = n_embd
        self.model_name = "openai-community/gpt2"
        self.mode = "random_init_input_embedding_only_pretrained_weights_timeout"

    def encode(self, texts: list[str], batch_size: int = 64, pool_type: str = "avg") -> np.ndarray:
        outputs = []
        self.embedding.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [str(t) if str(t).strip() else "No information available" for t in texts[start : start + batch_size]]
                toks = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                emb = self.embedding(toks["input_ids"])
                mask = toks["attention_mask"].unsqueeze(-1).expand_as(emb)
                if pool_type == "max":
                    pooled = emb.masked_fill(mask == 0, float("-inf")).max(dim=1).values
                elif pool_type == "min":
                    pooled = emb.masked_fill(mask == 0, float("inf")).min(dim=1).values
                else:
                    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                outputs.append(pooled.cpu().numpy())
        return np.vstack(outputs).astype("float32")


class TaTSForecastModel(nn.Module):
    def __init__(self, backbone: str, seq_len: int, label_len: int, pred_len: int, text_dim: int = 12, llm_dim: int = 768, use_text: bool = True, d_model: int = 32, n_heads: int = 4, e_layers: int = 1, d_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.use_text = use_text
        self.pred_len = pred_len
        channels = 1 + (text_dim if use_text else 0)
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=channels,
            dec_in=channels,
            c_out=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            factor=1,
            dropout=dropout,
            embed="timeF",
            freq="w",
            activation="gelu",
            output_attention=False,
            num_class=2,
        )
        self.backbone_name = backbone
        self.backbone = load_official_backbone(backbone, cfg)
        if use_text:
            self.text_projection = nn.Sequential(
                nn.Linear(llm_dim, max(4, llm_dim // 8)),
                nn.ReLU(),
                nn.Linear(max(4, llm_dim // 8), text_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
        else:
            self.text_projection = None

    def forward(self, x_num: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_text:
            if text_emb is None:
                raise ValueError("text_emb is required when use_text=True")
            prompt_emb = self.text_projection(text_emb)
            x = torch.cat([x_num, prompt_emb], dim=-1)
        else:
            x = x_num
        dec_inp = torch.zeros((x.shape[0], self.pred_len, x.shape[-1]), dtype=x.dtype, device=x.device)
        out = self.backbone(x, None, dec_inp, None)
        return out[:, -self.pred_len :, :1]
