from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gpt2_manifest(llm_path: str | Path, model: Any | None = None) -> dict[str, Any]:
    path = Path(llm_path)
    safetensors = path / "model.safetensors"
    pytorch_bin = path / "pytorch_model.bin"
    weight_file = safetensors if safetensors.exists() else pytorch_bin
    if not weight_file.exists():
        raise FileNotFoundError(f"No GPT-2 weight file found under {path}.")
    manifest = {
        "llm_path": str(path),
        "local_files_only": True,
        "weight_file": str(weight_file),
        "weight_sha256": sha256_file(weight_file),
        "model_class": type(model).__name__ if model is not None else None,
        "hidden_size": int(getattr(getattr(model, "config", None), "hidden_size", 0)) if model is not None else None,
        "random_init": False,
    }
    return manifest


def assert_no_random_init(model: Any, manifest: dict[str, Any]) -> None:
    if manifest.get("random_init"):
        raise RuntimeError("Random GPT-2 initialization is forbidden.")
    if type(model).__name__ != "GPT2Model":
        raise RuntimeError(f"Expected GPT2Model, got {type(model).__name__}.")
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != 768:
        raise RuntimeError(f"Expected GPT-2 hidden_size=768, got {hidden_size}.")


def load_gpt2_strict(llm_path: str | Path, llm_layers: int | None = None) -> tuple[Any, Any, dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

    path = Path(llm_path)
    if not path.exists():
        raise FileNotFoundError(f"Strict local GPT-2 path does not exist: {path}")

    config = GPT2Config.from_pretrained(str(path), local_files_only=True)
    if llm_layers is not None:
        config.num_hidden_layers = llm_layers
    config.output_attentions = True
    config.output_hidden_states = True
    model = GPT2Model.from_pretrained(str(path), local_files_only=True, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained(str(path), local_files_only=True)
    manifest = build_gpt2_manifest(path, model)
    assert_no_random_init(model, manifest)
    return model, tokenizer, manifest


def write_gpt2_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
