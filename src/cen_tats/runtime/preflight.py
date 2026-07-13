from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = "D:/Miniconda/envs/tats/python.exe"
DEFAULT_GPT2_PATH = "D:/models/gpt2"
REQUIRED_GPT2_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "pandas",
    "sklearn",
    "tqdm",
    "matplotlib",
    "sktime",
    "reformer_pytorch",
)


class PreflightError(RuntimeError):
    """Raised when a hard P0 preflight requirement is not satisfied."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_imports() -> None:
    root = repo_root()
    src = root / "src"
    tats = root / "third_party" / "TaTS"
    for path in (src, tats):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path))).replace("\\", "/")


def read_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PreflightError(f"Config must be a mapping: {path}")
    return data


def config_path() -> Path:
    root = repo_root()
    local = root / "configs" / "v6" / "preflight.local.yaml"
    if local.exists():
        return local
    return root / "configs" / "v6" / "preflight.example.yaml"


def load_preflight_config(path: str | Path | None = None) -> dict[str, Any]:
    return read_yaml_config(path or config_path())


def check_python_environment(expected_python: str = EXPECTED_PYTHON) -> dict[str, Any]:
    actual = normalized_path(sys.executable)
    expected = normalized_path(expected_python)
    ok = actual == expected
    result = {
        "expected_python": expected_python,
        "sys_executable": sys.executable,
        "python_version": sys.version,
        "python_path_ok": ok,
    }
    if not ok:
        raise PreflightError(f"Python executable mismatch: expected {expected_python}, got {sys.executable}")
    return result


def check_required_packages(packages: tuple[str, ...] = REQUIRED_PACKAGES) -> dict[str, Any]:
    found = {name: importlib.util.find_spec(name) is not None for name in packages}
    missing = [name for name, ok in found.items() if not ok]
    result = {"packages": found, "missing": missing}
    if missing:
        raise PreflightError(f"Missing required packages: {', '.join(missing)}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpt2_file_manifest(model_path: str | Path = DEFAULT_GPT2_PATH) -> dict[str, Any]:
    base = Path(model_path)
    files = []
    for name in REQUIRED_GPT2_FILES:
        path = base / name
        if path.exists():
            stat = path.stat()
            files.append(
                {
                    "file": name,
                    "path": str(path),
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                    "last_write_time": stat.st_mtime,
                }
            )
        else:
            files.append(
                {
                    "file": name,
                    "path": str(path),
                    "exists": False,
                    "size_bytes": None,
                    "sha256": None,
                    "last_write_time": None,
                }
            )
    return {
        "model_path": str(base),
        "model_dir_exists": base.exists(),
        "required_files": files,
        "complete": all(item["exists"] for item in files),
    }


def check_gpt2_files(model_path: str | Path = DEFAULT_GPT2_PATH) -> dict[str, Any]:
    manifest = gpt2_file_manifest(model_path)
    missing = [item["file"] for item in manifest["required_files"] if not item["exists"]]
    if missing:
        raise PreflightError(f"GPT-2 files incomplete under {model_path}: {', '.join(missing)}")
    return manifest


def force_hf_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_pretrained_gpt2_strict(
    model_path: str | Path = DEFAULT_GPT2_PATH,
    *,
    pretrained: bool = True,
    local_files_only: bool = True,
    device: str | None = None,
):
    if not pretrained:
        raise PreflightError("Formal CEN-TaTS v6 requires pretrained=True for GPT-2.")
    if not local_files_only:
        warnings.warn("local_files_only=False may allow network access; formal preflight should keep it True.", RuntimeWarning)
    if local_files_only:
        force_hf_offline()

    check_gpt2_files(model_path)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=local_files_only)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(str(model_path), local_files_only=local_files_only)
    if device:
        model = model.to(device)
    model.eval()

    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", -1)))
    if hidden_size != 768:
        raise PreflightError(f"GPT-2 hidden size must be 768, got {hidden_size}")

    param_count = sum(int(p.numel()) for p in model.parameters())
    if not 100_000_000 <= param_count <= 200_000_000:
        raise PreflightError(f"GPT-2 parameter count looks wrong: {param_count}")

    import torch

    all_finite = all(bool(torch.isfinite(p.detach()).all().item()) for p in model.parameters())
    if not all_finite:
        raise PreflightError("GPT-2 contains non-finite parameters.")

    metadata = {
        "model_class": type(model).__name__,
        "hidden_size": hidden_size,
        "parameter_count": param_count,
        "all_parameters_finite": all_finite,
        "model_path": str(model_path),
        "pretrained": True,
        "local_files_only": local_files_only,
        "random_init": False,
    }
    return tokenizer, model, metadata


def encode_texts_masked_average(tokenizer, model, texts: list[str], *, max_length: int = 256, device: str | None = None):
    import torch

    if device is None:
        device = next(model.parameters()).device
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    hidden = outputs.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


def check_no_random_fallback(config: dict[str, Any]) -> dict[str, Any]:
    tats = config.get("tats", config)
    pretrained = bool(tats.get("pretrained", True))
    allow_random = bool(tats.get("allow_random_fallback", False))
    local_files_only = bool(tats.get("local_files_only", True))
    lm_mode = str(tats.get("lm_mode", "pretrained")).lower()
    if not pretrained:
        raise PreflightError("pretrained=False is forbidden for formal CEN-TaTS v6.")
    if allow_random:
        raise PreflightError("Random embedding fallback is forbidden for formal CEN-TaTS v6.")
    if "random" in lm_mode or "fallback" in lm_mode:
        raise PreflightError(f"Random/fallback lm_mode is forbidden: {lm_mode}")
    if not local_files_only:
        warnings.warn("local_files_only=False is discouraged for reproducible preflight.", RuntimeWarning)
    return {
        "pretrained": pretrained,
        "allow_random_fallback": allow_random,
        "local_files_only": local_files_only,
        "lm_mode": lm_mode,
        "random_fallback_forbidden": True,
    }


def check_backbone(config_or_backbone: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(config_or_backbone, str):
        backbone = config_or_backbone
    else:
        backbone = str(config_or_backbone.get("training", {}).get("backbone", ""))
    if backbone == "PatchTST":
        raise PreflightError("PatchTST is forbidden for formal CEN-TaTS v6 preflight.")
    if backbone != "iTransformer":
        raise PreflightError(f"Formal CEN-TaTS v6 requires iTransformer, got {backbone!r}")
    return {"backbone": backbone, "backbone_ok": True}


def check_cuda(model_path: str | Path = DEFAULT_GPT2_PATH, gpu: int = 0) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available():
        raise PreflightError("torch.cuda.is_available() is False.")
    if torch.cuda.device_count() <= gpu:
        raise PreflightError(f"Requested cuda:{gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")

    device = torch.device(f"cuda:{gpu}")
    result.update(
        {
            "gpu_name": torch.cuda.get_device_name(gpu),
            "device_capability": list(torch.cuda.get_device_capability(gpu)),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    )
    if "NVIDIA" not in result["gpu_name"]:
        raise PreflightError(f"GPU name does not contain NVIDIA: {result['gpu_name']}")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result["mem_get_info"] = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}

    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = a @ b
    torch.cuda.synchronize(device)
    if c.device.type != "cuda":
        raise PreflightError(f"CUDA matmul output is on {c.device}, expected cuda:{gpu}")
    if not bool(torch.isfinite(c).all().item()):
        raise PreflightError("CUDA matmul produced non-finite values.")

    tokenizer, model, gpt2_meta = load_pretrained_gpt2_strict(model_path, device=str(device))
    text_batch = tokenizer(["CUDA GPT-2 forward check."], return_tensors="pt", padding=True, truncation=True, max_length=32)
    text_batch = {key: value.to(device) for key, value in text_batch.items()}
    with torch.no_grad():
        out = model(**text_batch)
    if out.last_hidden_state.device.type != "cuda":
        raise PreflightError(f"GPT-2 output device is {out.last_hidden_state.device}, expected cuda:{gpu}")
    if not bool(torch.isfinite(out.last_hidden_state).all().item()):
        raise PreflightError("GPT-2 CUDA forward produced non-finite values.")

    result.update(
        {
            "gpu_matmul_ok": True,
            "gpu_forward_ok": True,
            "gpt2_model_class": gpt2_meta["model_class"],
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    )
    return result


def nvidia_smi_text() -> str:
    completed = subprocess.run(["nvidia-smi"], check=False, text=True, capture_output=True)
    return completed.stdout + completed.stderr


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def run_all_preflight_checks(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_preflight_config()
    env = config.get("environment", {})
    tats = config.get("tats", {})
    training = config.get("training", {})
    return {
        "python_environment": check_python_environment(env.get("python_path", EXPECTED_PYTHON)),
        "required_packages": check_required_packages(),
        "gpt2_files": check_gpt2_files(tats.get("model_path", DEFAULT_GPT2_PATH)),
        "no_random_fallback": check_no_random_fallback(config),
        "backbone": check_backbone(config),
        "cuda": check_cuda(tats.get("model_path", DEFAULT_GPT2_PATH), int(training.get("gpu", 0))),
    }
