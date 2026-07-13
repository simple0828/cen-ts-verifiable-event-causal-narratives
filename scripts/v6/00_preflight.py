from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cen_tats.runtime.preflight import (  # noqa: E402
    DEFAULT_GPT2_PATH,
    PreflightError,
    check_backbone,
    check_cuda,
    check_gpt2_files,
    check_no_random_fallback,
    check_python_environment,
    check_required_packages,
    load_preflight_config,
    load_pretrained_gpt2_strict,
    write_json,
)


RESULT_PATH = ROOT / "results" / "v6" / "preflight" / "preflight_checks.json"


def _pass(value: Any) -> dict[str, Any]:
    return {"status": "PASS", "details": value}


def _fail(exc: BaseException) -> dict[str, Any]:
    return {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}


def _not_checked() -> dict[str, Any]:
    return {"status": "NOT_CHECKED"}


def main() -> int:
    config = load_preflight_config()
    steps = [
        ("python_environment", lambda: check_python_environment(config.get("environment", {}).get("python_path", "D:/Miniconda/envs/tats/python.exe"))),
        ("required_packages", check_required_packages),
        ("gpt2_files", lambda: check_gpt2_files(config.get("tats", {}).get("model_path", DEFAULT_GPT2_PATH))),
        ("no_random_fallback", lambda: check_no_random_fallback(config)),
        ("backbone", lambda: check_backbone(config)),
        ("gpt2_strict_load", lambda: load_pretrained_gpt2_strict(config.get("tats", {}).get("model_path", DEFAULT_GPT2_PATH))[2]),
        ("cuda", lambda: check_cuda(config.get("tats", {}).get("model_path", DEFAULT_GPT2_PATH), int(config.get("training", {}).get("gpu", 0)))),
    ]
    results: dict[str, Any] = {}
    failed = False
    for name, func in steps:
        if failed:
            results[name] = _not_checked()
            continue
        try:
            results[name] = _pass(func())
            print(f"{name}: PASS")
        except Exception as exc:
            failed = True
            results[name] = _fail(exc)
            print(f"{name}: FAIL")
            print(exc)
    write_json(RESULT_PATH, results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
