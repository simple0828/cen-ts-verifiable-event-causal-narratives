from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.runtime.preflight import DEFAULT_GPT2_PATH, check_cuda, nvidia_smi_text, write_json  # noqa: E402


RESULT_DIR = ROOT / "results" / "v6" / "preflight"
LOG_PATH = RESULT_DIR / "cuda_verification.log"
JSON_PATH = RESULT_DIR / "cuda_verification.json"


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    try:
        logging.info("nvidia-smi output follows:\n%s", nvidia_smi_text())
        result = check_cuda(DEFAULT_GPT2_PATH, gpu=0)
        result["success"] = True
        write_json(JSON_PATH, result)
        logging.info("CUDA verification passed.")
        return 0
    except Exception as exc:
        result = {"success": False, "error": repr(exc), "traceback": traceback.format_exc()}
        write_json(JSON_PATH, result)
        logging.exception("CUDA verification failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
