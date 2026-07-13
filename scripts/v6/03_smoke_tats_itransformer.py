from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cen_tats.runtime.preflight import DEFAULT_GPT2_PATH, write_json  # noqa: E402
from cen_tats.runtime.tats_smoke import run_itransformer_smoke  # noqa: E402


RESULT_DIR = ROOT / "results" / "v6" / "preflight"
LOG_PATH = RESULT_DIR / "tats_itransformer_smoke.log"
JSON_PATH = RESULT_DIR / "tats_itransformer_smoke.json"


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    try:
        result = run_itransformer_smoke(model_path=DEFAULT_GPT2_PATH, device="cuda:0")
        write_json(JSON_PATH, result)
        logging.info("TaTS iTransformer smoke passed: %s", result)
        return 0
    except Exception as exc:
        result = {"success": False, "error": repr(exc), "traceback": traceback.format_exc()}
        write_json(JSON_PATH, result)
        logging.exception("TaTS iTransformer smoke failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
