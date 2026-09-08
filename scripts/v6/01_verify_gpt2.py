from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cen_ts.runtime.preflight import (  # noqa: E402
    DEFAULT_GPT2_PATH,
    encode_texts_masked_average,
    force_hf_offline,
    load_pretrained_gpt2_strict,
    write_json,
)


RESULT_DIR = ROOT / "results" / "v6" / "preflight"
LOG_PATH = RESULT_DIR / "gpt2_verification.log"
JSON_PATH = RESULT_DIR / "gpt2_verification.json"


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    force_hf_offline()
    texts = [
        "A major oil producer announced a production cut.",
        "A severe weather event disrupted regional energy supply.",
        "No material event was reported.",
    ]
    try:
        tokenizer, model, metadata = load_pretrained_gpt2_strict(DEFAULT_GPT2_PATH, pretrained=True, local_files_only=True)
        embeddings = encode_texts_masked_average(tokenizer, model, texts, max_length=256)
        import torch
        import torch.nn.functional as F

        cos = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        pairwise = cos.detach().cpu().tolist()
        embedding_variance = float(embeddings.var(unbiased=False).item())
        not_identical = bool(not torch.allclose(embeddings[0], embeddings[1]) and not torch.allclose(embeddings[0], embeddings[2]) and not torch.allclose(embeddings[1], embeddings[2]))
        result = {
            "success": True,
            **metadata,
            "texts": texts,
            "embedding_shape": list(embeddings.shape),
            "embedding_variance": embedding_variance,
            "pairwise_cosine_similarity": pairwise,
            "embeddings_not_identical": not_identical,
            "network_requests_allowed": False,
            "hf_hub_offline": True,
        }
        if embedding_variance <= 0 or not not_identical:
            raise RuntimeError(f"GPT-2 embeddings failed nonconstant check: variance={embedding_variance}, not_identical={not_identical}")
        logging.info("GPT-2 strict local verification passed.")
        write_json(JSON_PATH, result)
        return 0
    except Exception as exc:
        result = {"success": False, "error": repr(exc), "traceback": traceback.format_exc()}
        logging.exception("GPT-2 strict local verification failed.")
        write_json(JSON_PATH, result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
