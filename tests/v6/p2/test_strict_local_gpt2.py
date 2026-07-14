import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tats_cen"))

from utils.strict_llm import build_gpt2_manifest


def test_strict_local_gpt2_manifest():
    manifest = json.loads((ROOT / "results" / "v6" / "p2" / "runs" / "p2_raw_official_reproduction_pw0.5_s2025" / "gpt2_manifest.json").read_text(encoding="utf-8"))
    assert manifest["local_files_only"] is True
    assert manifest["model_class"] == "GPT2Model"
    assert manifest["hidden_size"] == 768
    assert manifest["random_init"] is False
    assert build_gpt2_manifest("D:/models/gpt2")["weight_sha256"] == manifest["weight_sha256"]
