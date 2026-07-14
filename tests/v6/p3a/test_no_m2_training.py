import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_no_m2_training():
    usage_path = ROOT / "results/v6/p3a/api_usage.json"
    if usage_path.exists():
        assert json.loads(usage_path.read_text(encoding="utf-8"))["event_tats_training_runs"] == 0
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "scripts/v6/p3a").glob("*.py"))
    assert "tats_cen/run.py" not in sources
