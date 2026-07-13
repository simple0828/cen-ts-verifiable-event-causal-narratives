import json
from pathlib import Path


def test_p1_records_no_paid_llm_calls() -> None:
    status = json.loads(Path("results/v6/p1/p1_status.json").read_text(encoding="utf-8"))
    assert status["paid_llm_calls"] == 0
