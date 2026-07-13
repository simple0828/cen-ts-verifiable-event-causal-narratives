import json
from pathlib import Path


def _run(method: str) -> dict:
    status = json.loads(Path("results/v6/p1/p1_status.json").read_text(encoding="utf-8"))
    run_id = next(row["run_id"] for row in status["methods"] if row["method"] == method)
    return json.loads(Path("results/v6/p1/runs") .joinpath(run_id, "input_hashes.json").read_text(encoding="utf-8"))


def test_p1_methods_share_same_numeric_test_windows() -> None:
    hashes = {m: _run(m)["x_test_hash"] for m in ["M0_Numerical-only", "M1_TaTS-Raw-Text", "Z0_Zero-Text-Control", "S0_Shuffled-Text-Diagnostic"]}
    assert len(set(hashes.values())) == 1
