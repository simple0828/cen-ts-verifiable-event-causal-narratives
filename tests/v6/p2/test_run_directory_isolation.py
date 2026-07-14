from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "results" / "v6" / "p2" / "runs"


def test_run_directory_isolation():
    raw = RUNS / "p2_raw_official_reproduction_pw0.5_s2025"
    constant = RUNS / "p2_constant_smoke_pw0.5_s2025"
    shuffled = RUNS / "p2_shuffled_smoke_pw0.5_s2025"
    for run in [raw, constant, shuffled]:
        assert (run / "test_metrics.json").exists()
        assert (run / "checkpoint_path.txt").exists()
    assert raw != constant != shuffled
    assert (raw / "checkpoint_path.txt").read_text(encoding="utf-8") != (constant / "checkpoint_path.txt").read_text(encoding="utf-8")
