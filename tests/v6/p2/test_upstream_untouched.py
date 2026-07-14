import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SAFE_TATS = "C:/Users/Administrator/Desktop/TS/cen-ts-verifiable-event-causal-narratives/third_party/TaTS"


def test_upstream_untouched():
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(ROOT / "third_party" / "TaTS"), "status", "--short"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == ""
