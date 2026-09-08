import json
from pathlib import Path
from cen_ts.paths import ROOT


def test_p1_each_method_has_isolated_checkpoint_dir() -> None:
    status = json.loads(Path("results/v6/p1/p1_status.json").read_text(encoding="utf-8"))
    # Historical manifests retain their original absolute paths after relocation.
    dirs = [ROOT / "results/v6/p1/runs" / Path(row["run_dir"].replace("\\", "/")).name for row in status["methods"]]
    assert len(dirs) == len(set(map(str, dirs))) == 4
    for run_dir in dirs:
        assert run_dir.joinpath("best_checkpoint.pth").exists()
        assert run_dir.joinpath("manifest.json").exists()
