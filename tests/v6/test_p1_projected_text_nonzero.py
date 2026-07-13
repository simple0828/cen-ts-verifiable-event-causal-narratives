import json
from pathlib import Path


def test_p1_m1_projected_text_is_nonzero() -> None:
    diag = json.loads(Path("results/v6/p1/input_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["M1_TaTS-Raw-Text"]["projected_text_nonzero"] is True
    assert diag["embedding_statistics"]["variance"] > 0
