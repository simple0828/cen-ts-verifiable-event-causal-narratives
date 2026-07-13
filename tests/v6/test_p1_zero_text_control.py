import json
from pathlib import Path


def test_p1_zero_text_control_keeps_shape_but_zeroes_projected_text() -> None:
    diag = json.loads(Path("results/v6/p1/input_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["Z0_Zero-Text-Control"]["combined_input_shape"] == diag["M1_TaTS-Raw-Text"]["combined_input_shape"]
    assert diag["Z0_Zero-Text-Control"]["projected_text_all_zero"] is True
    assert diag["Z0_Zero-Text-Control"]["combined_input_hash"] != diag["M1_TaTS-Raw-Text"]["combined_input_hash"]
