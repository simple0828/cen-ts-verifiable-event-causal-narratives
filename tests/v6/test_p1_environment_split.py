import json
from pathlib import Path


def test_p1_environment_split_is_chronological_and_hashed() -> None:
    audit = json.loads(Path("results/v6/p1/environment_data_audit.json").read_text(encoding="utf-8"))
    bounds = audit["split_boundaries"]
    assert bounds["train"][1] == bounds["model_val"][0]
    assert bounds["model_val"][1] == bounds["test"][0]
    assert bounds["train"][0] < bounds["train"][1] < bounds["model_val"][1] < bounds["test"][1]
    assert len(audit["split_hash"]) == 64
    assert audit["test_used_for_parameter_selection"] is False
    assert audit["same_split_for_all_methods"] is True
