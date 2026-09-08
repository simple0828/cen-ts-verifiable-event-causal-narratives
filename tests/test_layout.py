from pathlib import Path


def test_public_layout_has_only_supported_entry_points():
    root = Path(__file__).resolve().parents[1]
    assert {path.name for path in (root / "scripts").glob("*.py")} == {
        "prepare_data.py", "train_baseline.py", "extract_events.py", "evaluate.py"
    }
    for removed in ("archive", "artifacts", "experiments", "prompts", "reports", "external", "vendor"):
        assert not (root / removed).exists()
