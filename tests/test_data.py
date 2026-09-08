from pathlib import Path

from cen_ts.data.prepare import prepare_environment_data


def test_current_dataset_is_loadable():
    result = prepare_environment_data(Path("data/processed/Environment.csv"))
    assert result["rows"] == 15248
    assert {"date", "OT", "fact"}.issubset(result["columns"])
