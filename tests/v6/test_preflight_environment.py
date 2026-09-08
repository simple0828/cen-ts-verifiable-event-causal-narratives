import sys
from cen_ts.runtime.preflight import check_python_environment


def test_python_executable_is_tats() -> None:
    result = check_python_environment(sys.executable)
    assert result["python_path_ok"] is True
