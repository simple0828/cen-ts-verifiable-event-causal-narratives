from cen_tats.runtime.preflight import check_python_environment


def test_python_executable_is_tats() -> None:
    result = check_python_environment("D:/Miniconda/envs/tats/python.exe")
    assert result["python_path_ok"] is True
