import pytest

from cen_tats.runtime.preflight import PreflightError, check_no_random_fallback


def test_random_embedding_fallback_is_forbidden() -> None:
    with pytest.raises(PreflightError):
        check_no_random_fallback({"tats": {"pretrained": True, "allow_random_fallback": True, "local_files_only": True}})


def test_pretrained_local_only_config_passes() -> None:
    result = check_no_random_fallback({"tats": {"pretrained": True, "allow_random_fallback": False, "local_files_only": True}})
    assert result["random_fallback_forbidden"] is True
