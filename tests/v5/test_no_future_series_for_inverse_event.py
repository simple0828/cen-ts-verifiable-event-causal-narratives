import pytest

from cen_tats.data.leakage_checks import assert_inverse_uses_history_only


def test_inverse_event_uses_history_only() -> None:
    assert_inverse_uses_history_only(20, list(range(10, 21)))
    with pytest.raises(AssertionError):
        assert_inverse_uses_history_only(20, [19, 21])
