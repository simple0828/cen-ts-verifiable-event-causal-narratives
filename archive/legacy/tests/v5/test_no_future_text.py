import pytest

from cen_tats.data.leakage_checks import assert_window_has_no_future_text


def test_no_future_text_passes() -> None:
    assert_window_has_no_future_text(10, list(range(5, 11)))


def test_no_future_text_fails() -> None:
    with pytest.raises(AssertionError):
        assert_window_has_no_future_text(10, [9, 11])
