import pytest

from cen_tats.data.leakage_checks import assert_train_only_graph


def test_train_only_graph() -> None:
    assert_train_only_graph({"split_used": "train_core"})
    with pytest.raises(AssertionError):
        assert_train_only_graph({"split_used": "test"})
