from cen_tats.data.split_manager import temporal_v5_split
from cen_tats.data.leakage_checks import assert_temporal_order


def test_temporal_split_order() -> None:
    split = temporal_v5_split(100)
    assert len(split.train_core) == 60
    assert len(split.prompt_dev) == 10
    assert len(split.model_val) == 10
    assert len(split.test) == 20
    assert_temporal_order(split)
