from cen_tats.data.split_manager import temporal_v5_split


def test_prompt_dev_test_isolation() -> None:
    split = temporal_v5_split(100)
    assert set(split.prompt_dev).isdisjoint(set(split.test))
    assert max(split.prompt_dev) < min(split.test)
