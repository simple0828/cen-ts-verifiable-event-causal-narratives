from cen_tats.tats.text_cache import cache_key


def test_cache_hash_changes_with_prompt() -> None:
    a = cache_key(["x"], "p1", "m", "module", "d", {"x": 1})
    b = cache_key(["x"], "p2", "m", "module", "d", {"x": 1})
    assert a != b
