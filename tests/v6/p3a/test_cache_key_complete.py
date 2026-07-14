import pytest

from cen_ts.llm_cache import CACHE_KEY_FIELDS, cache_key


def test_cache_key_requires_every_component():
    complete = {field: field for field in CACHE_KEY_FIELDS}
    assert len(cache_key(complete)) == 64
    complete.pop("report_time")
    with pytest.raises(ValueError):
        cache_key(complete)
