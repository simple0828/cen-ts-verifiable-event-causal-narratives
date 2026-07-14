import json

from cen_ts.api_config import APIConfig


def test_api_key_not_logged_or_manifested():
    secret = "sk-this-must-never-appear"
    config = APIConfig(api_key=secret, base_url="https://example.invalid/v1", model="model")
    assert secret not in repr(config)
    assert secret not in json.dumps(config.safe_manifest())
    assert "api_key" not in " ".join(config.safe_manifest().keys()).replace("api_key_present", "")
