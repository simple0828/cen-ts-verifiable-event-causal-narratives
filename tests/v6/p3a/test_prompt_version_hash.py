import hashlib
from pathlib import Path

from cen_ts.event_extractor import PROMPT_VERSION


ROOT = Path(__file__).resolve().parents[3]


def test_prompt_version_hash():
    prompt = ROOT / "prompts/v6/event_extraction/p_extract_v1.txt"
    schema = ROOT / "prompts/v6/event_extraction/schema_v1.json"
    assert PROMPT_VERSION == "p_extract_v1"
    assert len(hashlib.sha256(prompt.read_bytes()).hexdigest()) == 64
    assert len(hashlib.sha256(schema.read_bytes()).hexdigest()) == 64
