from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_no_paid_llm_calls_in_p2_frontend():
    needles = ["OpenAI(", "AzureOpenAI(", "api_key", "chat.completions", "responses.create"]
    for path in (ROOT / "tats_cen" / "cen_ts").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in source
