from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_no_paid_llm_calls_in_p2_frontend():
    needles = ["OpenAI(", "AzureOpenAI(", "api_key", "chat.completions", "responses.create"]
    p3a_api_modules = {"api_config.py", "llm_client.py", "p3a_pipeline.py"}
    for path in (ROOT / "src" / "cen_ts").glob("*.py"):
        # P3A adds an explicitly configured real-API surface. This P2 regression
        # guard remains scoped to the P2 frontend instead of banning later stages.
        if path.name in p3a_api_modules:
            continue
        source = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in source
