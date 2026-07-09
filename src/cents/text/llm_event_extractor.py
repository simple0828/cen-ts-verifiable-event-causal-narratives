from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from cents.text.prompt_templates import EVENT_EXTRACTION_PROMPT
from cents.utils.io import ensure_dir, write_json


def run_llm_event_probe(
    df: pd.DataFrame,
    domain: str,
    target_variable: str = "OT",
    max_rows: int = 3,
    out_path: str | Path = "experiments/tables/llm_event_extraction_status.json",
    model: str | None = None,
) -> dict:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    status = {
        "domain": domain,
        "target_variable": target_variable,
        "max_rows": max_rows,
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "model": model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "attempted": False,
        "success": False,
        "llm_calls": 0,
        "events": [],
        "error": "",
    }
    if not status["api_key_present"]:
        status["error"] = "OPENAI_API_KEY is not set."
        write_json(out_path, status)
        return status
    try:
        from openai import OpenAI

        client = OpenAI()
        variables = [c for c in df.select_dtypes(include="number").columns.tolist() if c != "text_rows"]
        rows = df[df.get("text", "").astype(str).str.len() > 20].head(max_rows)
        for _, row in rows.iterrows():
            prompt = (
                EVENT_EXTRACTION_PROMPT
                .replace("{domain}", domain)
                .replace("{target_variable}", target_variable)
                .replace("{variables}", ", ".join(variables[:20]))
                .replace("{timestamp}", str(row.get("date", "")))
                .replace("{text}", str(row.get("text", ""))[:2500])
            )
            status["attempted"] = True
            response = client.chat.completions.create(
                model=status["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
            )
            status["llm_calls"] += 1
            content = response.choices[0].message.content or ""
            parsed = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            status["events"].extend(parsed)
        status["success"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, status)
    return status
