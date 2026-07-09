from __future__ import annotations

from pathlib import Path

import pandas as pd

from cents.utils.io import ensure_dir, read_json


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No completed results yet._"
    return df.head(max_rows).to_markdown(index=False)


def generate_report(tables_dir: str | Path = "experiments/tables", reports_dir: str | Path = "reports") -> Path:
    tables_dir = Path(tables_dir)
    reports_dir = ensure_dir(reports_dir)
    main = pd.read_csv(tables_dir / "main_results.csv") if (tables_dir / "main_results.csv").exists() else pd.DataFrame()
    ablation = pd.read_csv(tables_dir / "ablation.csv") if (tables_dir / "ablation.csv").exists() else pd.DataFrame()
    robustness = pd.read_csv(tables_dir / "robustness.csv") if (tables_dir / "robustness.csv").exists() else pd.DataFrame()
    cases = []
    for case_file in Path("experiments/runs").glob("*/case_studies.json"):
        cases.extend(read_json(case_file, default=[]))
    if not main.empty and "event_with_verifier" in set(main["method"]) and "raw_text_embedding" in set(main["method"]):
        ev = main[main["method"] == "event_with_verifier"]["mse"].mean()
        raw = main[main["method"] == "raw_text_embedding"]["mse"].mean()
        judgement = "Preliminary support: verifier-filtered event features outperform raw text embeddings on mean MSE." if ev < raw else "Mixed: verifier-filtered event features did not beat raw text embedding on mean MSE in this lightweight run."
    else:
        judgement = "Insufficient completed results for a strong claim."

    content = f"""# CEN-TS: Verifiable Causal Event Narratives for Text-Paired Time Series Forecasting

## Abstract

This report documents a first-stage reproducible prototype for text-paired time series forecasting. CEN-TS extracts structured events from timestamped text, builds lagged event-variable graphs, verifies event explanations against observed target movement, and compares verified event features against numerical-only and raw text embedding baselines. Results are preliminary and use lightweight models to validate the experimental loop.

## Motivation

Raw text embeddings can inject useful context, but they are difficult to audit and can be brittle under irrelevant text. Event-causal narratives expose intermediate claims: what happened, what variable it should affect, with what polarity, lag, and confidence. A verifier then checks whether the claimed event is directionally consistent with the numerical sequence.

## Related Experimental Anchors

- TaTS: implemented as `raw_text_embedding` using TF-IDF + TruncatedSVD text-as-time-series features.
- Augur: represented by lagged variable-variable correlations in the causal graph.
- Inferring Events: represented by event consistency scoring from pre/post target movement and counterfactual windows.
- APO: represented by a lightweight prompt-program interface; no API calls were required for this preliminary run.

## Dataset

Time-MMD was cloned into `data/raw/Time-MMD` and processed into aligned numerical/text CSV files under `data/processed/TimeMMD`. The loader discovers available domains, joins textual `fact/preds` records to numerical timestamps, and records coverage statistics. GAMETime was cloned into `data/raw/GAMETime` and registered for verifier sanity checks in the next appendix-style experiment.

Default split is temporal: 70% train, 10% validation, 20% test. The smoke configuration uses `L=24` and horizons `H in {{3, 6}}` unless frequency-specific settings are expanded.

## Methods

- Numerical-only: ridge regression over flattened numerical windows.
- TaTS-style fusion: numerical windows plus TF-IDF/SVD text embeddings.
- Augur-lite: lagged correlation graph among numerical variables.
- Event feature model: keyword and polarity based structured event extraction.
- Event graph: event-to-variable edges plus lagged variable-variable edges.
- Event consistency verifier: pre/post target delta, direction agreement, lag effect, and random-window counterfactual lift.
- Full CEN-TS: verified event features in the forecasting model, plus sample causal narratives.
- Narrative-APO: interface implemented; full LLM optimization is deferred to the next stage.

## Metrics

Forecasting metrics include MSE, MAE, RMSE, MAPE, and directional accuracy. Explanation metrics include event consistency, groundedness, selected event count, and narrative token length. Robustness metrics include MSE changes under shuffled text and random text injection.

## Main Results

{_md_table(main)}

## Ablation Study

{_md_table(ablation)}

## Robustness Study

{_md_table(robustness)}

## Case Study

"""
    for idx, case in enumerate(cases[:2], start=1):
        event = case.get("event", {})
        content += f"""### Case {idx}: {case.get('domain', 'unknown')}

- Raw text: {case.get('raw_text', '')[:500]}
- Extracted event: {event.get('event_phrase', '')[:300]}
- Polarity / lag: {event.get('polarity', 'uncertain')} / {event.get('expected_lag_min', 0)}-{event.get('expected_lag_max', 0)}
- Verifier score: {float(event.get('final_consistency_score', 0.0)):.3f}
- Graph edges: {case.get('graph_edges', [])}
- Final narrative excerpt:

```text
{case.get('narrative', '')[:900]}
```

"""
    content += f"""## Failure Analysis

- Rule-based event extraction can confuse generic price trend statements with actionable causal events.
- Lag ranges are currently coarse and fixed to 1-3 time steps for most extracted events.
- Some Time-MMD domains have sparse text coverage over early numerical history, so text features may affect only later windows.
- The lightweight ridge model is useful for pipeline validation but is not a replacement for stronger PatchTST, DLinear, iTransformer, or MM-TSFlib baselines.

## Conclusion

{judgement} The current evidence should be treated as preliminary. The most valuable completed contribution is the closed-loop experimental harness: aligned data, baselines, structured events, graph construction, verifier scoring, robustness checks, and reproducible reporting.

## Next Experiments

1. Strengthen baselines by aligning with MM-TSFlib and TaTS settings.
2. Strengthen event extraction with a stronger LLM and domain-specific event schemas.
3. Add Granger, PCMCI, and transfer entropy for causal edge validation.
4. Improve verifier with matched counterfactual windows and bootstrap significance.
5. Run full Narrative-APO using validation errors to refine prompt programs.
6. Expand domains and use OOD temporal splits.
7. Add human or LLM-as-judge evaluation for narrative groundedness.
8. Prepare paper-quality figures, tables, and appendix diagnostics.
"""
    md_path = reports_dir / "experiment_report.md"
    md_path.write_text(content, encoding="utf-8")
    tex = content.replace("# ", "\\section*{").replace("\n## ", "}\n\\section*{").replace("\n### ", "}\n\\subsection*{") + "}\n"
    (reports_dir / "experiment_report.tex").write_text(tex, encoding="utf-8")
    return md_path

