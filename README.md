# CEN-TS: Verifiable Causal Event Narratives for Text-Paired Time Series Forecasting

## Project Overview

CEN-TS is a first-stage experimental system for text-paired time series forecasting. It tests whether structured, verifiable event-causal narratives can improve forecasting, robustness, and explanation groundedness compared with numerical-only and raw text embedding fusion baselines.

## Research Hypothesis

Instead of directly concatenating text embeddings with numerical time-series windows, CEN-TS converts timestamped text into structured events, builds lagged event-variable causal graphs, verifies event explanations against observed time-series changes, and uses the verified event features or narratives for forecasting.

## Relation to Prior Work

- Augur: CEN-TS includes an Augur-lite lagged association module for variable-variable causal summaries.
- TaTS: CEN-TS implements a raw text-as-time-series embedding baseline.
- Inferring Events from Time Series: CEN-TS implements an event consistency verifier that checks whether post-event time-series movement agrees with event polarity and lag.
- APO / PromptWizard: CEN-TS includes a lightweight Narrative-APO interface for future prompt-program optimization.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data Download

The project intentionally downloads the public datasets into this repository's local data and external folders:

```bash
mkdir -p data/raw external
git clone https://github.com/AdityaLab/Time-MMD.git data/raw/Time-MMD
git clone https://github.com/AdityaLab/MM-TSFlib.git external/MM-TSFlib
git clone https://github.com/BennyTMT/GAMETime.git data/raw/GAMETime
```

Optional reference:

```bash
git clone https://github.com/microsoft/PromptWizard.git external/PromptWizard
```

Large raw and processed data are ignored by git.

## Quick Start

```bash
python scripts/00_check_environment.py
python scripts/01_prepare_timemmd.py --raw_dir data/raw/Time-MMD --out_dir data/processed/TimeMMD
python scripts/03_run_smoke_test.py --config configs/exp/smoke_test.yaml
python scripts/12_collect_results.py
python scripts/13_generate_report.py
```

## Main Experiments

```bash
python scripts/04_run_numeric_baselines.py --config configs/exp/main_timemmd.yaml
python scripts/05_run_text_embedding_baselines.py --config configs/exp/main_timemmd.yaml
python scripts/06_extract_events.py --config configs/exp/main_timemmd.yaml
python scripts/07_build_event_graph.py --config configs/exp/main_timemmd.yaml
python scripts/08_run_event_feature_models.py --config configs/exp/main_timemmd.yaml
python scripts/09_run_verifier_ablation.py --config configs/exp/ablation_timemmd.yaml
python scripts/11_run_robustness_tests.py --config configs/exp/robustness_timemmd.yaml
python scripts/14_run_granger_lite.py --config configs/exp/main_timemmd.yaml
python scripts/15_run_gametime_sanity.py
python scripts/16_scan_mmtsflib.py
python scripts/17_run_llm_event_probe.py --config configs/exp/smoke_test.yaml --max_rows 3
```

## Results

The current preliminary report is generated at [reports/experiment_report.md](reports/experiment_report.md).

## Manual GitHub Push

If GitHub CLI is unavable or unauthenticated:

```bash
git remote add origin https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives.git
git branch -M main
git push -u origin main
```

## Citation Placeholder

Please cite the final versions of:

- Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models.
- Inferring Event Descriptions from Time Series with Language Models.
- Language in the Flow of Time / Texts as Time Series.
- Automatic Prompt Optimization / PromptWizard.
