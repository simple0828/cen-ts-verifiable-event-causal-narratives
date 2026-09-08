# CEN-TS

## 1. Project Goal

CEN-TS studies whether grounded event descriptions can make text-paired time-series forecasting more accurate and auditable. The repository contains the reproducible Environment baseline, the current event-extraction implementation, and only the artifacts needed to verify the latest valid result.

## 2. Method Overview

The retained baseline follows TaTS: daily numeric observations and their same-timestamp text are encoded by an iTransformer and a frozen local GPT-2 text channel, then combined for 48-step forecasting. Event extraction is a separate, strict-JSON preprocessing stage. Its prompt and schema live in `src/cen_ts/events/`; extraction is restricted to the supplied source text and to the training split. The published baseline algorithm, chronological split, seed, hyperparameters, and metric definitions are unchanged.

## 3. Repository Layout

```text
.
├── configs/
│   ├── baseline.yaml
│   └── event_extraction.yaml
├── data/
│   ├── raw/
│   └── processed/Environment.csv
├── scripts/
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── extract_events.py
│   └── evaluate.py
├── src/cen_ts/
│   ├── data/
│   ├── models/
│   ├── events/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── third_party/tats/
├── results/latest/
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## 4. Installation

Python 3.10 or newer is required. Create an isolated environment and install the package with the TaTS and test dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[tats,dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[tats,dev]"
```

The full baseline requires local GPT-2 weights; network fallback is disabled. Download them once (the ignored `models/` directory is not committed):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', local_dir='models/gpt2')"
```

Set `CEN_TS_GPT2_PATH` only if the weights are stored elsewhere.

## 5. Data Preparation

The exact processed Environment table used by the retained run is included at `data/processed/Environment.csv`. Validate it without rewriting it:

```bash
python scripts/prepare_data.py --input data/processed/Environment.csv
```

To prepare the same schema from a locally supplied, non-regenerable source file, place it under the ignored `data/raw/` directory and run:

```bash
python scripts/prepare_data.py --input data/raw/Environment.csv --output data/processed/Environment.csv
```

The input must contain 15,248 rows and the `date`, `OT`, and `fact` columns. Raw sources remain local so their original contents and licensing can be managed separately.

## 6. Event Extraction

First validate the configuration, embedded prompt, schema, and dataset without an API call:

```bash
python scripts/extract_events.py --config configs/event_extraction.yaml --dry-run
```

For a real training-split extraction, configure the OpenAI-compatible endpoint variables required by `src/cen_ts/events/api_config.py`, then run:

```bash
python scripts/extract_events.py --config configs/event_extraction.yaml --output data/processed/events.jsonl
```

Responses are cached under ignored `.cache/cen_ts/events/`. API responses and cache files are not publication artifacts.

## 7. Training and Evaluation

Run a cheap CPU model-shape check:

```bash
python scripts/train_baseline.py --config configs/baseline.yaml --smoke-test
```

Reproduce the full baseline only when CUDA and local GPT-2 weights are available:

```bash
python scripts/train_baseline.py --config configs/baseline.yaml
```

Recompute the normalized metrics from the retained predictions and verify them against the published JSON:

```bash
python scripts/evaluate.py --predictions results/latest/predictions.csv --reference results/latest/metrics.json
```

Run the offline test suite:

```bash
python -m pytest -q
```

## 8. Current Results

The latest valid run is the Environment raw-text baseline recorded at source commit `2bd8b526cf5d0069da842628382847d9680987df` with seed 2025. It completed training and passed retained-source parity checks. An unfinished event-extraction study was not promoted over this result.

| Scale | MAE | MSE | RMSE | MAPE | MSPE |
|---|---:|---:|---:|---:|---:|
| Normalized | 0.369895 | 0.265448 | 0.515217 | 1.105526 | 19.349234 |
| Original | 15.501421 | 466.192383 | 21.591488 | — | — |

`results/latest/` contains the exact configuration, compact run log, core metrics, and 2,976 × 48 normalized forecasts needed for independent metric verification. The checkpoint is omitted because evaluation is reproducible from the predictions and retraining is defined by the pinned code and configuration.

## 9. Current Limitations

The valid result is a single-seed baseline on one dataset. The event extractor has an offline dry-run and schema/grounding tests, but the latest extraction study did not pass its temporal-quality gate and lacks completed manual annotation; no event-based forecasting improvement is claimed. Full retraining requires CUDA and separately downloaded GPT-2 weights, while real extraction requires a paid compatible API endpoint.
