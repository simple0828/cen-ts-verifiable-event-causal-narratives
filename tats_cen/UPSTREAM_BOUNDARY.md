# CEN-TaTS Upstream Boundary

## A. Allowed Changes

- `run.py` argument interface.
- Local GPT-2 path parameterization.
- Strict offline GPT-2 loading.
- Text column selection.
- Text variant input.
- `prior_weight` parameterization.
- Logging.
- Manifests.
- Hash recording.
- Cache keys.
- Isolated experiment directories.
- CEN-TaTS frontend modules under `cen_ts/`.
- Automated tests.

## B. Forbidden Changes

- `models/iTransformer.py`.
- Official attention.
- Official forecasting head.
- Official normalization.
- Official text input embedding route.
- Official mask-average pooling.
- Official projection MLP.
- Official decoder input logic.
- Official loss.
- Official optimizer.
- Official training protocol.

## C. Only Methodological Change In Later Stages

The only intended change for later CEN-TaTS methods is the time-aligned text content fed into TaTS. Event extraction, causal graphs, verification, and APO must enter through text variants, not through model-architecture changes.
