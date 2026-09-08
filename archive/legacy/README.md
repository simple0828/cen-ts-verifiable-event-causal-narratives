# Historical implementations

These files are retained for source and experiment provenance. They are excluded
from package discovery, the active import path, and default pytest collection.
The current entry point is `python scripts/run.py <stage>` from the repository root.

- `src/cents/`: the pre-v5 baseline/event implementation; no v6 references.
- `src/cen_tats/`: v5 pipeline, events, causal, verifier, APO, and TaTS adapter.
  The v6-dependent runtime and `evaluation/forecast_metrics.py` were moved to
  `src/cen_ts/` instead of being duplicated here.
- `scripts/`, `tests/`, `configs/`: corresponding historical workflows.
- `pre_v5/`: the former root `legacy/` directory.
- `README_before_layout.md`: previous documentation, including historical commands.

Historical imports and machine paths are preserved as provenance. These archived
workflows are not supported by the active editable install. Check out the original
commit in a separate worktree when reproducing them. Prompts, data splits, reports,
and results remain at their original project paths.
