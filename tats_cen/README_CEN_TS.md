# CEN-TaTS Minimal Fork

This directory is the P2 development fork created from official TaTS commit `a053503674c61c54d101d01d47c9d680288a7c9a`.

`third_party/TaTS` remains the untouched upstream snapshot. This fork adds only the engineering interface needed for CEN-TaTS experiments:

- strict local GPT-2 loading from `--llm_path`;
- explicit `--text_mode` and `--text_column`;
- isolated `--save_root` and `--run_name`;
- `--prior_weight` protocols;
- manifests and input/text/GPT-2 hashes;
- frontend text variant interfaces under `cen_ts/`.

P2 implements only `raw`, `constant`, and `shuffled` text variants. Event extraction, causal graph construction, inverse event generation, verification, narrative generation, and APO are reserved for later stages and raise `NotImplementedError`.

The model math is intended to match official TaTS:

- `models/iTransformer.py` is unchanged;
- official GPT-2 input embedding lookup is preserved;
- official mask-aware pooling is preserved;
- official projection MLP is preserved;
- official decoder, optimizer, loss, validation, test, and prior mix are preserved.

Primary P2 scripts:

- `scripts/v6/p2/20_create_tats_cen_fork.py`
- `scripts/v6/p2/21_build_text_variant.py`
- `scripts/v6/p2/22_run_tats_cen.py`
- `scripts/v6/p2/23_compare_p1b_parity.py`
