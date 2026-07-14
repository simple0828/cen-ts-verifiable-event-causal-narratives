# tats_cen Upstream Diff Audit

| File path | Modification type | Reason | Changes model math? | Changes text encoding? | Changes training protocol? | Affects attribution? |
| --- | --- | --- | --- | --- | --- | --- |
| `models/iTransformer.py` | UNCHANGED | Upstream model backbone preserved | No | No | No | No |
| `data_provider/data_loader.py` | Modified | Add explicit text column checks, manifests, and official-equivalent missing-text fill | No | No | No | No |
| `data_provider/m4.py` | Modified | Optional `patoolib` stub for non-M4 Environment imports | No | No | No | No |
| `exp/exp_long_term_forecasting.py` | Modified | Replace GPT-2 network fallback with strict local loader and manifest | No | No | No | No |
| `run.py` | Modified | Add CEN-TaTS CLI, run isolation, manifests, and text mode resolution | No | No | No | No |
| `utils/strict_llm.py` | Added | Strict local GPT-2 loading helper | No | No | No | No |
| `cen_ts/*` | Added | Unified text variant frontend and reserved NotImplemented interfaces | No | No | No | Enables later attribution by text input only |

- Official pooling: UNCHANGED.
- Official projection: UNCHANGED.
- Official training loop math: UNCHANGED; P2 adds logging/args/path isolation only.
