# P1b Exact Official TaTS Reproduction Report

## 1. Scope
This run directly uses `third_party/TaTS` official `data_provider`, `exp`, `models`, and training/test loops. It does not run event extraction, causal graph construction, verifier, APO, or paid LLM APIs.

## 2. Upstream State
- upstream_commit: `a053503674c61c54d101d01d47c9d680288a7c9a`
- upstream_status: `clean`
- upstream_diff_patch: `results/v6/p1b_official_tats/upstream_diff.patch`

## 3. Command
`D:/Miniconda/envs/tats/python.exe scripts/v6/14_run_p1b_exact_official_tats.py`

## 4. Preflight
- GPT-2 path: `D:\models\gpt2`
- GPT-2 class: `GPT2Model`
- random_init: `False`
- CUDA: `True`
- model: `iTransformer`
- fact column exists: `True`
- pool_type: `avg`

## 5. Official Resolved Args
- dataset: Environment
- model: iTransformer
- seq_len/label_len/pred_len: 24/12/48
- text_emb: 12
- seed: 2025
- train_epochs/patience: 5/5
- prior_weight: 0.5
- use_amp: False
- llm_layers: 6. This is the official script setting; the run still uses the local pretrained GPT-2 input embedding table from `D:/models/gpt2`.

## 6. P1 Adapter M1 vs P1b Official M1
| Item | P1 adapter M1 | P1b official M1 | Same? |
| --- | --- | --- | --- |
| Text encoding | GPT-2 forward hidden states, mask-average pooled | `llm_model.get_input_embeddings()(input_ids)`, mask-average pooled | No |
| GPT-2 path | `D:/models/gpt2` | `D:/models/gpt2` via strict monkeypatch wrapper | Yes |
| Pooling | avg | avg | Yes |
| Projection MLP | custom P1 adapter projection | official `Linear-ReLU-Linear-ReLU-Dropout(0.3)` | No |
| iTransformer params | d_model=512, n_heads=8, e_layers=2, d_ff=2048 | d_model=512, n_heads=8, e_layers=2, d_ff=2048 | Yes |
| Data split | non-overlap P1 split windows | official TaTS 70/10/20 border windows with val/test history overlap | No |
| Train/test windows | train 10602, val 1455, test 2978 | train 10602/used 10592, val 1479/used 1472, test 3002/used 2976 | No |
| Training loop | P1 custom adapter loop | official `Exp_Long_Term_Forecast.train/test` | No |

## 7. P1b Training Metrics
| epoch | train_loss | vali_loss | test_mse | test_mae |
| --- | --- | --- | --- | --- |
| 1 | 0.7109938 | 0.3127179 | 0.2713896 | 0.3706869 |
| 2 | 0.7009331 | 0.3155992 | 0.2696501 | 0.3746452 |
| 3 | 0.69281 | 0.3164206 | 0.2688828 | 0.3765648 |
| 4 | 0.684378 | 0.3134673 | 0.2668724 | 0.374883 |
| 5 | 0.6785769 | 0.3083597 | 0.2654481 | 0.3698953 |

## 8. Test Metrics
- P1 adapter M1 unscaled MSE: `561.3306528118445`
- P1b official native scaled MSE: `0.26544812321662903`
- P1b official native scaled MAE: `0.36989524960517883`
- P1b official unscaled MSE: `466.1923828125`
- P1b official unscaled MAE: `15.501420974731445`

## 9. Prediction Comparison
- aligned prediction correlation: `0.573463457072815`
- aligned prediction mean absolute difference: `0.19655349850654602`
- aligned target mean absolute difference: `8.024922237837018e-08`

## 10. ADAPTER_PARITY
ADAPTER_PARITY=FAIL

Reasons:
- P1b official data_loader uses llm_model.get_input_embeddings()(input_ids); P1 adapter pooled GPT-2 model hidden states.
- P1b official projection is Linear-ReLU-Linear-ReLU-Dropout; P1 adapter uses a different projection block.
- P1b official train/val/test windows follow TaTS overlapping border1s and DataLoader drop_last=True; P1 adapter uses non-overlapping split windows and drop_last=False.
- P1b official forecasting loop mixes outputs with prior_history_avg using prior_weight=0.5; P1 adapter did not use this official prior mix.
- P1b official decoder input includes label_len target history and text decoder channels; P1 adapter used a simplified zero decoder input.

Conclusion: the current P1 adapter is useful as an engineering-chain check, but it is not an exact implementation of the official TaTS M1 baseline.
