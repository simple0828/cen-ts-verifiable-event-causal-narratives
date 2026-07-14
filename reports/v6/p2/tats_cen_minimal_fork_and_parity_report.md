# CEN-TaTS v6 P2 Minimal Fork and Parity Report

## 1. 执行摘要
P2_STATUS=PASS；P2_PARITY=PASS。`tats_cen` 已从官方 TaTS commit `a053503674c61c54d101d01d47c9d680288a7c9a` 创建最小分叉，并通过 raw 模式等价性验收。

## 2. P2_STATUS
`PASS`

## 3. P2_PARITY
`PASS`

## 4. Git 分支与 commit
- branch: `feature/tats-cen-minimal-fork-v6`
- P2 commit: pending at report generation time; final commit is recorded after commit.

## 5. TaTS 上游 commit
`a053503674c61c54d101d01d47c9d680288a7c9a`

## 6. third_party/TaTS 是否保持 clean
`True`

## 7. tats_cen 创建方式
`scripts/v6/p2/20_create_tats_cen_fork.py` copied upstream git tracked files and wrote SHA256 manifests.

## 8. 文件复制清单
`results/v6/p2/upstream_manifest.json` and `results/v6/p2/tats_cen_initial_manifest.json`.

## 9. 修改边界
`tats_cen/UPSTREAM_BOUNDARY.md` defines allowed and forbidden changes.

## 10. 新增命令行参数
`--text_mode`, `--text_column`, `--llm_path`, `--strict_local_llm`, `--prior_weight`, `--save_root`, `--run_name`, `--prompt_version`, `--event_cache_path`, `--text_variant_manifest`, `--fail_on_missing_text`, `--record_input_hashes`.

## 11. 文本模式接口
Implemented: raw, constant, shuffled. Reserved modes raise `NotImplementedError`: event, causal_event, verified_event, apo_event.

## 12. GPT-2 严格加载
local_files_only=True; random_fallback=False.

## 13. data_loader 最小修改
Only text column selection, checks, manifests, and official-equivalent missing-text fill were added. Tokenization, `get_input_embeddings`, and mask-average pooling are preserved.

## 14. projection 和 iTransformer 保持情况
iTransformer unchanged=True; projection unchanged=True; pooling unchanged=True.

## 15. prior_weight 参数化
Protocol A uses `prior_weight=0.5`; Protocol B smoke uses `prior_weight=0.0` and completed train/test.

## 16. variant builder
Generated raw/constant/shuffled CSV manifests and split-local shuffle mapping.

## 17. raw 数据一致性
raw_variant_numeric_identity=True. Raw parity run uses the official CSV directly to avoid pandas rewrite effects.

## 18. P1b raw parity
P1b scaled MSE=0.26544812321662903; tats_cen scaled MSE=0.26544812321662903.

## 19. 第一批输入 parity
numeric=True; token=True.

## 20. forward/loss/gradient parity
forward_max_abs_diff=0.0; loss_abs_diff=0.0; gradient_max_abs_diff=0.0.

## 21. 完整训练指标 parity
metric_relative_difference=0.0; prediction_correlation=1.0.

## 22. constant smoke
completed=True; predictions_differ_from_raw=True.

## 23. shuffled smoke
completed=True; predictions_differ_from_raw=True.

## 24. 自动测试
`D:/Miniconda/envs/tats/python.exe -m pytest tests/v6 -q` -> 37 passed.

## 25. 已修改文件
See `reports/v6/p2/tats_cen_upstream_diff.md`.

## 26. 风险与局限
P2 does not run real event extraction, causal graph construction, verifier, APO, or multi-seed experiments. Constant/shuffled are smoke checks, not paper-level conclusions.

## 27. 是否允许进入事件抽取阶段
允许。P2 parity and smoke gates passed.

## 28. 下一阶段建议
Start event extraction only through the text variant interface; keep iTransformer, projection, pooling, decoder, optimizer, and training protocol frozen.
