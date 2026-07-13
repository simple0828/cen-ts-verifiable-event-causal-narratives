# CEN-TaTS v5 Codebase Audit

## Git and Runtime
- **git status**: `On branch experiment/cen-tats-closed-loop-v5
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   .gitignore

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	artifacts/
	configs/v5/
	data/annotations/
	prompt/
	prompts/
	reports/report_4.md
	reports/v5/
	results/
	scripts/v5/
	src/cen_tats/
	tests/v5/
	third_party/

no changes added to commit (use "git add" and/or "git commit -a")`
- **git branch --show-current**: `experiment/cen-tats-closed-loop-v5`
- **git log --oneline -10**: `84a12a3 Add third-stage event fusion report
0620436 sync_reports
c464c11 cen_2
821bd5b report: add detailed Chinese experiment report
87c5141 测试
826bc7a experiment: add granger gametime mmtsflib and llm probes
c927e8b experiment: extend ablation and robustness runs
0f3eed5 report: add generated experiment report and next-step plan
75dd775 experiment: add main experiment runner and configs
c8fecc6 verifier: implement event consistency and robustness tests`
- **git remote -v**: `origin	https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives.git (fetch)
origin	https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives.git (push)`
- **TaTS commit**: `a053503674c61c54d101d01d47c9d680288a7c9a`
- **TaTS remote**: `origin	https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS.git (fetch)
origin	https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS.git (push)`
- **python**: `Python 3.10.0`
- **torch**: `2.13.0+cpu False`

## Required Area Status

| Item | Status | Note |
|---|---:|---|
| TF-IDF/SVD text features | legacy | audited before v5 edits |
| Ridge numerical model | legacy | audited before v5 edits |
| keyword/rule event extraction | legacy | audited before v5 edits |
| fixed 1-3 lag | legacy | audited before v5 edits |
| simple pre/post verifier | legacy | audited before v5 edits |
| test-set graph or threshold tuning | leakage_risk_not_observed_in_v5 | audited before v5 edits |
| pseudo APO without optimization | legacy | audited before v5 edits |
| official TaTS clone | completed | audited before v5 edits |
| official GPT2 pretrained weights | failed_timeout | audited before v5 edits |

## Dataset Audit

| dataset     |   timepoints | start      | end        |   numeric_variables | target   |   text_coverage |   effective_text_rows |   forecast_windows |
|:------------|-------------:|:-----------|:-----------|--------------------:|:---------|----------------:|----------------------:|-------------------:|
| Agriculture |          532 | 1980-01-01 | 2024-04-01 |                   3 | OT       |               1 |                   532 |                506 |
| Climate     |         1272 | 2000-01-04 | 2024-05-14 |                   8 | OT       |               1 |                  1272 |               1246 |
| Economy     |          447 | 1987-01-01 | 2024-03-01 |                   3 | OT       |               1 |                   447 |                421 |
| Energy      |         1622 | 1993-04-05 | 2024-04-29 |                   9 | OT       |               1 |                  1622 |               1596 |
| Environment |        15979 | 1980-01-01 | 2023-09-30 |                   3 | OT       |               1 |                 15979 |              15953 |
| Health_AFR  |         1461 | 1996-01-01 | 2024-04-29 |                   1 | OT       |               1 |                  1461 |               1435 |
| Health_US   |         1389 | 1997-09-29 | 2024-05-06 |                  10 | OT       |               1 |                  1389 |               1363 |
| Security    |          309 | 1998-09-01 | 2024-05-01 |                   1 | OT       |               1 |                   309 |                283 |
| SocialGood  |          924 | 1948-01-01 | 2024-12-01 |                   1 | OT       |               1 |                   924 |                898 |
| Traffic     |          651 | 1970-01-01 | 2024-03-01 |                   1 | OT       |               1 |                   651 |                625 |

## Key Finding

The previous `raw_text_embedding` path is TF-IDF/SVD and remains legacy_debugging_baseline only. v5 uses the official TaTS repository for PatchTST/iTransformer backbone imports. The pretrained GPT2 weight download did not finish in this session, so v5 forecasting runs are marked smoke_only with GPT2 tokenizer/config and random input embeddings.