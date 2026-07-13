# CEN-TaTS 第四轮实验报告：
基于事件因果关联、双向验证与自动提示优化的文本配对时间序列预测

## 1. 本轮实验目标

本轮废弃弱 raw-text concat，把旧 TF-IDF/SVD 仅保留为 legacy_debugging_baseline。代码迁移了官方 TaTS 仓库，并在不修改 PatchTST/iTransformer backbone 的前提下，只改变输入 TaTS 文本编码器的辅助文本。

## 2. 核心结论摘要

本轮状态：smoke_only: GPT2 tokenizer/config available; pretrained GPT2 weights download timed out, so random input embeddings were used and are not claimed as formal pretrained TaTS. 因此主数值表是可复现实验 smoke，不是正式 pretrained TaTS 论文主结果。PatchTST 官方 channel-independent 行为使文本通道对目标通道影响极弱或为零，这是本轮最重要的工程诊断。

## 3. 当前代码审计

审计文件见 `reports/v5/codebase_audit.md`。旧 TF-IDF/SVD、Ridge、关键词事件、固定 lag、简单 pre/post verifier 与伪 APO 均标记为 legacy，不进入正式 v5 主表。

## 4. 参考工作与代码迁移

`third_party/UPSTREAMS.md` 记录了 TaTS commit、许可证与使用文件。Augur 和 APO 均未声称使用官方代码，而是 paper-aligned reimplementation。

## 5. 数据集

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

## 6. 方法总览

流程包括文本事件抽取、train_core ontology、Augur-style event-target lagged association graph、历史序列反推事件、lag-aware verifier、固定结构叙事、TaTS tokenizer/embedding/projection/backbone 接入和离线 APO prompt beam search。

## 7. 事件—目标关联图

| dataset   |   candidate_edges |   accepted_edges |   event_coverage |   avg_support |   direction_stability |   permutation_pass_rate |
|:----------|------------------:|-----------------:|-----------------:|--------------:|----------------------:|------------------------:|
| Energy    |                10 |                0 |               10 |          97.2 |              0.624155 |                     0.2 |

## 8. 双向 verifier

| dataset   |   threshold |   AUROC |   AUPRC |   F1 |   correct_event_mean_score |   wrong_event_mean_score |   wrong_keep_rate |   wrong_delete_rate |
|:----------|------------:|--------:|--------:|-----:|---------------------------:|-------------------------:|------------------:|--------------------:|
| Energy    |    0.331625 |       1 |       1 |    1 |                   0.336905 |                0.0857036 |                 0 |                   0 |

## 9. APO

| dataset   |   score |   token_cost | prompt_hash      |   round |   candidate_count |   llm_calls |   api_retries |
|:----------|--------:|-------------:|:-----------------|--------:|------------------:|------------:|--------------:|
| Energy    |    1.96 |           86 | 03c8d59b0e30b2ee |       0 |                 3 |           0 |             0 |
| Energy    |    2.2  |          116 | 4f9e1b58a94ad606 |       1 |                 9 |           0 |             0 |
| Energy    |    2.44 |          146 | 21aa67e77f8866bd |       2 |                 9 |           0 |             0 |

## 10. 实验配置

{
  "data": {
    "datasets": [
      "Energy"
    ],
    "candidate_datasets": [
      "Energy",
      "Health_US",
      "Health_AFR"
    ],
    "history_length": 24,
    "label_len": 12,
    "horizons": [
      3
    ],
    "target": "OT"
  },
  "events": {
    "min_type_support": 3
  },
  "causal": {
    "max_lag": 7,
    "min_support": 5
  },
  "verifier": {
    "weights": {
      "evidence": 0.35,
      "causal": 0.35,
      "cycle": 0.2,
      "lag_fit": 0.1
    }
  },
  "apo": {
    "beam_width": 3,
    "candidates_per_parent": 3,
    "max_depth": 3,
    "minibatch_size": 16,
    "stage_a_top_k": 5,
    "stage_b_top_k": 2,
    "weights": {
      "forecast": 1.0,
      "event": 0.2,
      "cycle": 0.2,
      "grounding": 0.2,
      "schema": 0.2,
      "token": 0.01
    }
  },
  "tats": {
    "llm_model": "openai-community/gpt2",
    "lm_mode": "random_init_input_embedding_only_pretrained_weights_timeout",
    "max_token_length": 256,
    "pooling": "avg",
    "embedding_seed": 2026,
    "embedding_batch_size": 64
  },
  "training": {
    "backbone": "PatchTST",
    "seeds": [
      2024,
      2025,
      2026
    ],
    "ablation_seeds": [
      2024
    ],
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "text_emb_dim": 12,
    "d_model": 32,
    "n_heads": 4,
    "e_layers": 1,
    "d_ff": 64,
    "dropout": 0.1,
    "label_len": 12,
    "use_cuda": false,
    "max_windows_by_split": {
      "train_core": 192,
      "prompt_dev": 48,
      "model_val": 64,
      "test": 96
    }
  },
  "llm": {
    "extractor_model": "",
    "causal_teacher_model": "",
    "inverse_event_model": "",
    "verifier_model": "",
    "apo_critic_model": "",
    "apo_editor_model": "",
    "evaluator_model": ""
  }
}

## 11. 主实验结果

| Method                     |   ('MSE', 'mean') |   ('MSE', 'std') |   ('MAE', 'mean') |   ('MAE', 'std') |   ('RMSE', 'mean') |   ('RMSE', 'std') |   ('NMSE', 'mean') |   ('NMSE', 'std') |   ('sMAPE', 'mean') |   ('sMAPE', 'std') |   ('Directional Accuracy', 'mean') |   ('Directional Accuracy', 'std') |   ('Trend Macro-F1', 'mean') |   ('Trend Macro-F1', 'std') |
|:---------------------------|------------------:|-----------------:|------------------:|-----------------:|-------------------:|------------------:|-------------------:|------------------:|--------------------:|-------------------:|-----------------------------------:|----------------------------------:|-----------------------------:|----------------------------:|
| M0_Numerical-only          |          0.119209 |        0.042991  |          0.250587 |        0.0496153 |           0.341422 |         0.0629326 |           0.156626 |         0.0564851 |            0.248276 |          0.0261604 |                           0.496528 |                        0.021684   |                     0.313248 |                  0.0129028  |
| M1_TaTS-Raw-Text           |          0.118362 |        0.0427959 |          0.249554 |        0.0505924 |           0.340233 |         0.0624928 |           0.155514 |         0.0562287 |            0.245465 |          0.0359419 |                           0.493056 |                        0.00601407 |                     0.305554 |                  0.00866303 |
| M2_Event-TaTS              |          0.118362 |        0.0427959 |          0.249554 |        0.0505924 |           0.340233 |         0.0624928 |           0.155514 |         0.0562287 |            0.245465 |          0.0359419 |                           0.493056 |                        0.00601407 |                     0.305554 |                  0.00866303 |
| M3_Causal-Event-TaTS       |          0.118362 |        0.0427959 |          0.249554 |        0.0505924 |           0.340233 |         0.0624928 |           0.155514 |         0.0562287 |            0.245465 |          0.0359419 |                           0.493056 |                        0.00601407 |                     0.305554 |                  0.00866303 |
| M4_Cycle-Verified-CEN-TaTS |          0.118362 |        0.0427959 |          0.249554 |        0.0505924 |           0.340233 |         0.0624928 |           0.155514 |         0.0562287 |            0.245465 |          0.0359419 |                           0.493056 |                        0.00601407 |                     0.305554 |                  0.00866303 |
| M5_APO-CEN-TaTS            |          0.118362 |        0.0427959 |          0.249554 |        0.0505924 |           0.340233 |         0.0624928 |           0.155514 |         0.0562287 |            0.245465 |          0.0359419 |                           0.493056 |                        0.00601407 |                     0.305554 |                  0.00866303 |

## 12. 事件窗口、高波动和转折点结果

|       MSE |      MAE |     RMSE |     NMSE |    sMAPE |   Directional Accuracy |   Trend Macro-F1 |   model_val_MSE | method                     | backbone   |   seed | run_id                                                       |   runtime_sec |   n_test_windows | prediction_path                                                                                                                                                          | Dataset   |   Horizon | Method                     | Structured event   | Causal graph   | Inverse event   | Lag-aware verifier   | APO   | Backbone   | TaTS LM status                                              | official_pretrained_tats_status   | split_hash       |
|----------:|---------:|---------:|---------:|---------:|-----------------------:|-----------------:|----------------:|:---------------------------|:-----------|-------:|:-------------------------------------------------------------|--------------:|-----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|----------:|:---------------------------|:-------------------|:---------------|:----------------|:---------------------|:------|:-----------|:------------------------------------------------------------|:----------------------------------|:-----------------|
| 0.116879  | 0.249528 | 0.341876 | 0.153565 | 0.24741  |               0.520833 |         0.320775 |       0.0437243 | M0_Numerical-only          | PatchTST   |   2024 | Energy_M0_Numerical-only_PatchTST_H3_S2024_754ba2fd          |     0.0775046 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M0_Numerical-only_PatchTST_H3_S2024_754ba2fd\predictions.npz          | Energy    |         3 | M0_Numerical-only          | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0774302 | 0.20151  | 0.278263 | 0.101734 | 0.22256  |               0.479167 |         0.29835  |       0.0375523 | M0_Numerical-only          | PatchTST   |   2025 | Energy_M0_Numerical-only_PatchTST_H3_S2025_215a0a3b          |     0.0668533 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M0_Numerical-only_PatchTST_H3_S2025_215a0a3b\predictions.npz          | Energy    |         3 | M0_Numerical-only          | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.163318  | 0.300723 | 0.404126 | 0.21458  | 0.274859 |               0.489583 |         0.32062  |       0.0974905 | M0_Numerical-only          | PatchTST   |   2026 | Energy_M0_Numerical-only_PatchTST_H3_S2026_e4487c68          |     0.070513  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M0_Numerical-only_PatchTST_H3_S2026_e4487c68\predictions.npz          | Energy    |         3 | M0_Numerical-only          | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218  | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | M1_TaTS-Raw-Text           | PatchTST   |   2024 | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2024_88e66259           |     0.108514  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2024_88e66259\predictions.npz           | Energy    |         3 | M1_TaTS-Raw-Text           | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0777893 | 0.200892 | 0.278907 | 0.102206 | 0.21201  |               0.5      |         0.314585 |       0.040779  | M1_TaTS-Raw-Text           | PatchTST   |   2025 | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2025_63051732           |     0.116173  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2025_63051732\predictions.npz           | Energy    |         3 | M1_TaTS-Raw-Text           | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.16308   | 0.301878 | 0.403831 | 0.214267 | 0.283462 |               0.489583 |         0.304762 |       0.0828012 | M1_TaTS-Raw-Text           | PatchTST   |   2026 | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2026_db9e0d6b           |     0.117202  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2026_db9e0d6b\predictions.npz           | Energy    |         3 | M1_TaTS-Raw-Text           | False              | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218  | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | M2_Event-TaTS              | PatchTST   |   2024 | Energy_M2_Event-TaTS_PatchTST_H3_S2024_5cc6dba5              |     0.146593  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M2_Event-TaTS_PatchTST_H3_S2024_5cc6dba5\predictions.npz              | Energy    |         3 | M2_Event-TaTS              | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0777893 | 0.200892 | 0.278907 | 0.102206 | 0.21201  |               0.5      |         0.314585 |       0.040779  | M2_Event-TaTS              | PatchTST   |   2025 | Energy_M2_Event-TaTS_PatchTST_H3_S2025_85d2c0ee              |     0.111855  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M2_Event-TaTS_PatchTST_H3_S2025_85d2c0ee\predictions.npz              | Energy    |         3 | M2_Event-TaTS              | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.16308   | 0.301878 | 0.403831 | 0.214267 | 0.283462 |               0.489583 |         0.304762 |       0.0828012 | M2_Event-TaTS              | PatchTST   |   2026 | Energy_M2_Event-TaTS_PatchTST_H3_S2026_92b34478              |     0.118501  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M2_Event-TaTS_PatchTST_H3_S2026_92b34478\predictions.npz              | Energy    |         3 | M2_Event-TaTS              | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218  | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | M3_Causal-Event-TaTS       | PatchTST   |   2024 | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2024_f8829a97       |     0.125402  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2024_f8829a97\predictions.npz       | Energy    |         3 | M3_Causal-Event-TaTS       | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0777893 | 0.200892 | 0.278907 | 0.102206 | 0.21201  |               0.5      |         0.314585 |       0.040779  | M3_Causal-Event-TaTS       | PatchTST   |   2025 | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2025_c76f05fa       |     0.125796  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2025_c76f05fa\predictions.npz       | Energy    |         3 | M3_Causal-Event-TaTS       | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.16308   | 0.301878 | 0.403831 | 0.214267 | 0.283462 |               0.489583 |         0.304762 |       0.0828012 | M3_Causal-Event-TaTS       | PatchTST   |   2026 | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2026_cdf8a155       |     0.116262  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2026_cdf8a155\predictions.npz       | Energy    |         3 | M3_Causal-Event-TaTS       | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218  | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | M4_Cycle-Verified-CEN-TaTS | PatchTST   |   2024 | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2024_51b9d758 |     0.115037  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2024_51b9d758\predictions.npz | Energy    |         3 | M4_Cycle-Verified-CEN-TaTS | True               | True           | True            | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0777893 | 0.200892 | 0.278907 | 0.102206 | 0.21201  |               0.5      |         0.314585 |       0.040779  | M4_Cycle-Verified-CEN-TaTS | PatchTST   |   2025 | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2025_d532c973 |     0.12224   |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2025_d532c973\predictions.npz | Energy    |         3 | M4_Cycle-Verified-CEN-TaTS | True               | True           | True            | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.16308   | 0.301878 | 0.403831 | 0.214267 | 0.283462 |               0.489583 |         0.304762 |       0.0828012 | M4_Cycle-Verified-CEN-TaTS | PatchTST   |   2026 | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2026_66617b03 |     0.11857   |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2026_66617b03\predictions.npz | Energy    |         3 | M4_Cycle-Verified-CEN-TaTS | True               | True           | True            | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218  | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | M5_APO-CEN-TaTS            | PatchTST   |   2024 | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2024_22acb996            |     0.129716  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2024_22acb996\predictions.npz            | Energy    |         3 | M5_APO-CEN-TaTS            | True               | True           | True            | True                 | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.0777893 | 0.200892 | 0.278907 | 0.102206 | 0.21201  |               0.5      |         0.314585 |       0.040779  | M5_APO-CEN-TaTS            | PatchTST   |   2025 | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2025_0b9f166c            |     0.100515  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2025_0b9f166c\predictions.npz            | Energy    |         3 | M5_APO-CEN-TaTS            | True               | True           | True            | True                 | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.16308   | 0.301878 | 0.403831 | 0.214267 | 0.283462 |               0.489583 |         0.304762 |       0.0828012 | M5_APO-CEN-TaTS            | PatchTST   |   2026 | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2026_0011a326            |     0.116572  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2026_0011a326\predictions.npz            | Energy    |         3 | M5_APO-CEN-TaTS            | True               | True           | True            | True                 | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |

## 13. 因果图消融

|      MSE |      MAE |     RMSE |     NMSE |    sMAPE |   Directional Accuracy |   Trend Macro-F1 |   model_val_MSE | method                 | backbone   |   seed | run_id                                                   |   runtime_sec |   n_test_windows | prediction_path                                                                                                                                                      | Dataset   |   Horizon | Method                 | Structured event   | Causal graph   | Inverse event   | Lag-aware verifier   | APO   | Backbone   | TaTS LM status                                              | official_pretrained_tats_status   | split_hash       |
|---------:|---------:|---------:|---------:|---------:|-----------------------:|-----------------:|----------------:|:-----------------------|:-----------|-------:|:---------------------------------------------------------|--------------:|-----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|----------:|:-----------------------|:-------------------|:---------------|:----------------|:---------------------|:------|:-----------|:------------------------------------------------------------|:----------------------------------|:-----------------|
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A4_Random-causal-graph | PatchTST   |   2024 | Energy_A4_Random-causal-graph_PatchTST_H3_S2024_09dafdb7 |      0.112519 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A4_Random-causal-graph_PatchTST_H3_S2024_09dafdb7\predictions.npz | Energy    |         3 | A4_Random-causal-graph | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A5_Teacher-only-graph  | PatchTST   |   2024 | Energy_A5_Teacher-only-graph_PatchTST_H3_S2024_4b0e0f2a  |      0.117513 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A5_Teacher-only-graph_PatchTST_H3_S2024_4b0e0f2a\predictions.npz  | Energy    |         3 | A5_Teacher-only-graph  | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |

## 14. verifier 消融

|      MSE |      MAE |     RMSE |     NMSE |    sMAPE |   Directional Accuracy |   Trend Macro-F1 |   model_val_MSE | method                             | backbone   |   seed | run_id                                                               |   runtime_sec |   n_test_windows | prediction_path                                                                                                                                                                  | Dataset   |   Horizon | Method                             | Structured event   | Causal graph   | Inverse event   | Lag-aware verifier   | APO   | Backbone   | TaTS LM status                                              | official_pretrained_tats_status   | split_hash       |
|---------:|---------:|---------:|---------:|---------:|-----------------------:|-----------------:|----------------:|:-----------------------------------|:-----------|-------:|:---------------------------------------------------------------------|--------------:|-----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|----------:|:-----------------------------------|:-------------------|:---------------|:----------------|:---------------------|:------|:-----------|:------------------------------------------------------------|:----------------------------------|:-----------------|
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A6_Verifier-without-inverse-event  | PatchTST   |   2024 | Energy_A6_Verifier-without-inverse-event_PatchTST_H3_S2024_65df633f  |      0.113221 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A6_Verifier-without-inverse-event_PatchTST_H3_S2024_65df633f\predictions.npz  | Energy    |         3 | A6_Verifier-without-inverse-event  | True               | False          | True            | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A7_Verifier-without-lag-aware-mask | PatchTST   |   2024 | Energy_A7_Verifier-without-lag-aware-mask_PatchTST_H3_S2024_7f0397cc |      0.144805 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A7_Verifier-without-lag-aware-mask_PatchTST_H3_S2024_7f0397cc\predictions.npz | Energy    |         3 | A7_Verifier-without-lag-aware-mask | True               | False          | False           | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A8_Random-verifier                 | PatchTST   |   2024 | Energy_A8_Random-verifier_PatchTST_H3_S2024_049f1759                 |      0.111044 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A8_Random-verifier_PatchTST_H3_S2024_049f1759\predictions.npz                 | Energy    |         3 | A8_Random-verifier                 | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |

## 15. APO 消融

|      MSE |      MAE |     RMSE |     NMSE |    sMAPE |   Directional Accuracy |   Trend Macro-F1 |   model_val_MSE | method                            | backbone   |   seed | run_id                                                              |   runtime_sec |   n_test_windows | prediction_path                                                                                                                                                                 | Dataset   |   Horizon | Method                            | Structured event   | Causal graph   | Inverse event   | Lag-aware verifier   | APO   | Backbone   | TaTS LM status                                              | official_pretrained_tats_status   | split_hash       |
|---------:|---------:|---------:|---------:|---------:|-----------------------:|-----------------:|----------------:|:----------------------------------|:-----------|-------:|:--------------------------------------------------------------------|--------------:|-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|----------:|:----------------------------------|:-------------------|:---------------|:----------------|:---------------------|:------|:-----------|:------------------------------------------------------------|:----------------------------------|:-----------------|
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A10_APO-without-forecast-feedback | PatchTST   |   2024 | Energy_A10_APO-without-forecast-feedback_PatchTST_H3_S2024_a3cf9cdf |      0.12666  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A10_APO-without-forecast-feedback_PatchTST_H3_S2024_a3cf9cdf\predictions.npz | Energy    |         3 | A10_APO-without-forecast-feedback | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A11_APO-without-event-feedback    | PatchTST   |   2024 | Energy_A11_APO-without-event-feedback_PatchTST_H3_S2024_38c70b46    |      0.12255  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A11_APO-without-event-feedback_PatchTST_H3_S2024_38c70b46\predictions.npz    | Energy    |         3 | A11_APO-without-event-feedback    | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A12_Manual-vs-APO-prompt          | PatchTST   |   2024 | Energy_A12_Manual-vs-APO-prompt_PatchTST_H3_S2024_a2b37e96          |      0.124522 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A12_Manual-vs-APO-prompt_PatchTST_H3_S2024_a2b37e96\predictions.npz          | Energy    |         3 | A12_Manual-vs-APO-prompt          | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |

## 16. 鲁棒性

|      MSE |      MAE |     RMSE |     NMSE |    sMAPE |   Directional Accuracy |   Trend Macro-F1 |   model_val_MSE | method                             | backbone   |   seed | run_id                                                               |   runtime_sec |   n_test_windows | prediction_path                                                                                                                                                                  | Dataset   |   Horizon | Method                             | Structured event   | Causal graph   | Inverse event   | Lag-aware verifier   | APO   | Backbone   | TaTS LM status                                              | official_pretrained_tats_status   | split_hash       |
|---------:|---------:|---------:|---------:|---------:|-----------------------:|-----------------:|----------------:|:-----------------------------------|:-----------|-------:|:---------------------------------------------------------------------|--------------:|-----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|----------:|:-----------------------------------|:-------------------|:---------------|:----------------|:---------------------|:------|:-----------|:------------------------------------------------------------|:----------------------------------|:-----------------|
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A1_Zero-text-TaTS                  | PatchTST   |   2024 | Energy_A1_Zero-text-TaTS_PatchTST_H3_S2024_8aa6cb61                  |      0.106076 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A1_Zero-text-TaTS_PatchTST_H3_S2024_8aa6cb61\predictions.npz                  | Energy    |         3 | A1_Zero-text-TaTS                  | True               | False          | False           | False                | False | PatchTST   | zero_text_control                                           | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A2_Shuffled-raw-text               | PatchTST   |   2024 | Energy_A2_Shuffled-raw-text_PatchTST_H3_S2024_66a958e8               |      0.114142 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A2_Shuffled-raw-text_PatchTST_H3_S2024_66a958e8\predictions.npz               | Energy    |         3 | A2_Shuffled-raw-text               | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A3_Shuffled-event-time             | PatchTST   |   2024 | Energy_A3_Shuffled-event-time_PatchTST_H3_S2024_14c1a9ef             |      0.119541 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A3_Shuffled-event-time_PatchTST_H3_S2024_14c1a9ef\predictions.npz             | Energy    |         3 | A3_Shuffled-event-time             | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A4_Random-causal-graph             | PatchTST   |   2024 | Energy_A4_Random-causal-graph_PatchTST_H3_S2024_09dafdb7             |      0.112519 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A4_Random-causal-graph_PatchTST_H3_S2024_09dafdb7\predictions.npz             | Energy    |         3 | A4_Random-causal-graph             | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A5_Teacher-only-graph              | PatchTST   |   2024 | Energy_A5_Teacher-only-graph_PatchTST_H3_S2024_4b0e0f2a              |      0.117513 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A5_Teacher-only-graph_PatchTST_H3_S2024_4b0e0f2a\predictions.npz              | Energy    |         3 | A5_Teacher-only-graph              | True               | True           | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A6_Verifier-without-inverse-event  | PatchTST   |   2024 | Energy_A6_Verifier-without-inverse-event_PatchTST_H3_S2024_65df633f  |      0.113221 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A6_Verifier-without-inverse-event_PatchTST_H3_S2024_65df633f\predictions.npz  | Energy    |         3 | A6_Verifier-without-inverse-event  | True               | False          | True            | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A7_Verifier-without-lag-aware-mask | PatchTST   |   2024 | Energy_A7_Verifier-without-lag-aware-mask_PatchTST_H3_S2024_7f0397cc |      0.144805 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A7_Verifier-without-lag-aware-mask_PatchTST_H3_S2024_7f0397cc\predictions.npz | Energy    |         3 | A7_Verifier-without-lag-aware-mask | True               | False          | False           | True                 | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A8_Random-verifier                 | PatchTST   |   2024 | Energy_A8_Random-verifier_PatchTST_H3_S2024_049f1759                 |      0.111044 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A8_Random-verifier_PatchTST_H3_S2024_049f1759\predictions.npz                 | Energy    |         3 | A8_Random-verifier                 | True               | False          | False           | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A9_Shuffled-inverse-event          | PatchTST   |   2024 | Energy_A9_Shuffled-inverse-event_PatchTST_H3_S2024_88b22ad4          |      0.116791 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A9_Shuffled-inverse-event_PatchTST_H3_S2024_88b22ad4\predictions.npz          | Energy    |         3 | A9_Shuffled-inverse-event          | True               | False          | True            | False                | False | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A10_APO-without-forecast-feedback  | PatchTST   |   2024 | Energy_A10_APO-without-forecast-feedback_PatchTST_H3_S2024_a3cf9cdf  |      0.12666  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A10_APO-without-forecast-feedback_PatchTST_H3_S2024_a3cf9cdf\predictions.npz  | Energy    |         3 | A10_APO-without-forecast-feedback  | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A11_APO-without-event-feedback     | PatchTST   |   2024 | Energy_A11_APO-without-event-feedback_PatchTST_H3_S2024_38c70b46     |      0.12255  |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A11_APO-without-event-feedback_PatchTST_H3_S2024_38c70b46\predictions.npz     | Energy    |         3 | A11_APO-without-event-feedback     | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |
| 0.114218 | 0.245892 | 0.337961 | 0.150068 | 0.240922 |               0.489583 |         0.297314 |       0.0443903 | A12_Manual-vs-APO-prompt           | PatchTST   |   2024 | Energy_A12_Manual-vs-APO-prompt_PatchTST_H3_S2024_a2b37e96           |      0.124522 |               96 | C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives\results\v5\runs\Energy_A12_Manual-vs-APO-prompt_PatchTST_H3_S2024_a2b37e96\predictions.npz           | Energy    |         3 | A12_Manual-vs-APO-prompt           | True               | False          | False           | False                | True  | PatchTST   | random_init_input_embedding_only_pretrained_weights_timeout | failed_timeout_weights_download   | 432fc5f4b01772f4 |

## 17. 统计显著性

| comparison                                         |   seed |    mean_diff |      ci_low |    ci_high |   n_boot |   p_value |   n_perm |
|:---------------------------------------------------|-------:|-------------:|------------:|-----------:|---------:|----------:|---------:|
| M1_TaTS-Raw-Text vs M0_Numerical-only              |   2024 |  0.00266155  | -0.0011688  | 0.00648475 |     1000 |  0.166833 |     1000 |
| M1_TaTS-Raw-Text vs M0_Numerical-only              |   2025 | -0.000359094 | -0.00818292 | 0.00751449 |     1000 |  0.923077 |     1000 |
| M1_TaTS-Raw-Text vs M0_Numerical-only              |   2026 |  0.000238005 | -0.0121813  | 0.0125494  |     1000 |  0.962038 |     1000 |
| M2_Event-TaTS vs M1_TaTS-Raw-Text                  |   2024 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M2_Event-TaTS vs M1_TaTS-Raw-Text                  |   2025 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M2_Event-TaTS vs M1_TaTS-Raw-Text                  |   2026 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M3_Causal-Event-TaTS vs M2_Event-TaTS              |   2024 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M3_Causal-Event-TaTS vs M2_Event-TaTS              |   2025 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M3_Causal-Event-TaTS vs M2_Event-TaTS              |   2026 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M4_Cycle-Verified-CEN-TaTS vs M3_Causal-Event-TaTS |   2024 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M4_Cycle-Verified-CEN-TaTS vs M3_Causal-Event-TaTS |   2025 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M4_Cycle-Verified-CEN-TaTS vs M3_Causal-Event-TaTS |   2026 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M4_Cycle-Verified-CEN-TaTS      |   2024 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M4_Cycle-Verified-CEN-TaTS      |   2025 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M4_Cycle-Verified-CEN-TaTS      |   2026 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M1_TaTS-Raw-Text                |   2024 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M1_TaTS-Raw-Text                |   2025 |  0           |  0          | 0          |     1000 |  1        |     1000 |
| M5_APO-CEN-TaTS vs M1_TaTS-Raw-Text                |   2026 |  0           |  0          | 0          |     1000 |  1        |     1000 |

## 18. 效率

| Method                             |   runtime_sec | run_id                                                               |
|:-----------------------------------|--------------:|:---------------------------------------------------------------------|
| M0_Numerical-only                  |     0.0775046 | Energy_M0_Numerical-only_PatchTST_H3_S2024_754ba2fd                  |
| M0_Numerical-only                  |     0.0668533 | Energy_M0_Numerical-only_PatchTST_H3_S2025_215a0a3b                  |
| M0_Numerical-only                  |     0.070513  | Energy_M0_Numerical-only_PatchTST_H3_S2026_e4487c68                  |
| M1_TaTS-Raw-Text                   |     0.108514  | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2024_88e66259                   |
| M1_TaTS-Raw-Text                   |     0.116173  | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2025_63051732                   |
| M1_TaTS-Raw-Text                   |     0.117202  | Energy_M1_TaTS-Raw-Text_PatchTST_H3_S2026_db9e0d6b                   |
| M2_Event-TaTS                      |     0.146593  | Energy_M2_Event-TaTS_PatchTST_H3_S2024_5cc6dba5                      |
| M2_Event-TaTS                      |     0.111855  | Energy_M2_Event-TaTS_PatchTST_H3_S2025_85d2c0ee                      |
| M2_Event-TaTS                      |     0.118501  | Energy_M2_Event-TaTS_PatchTST_H3_S2026_92b34478                      |
| M3_Causal-Event-TaTS               |     0.125402  | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2024_f8829a97               |
| M3_Causal-Event-TaTS               |     0.125796  | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2025_c76f05fa               |
| M3_Causal-Event-TaTS               |     0.116262  | Energy_M3_Causal-Event-TaTS_PatchTST_H3_S2026_cdf8a155               |
| M4_Cycle-Verified-CEN-TaTS         |     0.115037  | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2024_51b9d758         |
| M4_Cycle-Verified-CEN-TaTS         |     0.12224   | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2025_d532c973         |
| M4_Cycle-Verified-CEN-TaTS         |     0.11857   | Energy_M4_Cycle-Verified-CEN-TaTS_PatchTST_H3_S2026_66617b03         |
| M5_APO-CEN-TaTS                    |     0.129716  | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2024_22acb996                    |
| M5_APO-CEN-TaTS                    |     0.100515  | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2025_0b9f166c                    |
| M5_APO-CEN-TaTS                    |     0.116572  | Energy_M5_APO-CEN-TaTS_PatchTST_H3_S2026_0011a326                    |
| A1_Zero-text-TaTS                  |     0.106076  | Energy_A1_Zero-text-TaTS_PatchTST_H3_S2024_8aa6cb61                  |
| A2_Shuffled-raw-text               |     0.114142  | Energy_A2_Shuffled-raw-text_PatchTST_H3_S2024_66a958e8               |
| A3_Shuffled-event-time             |     0.119541  | Energy_A3_Shuffled-event-time_PatchTST_H3_S2024_14c1a9ef             |
| A4_Random-causal-graph             |     0.112519  | Energy_A4_Random-causal-graph_PatchTST_H3_S2024_09dafdb7             |
| A5_Teacher-only-graph              |     0.117513  | Energy_A5_Teacher-only-graph_PatchTST_H3_S2024_4b0e0f2a              |
| A6_Verifier-without-inverse-event  |     0.113221  | Energy_A6_Verifier-without-inverse-event_PatchTST_H3_S2024_65df633f  |
| A7_Verifier-without-lag-aware-mask |     0.144805  | Energy_A7_Verifier-without-lag-aware-mask_PatchTST_H3_S2024_7f0397cc |
| A8_Random-verifier                 |     0.111044  | Energy_A8_Random-verifier_PatchTST_H3_S2024_049f1759                 |
| A9_Shuffled-inverse-event          |     0.116791  | Energy_A9_Shuffled-inverse-event_PatchTST_H3_S2024_88b22ad4          |
| A10_APO-without-forecast-feedback  |     0.12666   | Energy_A10_APO-without-forecast-feedback_PatchTST_H3_S2024_a3cf9cdf  |
| A11_APO-without-event-feedback     |     0.12255   | Energy_A11_APO-without-event-feedback_PatchTST_H3_S2024_38c70b46     |
| A12_Manual-vs-APO-prompt           |     0.124522  | Energy_A12_Manual-vs-APO-prompt_PatchTST_H3_S2024_a2b37e96           |

## 19. 成功案例

```json
{
  "dataset": "Energy",
  "time_index": 0,
  "items": [
    {
      "event": {
        "event_id": "d5f3ad4084c934b7",
        "report_time": "1993-04-05T00:00:00",
        "event_time": "1993-04-05T00:00:00",
        "actor": "Energy",
        "action": "supply increase",
        "object": "OT",
        "event_type": "supply_increase",
        "affected_target": "OT",
        "expected_direction": "negative",
        "intensity": "weak",
        "candidate_lag_min": 1,
        "candidate_lag_max": 7,
        "expected_duration": 7,
        "factuality": "forecast",
        "source": "Time-MMD aligned text",
        "evidence_span": "The global oil market is experiencing changes due to shale gas production in the United States and rising oil production in Iraq, which may lead to lower oil prices.",
        "extraction_confidence": 0.8,
        "canonical_event_id": "7d61b615343705c6",
        "independent_source_count": 1,
        "duplicate_count": 0,
        "time_index": 0,
        "domain": "Energy",
        "raw_event_type": "supply_increase"
      },
      "verifier": {
        "event_id": "d5f3ad4084c934b7",
        "verified": false,
        "verification_score": 0.252,
        "evidence_score": 0.72,
        "causal_relevance": 0.0,
        "cycle_consistency": 0.7175,
        "lag_fit": 0.0,
        "observable": false,
        "corrected_direction": "negative",
        "corrected_lag_min": 1,
        "corrected_lag_max": 7,
        "effect_status": "not_started",
        "decision_reason": [
          "factuality_forecast",
          "no_accepted_graph_edge"
        ],
        "supporting_graph_edge": null,
        "dataset": "Energy",
        "time_index": 0,
        "event_type": "supply_increase"
      },
      "narrative": "Event fact: A supply_increase event was reported on 1993-04-05T00:00:00. Target relevance: The event has no accepted causal-association edge. Expected direction: negative. Estimated lag: 1 to 7 time steps. Expected duration: 7 time steps. Textual evidence: The global oil market is experiencing changes due to shale gas production in the United States and rising oil production in Iraq, which may lead to lower oil prices.. Numerical consistency: not_started; observable=False. Verification confidence: 0.2520."
    }
  ]
}
```

## 20. 失败案例

失败案例包括：pretrained GPT2 权重下载超时；PatchTST 官方 channel-independent 导致文本通道难以影响目标通道；offline teacher 不能替代真实 LLM teacher；silver proxy 不能称作人工标注。

## 21. 假设检验结论

H1-H7 在本轮均只能标记为 smoke_only / 部分支持或不支持；正式结论必须等待 pretrained TaTS 完成下载和更大预算复现实验。

## 22. 局限

观测数据无法证明真实因果；ontology 依赖领域；LLM 抽取未大规模运行；反向事件多解；APO 可能 prompt_dev 过拟合；文本时间戳可能不精确；人工标注不足；计算和 API 成本限制了正式矩阵。

## 23. 下一步计划

第一，完成 GPT2/BERT 权重下载后复跑正式 TaTS。第二，在不改 backbone 的前提下优先用 iTransformer 检验文本通道是否实际生效。第三，补人工/silver annotation 和更严格事件窗口。

## 24. 复现命令

```powershell
python scripts/v5/00_audit_repo.py --config configs/v5/cen_tats_v5.yaml
python scripts/v5/run_all.py --config configs/v5/cen_tats_v5.yaml
```

## 25. Git 提交记录

84a12a3 Add third-stage event fusion report
0620436 sync_reports
c464c11 cen_2
821bd5b report: add detailed Chinese experiment report
87c5141 测试
826bc7a experiment: add granger gametime mmtsflib and llm probes
c927e8b experiment: extend ablation and robustness runs
0f3eed5 report: add generated experiment report and next-step plan
75dd775 experiment: add main experiment runner and configs
c8fecc6 verifier: implement event consistency and robustness tests