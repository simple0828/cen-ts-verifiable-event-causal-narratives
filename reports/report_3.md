# CEN-TS 中文实验报告（三）：强 Backbone 事件融合、事件窗口评价与归一化指标

生成日期：2026-07-10  
项目路径：`C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives`  
GitHub 仓库：`https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives`

## 1. 报告摘要

本报告严格承接 `reports/report_1.md` 与 `reports/report_2.md` 的结论，执行第二阶段报告中列出的全部下一步 priority。

`report_2` 的关键判断是：第一阶段 ridge backbone 太弱，第二阶段接入 MM-TSFlib/PyTorch 后，`TimesNet` 与 `PatchTST` 已明显超过 ridge numerical-only 和 ridge event_with_verifier；因此第三阶段不应继续重复 ridge 消融，而应把 CEN-TS 的事件/verifier 特征接入强 backbone，并补充事件窗口评价与归一化指标。

本轮完成的工作包括：

- 新增 `scripts/19_run_torch_event_fusion.py`。
- 在强 backbone 上比较四组输入：
  - `strong_numerical_only`
  - `strong_raw_text_fusion`
  - `strong_event_no_verifier`
  - `strong_event_with_verifier`
- 使用 `TimesNet` 和 `PatchTST` 两个第二阶段最重要的强 backbone。
- 保持 `Energy / Health_US / Health_AFR`、`H=3/6`、`seeds=2026/2027/2028`、`history_length=24` 的同配置矩阵。
- 新增 normalized metrics：
  - `NMSE`
  - `sMAPE`
  - relative improvement over persistence
  - per-domain/horizon rank
- 新增事件窗口与高波动窗口评价：
  - `event_window`
  - `high_volatility_window`
- 对 Energy 执行 strong event fusion 下的 shuffled-text robustness check。
- 继续遵守 `report_2` 对 GAMETime 与 LLM extraction 的判断：在完整 GAMETime benchmark 数据不可用前不写主实验数字；LLM extraction 不抢在 strong fusion 失败案例分析之前做大规模调用。

本轮最重要的新结论是：

> 强事件融合没有整体超过最强 numerical-only TimesNet。总体 MSE 最优仍是 `strong_numerical_only + TimesNet = 0.3986`；`strong_event_with_verifier + PatchTST = 0.4572`，略优于 `strong_numerical_only + PatchTST = 0.4607`，但优势很小；`strong_event_with_verifier + TimesNet = 0.4576`，反而弱于 TimesNet numerical-only。

因此第三阶段给出的科研判断更细：**verified event features 在 PatchTST 上出现了轻微正增益，在 Energy 上对 TimesNet 也有局部正增益，但目前不是稳定提升所有强 backbone 的通用模块。** CEN-TS 的下一步应从“简单拼接事件特征”转向“事件门控、残差校正、事件窗口专用训练和 verifier 校准”。

## 2. 第三阶段实验目标

本轮严格执行 `report_2` 第 11 节的下一阶段计划。

### Priority 1：强 backbone + event feature fusion

已完成。新增脚本把事件特征接入 torch/MM-TSFlib backbone，比较：

| 组别 | 实现状态 |
|---|---|
| strong numerical-only | 已完成 |
| strong raw text fusion | 已完成 |
| strong event_no_verifier | 已完成 |
| strong event_with_verifier | 已完成 |

### Priority 2：优先 Energy 做事件窗口评估

已完成。Energy 被单独输出 shuffled-text robustness，并且所有 domain 都输出 event-window / high-volatility-window 子集指标。

### Priority 3：引入 normalized metrics

已完成。主表新增：

- `nmse`
- `smape`
- `relative_improvement_over_persistence`
- `rank_mse_in_domain_horizon`

### Priority 4：继续跳过 GAMETime 主实验，直到完整数据可用

已执行。当前 clone 仍不能证明包含完整 GAMETime benchmark 数据，因此本报告不生成 GAMETime 主实验数字。

### Priority 5：LLM extraction 不抢在 strong fusion 前面

已执行。第一阶段 LLM probe 已接通但返回空事件；本轮优先完成 strong fusion 实验，把 LLM prompt 优化留给下一轮基于失败案例进行。

## 3. 代码改动

### 3.1 新增脚本

新增文件：

```text
scripts/19_run_torch_event_fusion.py
```

运行命令：

```bash
python scripts/19_run_torch_event_fusion.py --config configs/exp/main_timemmd.yaml --models TimesNet PatchTST --epochs 3 --batch-size 64
```

该脚本复用第二阶段 `scripts/18_run_torch_strong_baselines.py` 的 MM-TSFlib 调用方式，但增加了三类能力：

1. 对同一 strong backbone 构造不同输入特征。
2. 保存 normal / event-window / high-volatility-window 指标。
3. 对 Energy 的 `strong_event_with_verifier` 做 shuffled-text robustness。

### 3.2 输出文件

本轮输出到：

```text
experiments/tables/torch_event_fusion.csv
experiments/tables/torch_event_fusion_summary.csv
experiments/tables/torch_event_window_metrics.csv
experiments/tables/torch_energy_robustness.csv
experiments/tables/torch_event_fusion_failures.csv
```

时间戳 run 目录为：

```text
experiments/runs/20260710_003331_torch_event_fusion/
```

主结果表共 `162` 行：

```text
18 persistence rows
+ 4 fusion methods * 2 models * 3 domains * 2 horizons * 3 seeds
= 162 rows
```

failure 表为空，说明本轮矩阵全部跑通。

## 4. 实验设置

### 4.1 数据与任务

本轮继续使用 Time-MMD processed 数据：

```text
data/processed/TimeMMD/
```

领域：

- Energy
- Health_US
- Health_AFR

任务设置：

| 项 | 设置 |
|---|---|
| target | `OT` |
| history length | `24` |
| horizon | `3, 6` |
| seeds | `2026, 2027, 2028` |
| max rows | `1200` |
| split | temporal split |

### 4.2 Backbone

本轮使用第二阶段中最有代表性的两个 strong backbone：

| 模型 | 选择原因 |
|---|---|
| `TimesNet` | 第二阶段总体最优，MSE 为 `0.3986` |
| `PatchTST` | 第二阶段第二强，且适合作为 Transformer-style 对照 |

训练设置保持 CPU 快速矩阵：

```text
epochs = 3
batch_size = 64
learning_rate = 1e-3
device = cpu
```

该设置仍不是最终调参结果，而是用于判断事件融合方向是否有初步信号。

### 4.3 输入组别

| 方法 | 输入 |
|---|---|
| `strong_numerical_only` | 数值变量窗口 |
| `strong_raw_text_fusion` | 数值变量窗口 + TF-IDF/SVD text embedding |
| `strong_event_no_verifier` | 数值变量窗口 + 未过滤事件特征 |
| `strong_event_with_verifier` | 数值变量窗口 + verifier-filtered event features |

事件特征仍沿用第一阶段：

```text
event_count
pos_events
neg_events
event_confidence
verified_event_score
```

## 5. 主结果

### 5.1 聚合结果

| 方法 | Backbone | MSE | MAE | NMSE | sMAPE | Trend F1 |
|---|---|---:|---:|---:|---:|---:|
| `persistence` | persistence | 0.2900 | 0.2240 | 0.2563 | 0.2020 | 0.0071 |
| `strong_numerical_only` | TimesNet | 0.3986 | 0.3036 | 0.3936 | 0.2768 | 0.5076 |
| `strong_event_with_verifier` | PatchTST | 0.4572 | 0.3242 | 0.4280 | 0.2851 | 0.4923 |
| `strong_event_no_verifier` | PatchTST | 0.4572 | 0.3242 | 0.4280 | 0.2851 | 0.4923 |
| `strong_event_with_verifier` | TimesNet | 0.4576 | 0.3220 | 0.4602 | 0.2947 | 0.4998 |
| `strong_raw_text_fusion` | PatchTST | 0.4579 | 0.3243 | 0.4288 | 0.2855 | 0.4942 |
| `strong_event_no_verifier` | TimesNet | 0.4587 | 0.3263 | 0.4670 | 0.2980 | 0.4953 |
| `strong_numerical_only` | PatchTST | 0.4607 | 0.3251 | 0.4308 | 0.2858 | 0.4924 |
| `strong_raw_text_fusion` | TimesNet | 0.5221 | 0.3526 | 0.5365 | 0.3200 | 0.4952 |

关键比较：

| 对比 | 结论 |
|---|---|
| `TimesNet numerical-only` vs all fusion | `TimesNet numerical-only` 仍是最强学习模型 |
| `PatchTST event_with_verifier` vs `PatchTST numerical-only` | MSE 从 `0.4607` 降到 `0.4572`，约 `0.7%` 小幅提升 |
| `TimesNet event_with_verifier` vs `TimesNet numerical-only` | MSE 从 `0.3986` 升到 `0.4576`，事件拼接伤害 TimesNet |
| `event_with_verifier` vs `raw_text_fusion` | 平均 MSE `0.4574` vs `0.4900`，事件结构比 raw text fusion 稳 |
| `event_with_verifier` vs `event_no_verifier` | 平均 MSE `0.4574` vs `0.4579`，verifier 过滤有极小收益 |

### 5.2 与 report_2 的关系

`report_2` 中第二阶段强 baseline 结论是：

```text
mmts_timesnet MSE = 0.3986
mmts_patchtst MSE = 0.4607
persistence MSE = 0.2631
```

本轮在同一 strong backbone 上加入事件特征后：

```text
strong_event_with_verifier + TimesNet = 0.4576
strong_event_with_verifier + PatchTST = 0.4572
```

这说明：

- 事件特征没有让 TimesNet 更强。
- 事件特征让 PatchTST 略好于自身 numerical-only。
- 事件特征仍比 raw text fusion 稳，尤其 TimesNet raw text fusion 明显变差。
- persistence 仍是短期平滑预测最强参照。

## 6. 分领域结果

### 6.1 Energy

| 方法 | Backbone | MSE | NMSE |
|---|---|---:|---:|
| `strong_event_with_verifier` | TimesNet | 0.0516 | 0.1004 |
| `strong_numerical_only` | TimesNet | 0.0539 | 0.1048 |
| `strong_event_no_verifier` | TimesNet | 0.0546 | 0.1063 |
| `strong_event_with_verifier` | PatchTST | 0.0610 | 0.1186 |
| `strong_numerical_only` | PatchTST | 0.0610 | 0.1187 |
| `strong_raw_text_fusion` | TimesNet | 0.0613 | 0.1193 |

Energy 是本轮最支持 CEN-TS 事件融合的领域。`strong_event_with_verifier + TimesNet` 相比 `strong_numerical_only + TimesNet` 有约 `4.1%` MSE 降低。

按 horizon 看：

| Horizon | 最好学习模型 | MSE |
|---:|---|---:|
| 3 | `strong_event_with_verifier + TimesNet` | 0.0360 |
| 6 | `strong_event_with_verifier + TimesNet` | 0.0672 |

这与第一阶段 robustness 中 Energy 文本扰动 drop 最大的结论一致：Energy 中的文本事件信号确实更可能与数值变化有关。

### 6.2 Health_AFR

| 方法 | Backbone | MSE | NMSE |
|---|---|---:|---:|
| `strong_numerical_only` | TimesNet | 0.0021 | 0.6371 |
| `strong_event_with_verifier` | PatchTST | 0.0022 | 0.6617 |
| `strong_raw_text_fusion` | PatchTST | 0.0022 | 0.6631 |
| `strong_event_with_verifier` | TimesNet | 0.0026 | 0.7727 |

Health_AFR 的绝对 MSE 很低，但 NMSE 并不低，说明第二阶段提出的“绝对误差受尺度影响”判断是对的。该领域不应只看 raw MSE。

### 6.3 Health_US

| 方法 | Backbone | MSE | NMSE |
|---|---|---:|---:|
| `strong_numerical_only` | TimesNet | 1.1398 | 0.4388 |
| `strong_event_with_verifier` | PatchTST | 1.3085 | 0.5037 |
| `strong_raw_text_fusion` | PatchTST | 1.3102 | 0.5044 |
| `strong_event_with_verifier` | TimesNet | 1.3184 | 0.5076 |

Health_US 中事件融合没有带来正收益。该领域需要更强事件抽取、目标子集划分或 lag 学习；简单事件 count/polarity 拼接不足以帮助 strong backbone。

## 7. 事件窗口与高波动窗口评价

### 7.1 Event-window

本轮使用同一组 verified-event reference mask，对所有方法切同一批事件窗口，避免只对 event-feature 方法有窗口评价。

| 方法 | Backbone | 平均窗口数 | MSE | MAE | Trend F1 |
|---|---|---:|---:|---:|---:|
| `strong_numerical_only` | TimesNet | 235 | 0.3986 | 0.3036 | 0.5076 |
| `strong_event_with_verifier` | PatchTST | 235 | 0.4572 | 0.3242 | 0.4923 |
| `strong_event_with_verifier` | TimesNet | 235 | 0.4576 | 0.3220 | 0.4998 |
| `strong_raw_text_fusion` | PatchTST | 235 | 0.4579 | 0.3243 | 0.4942 |
| `strong_numerical_only` | PatchTST | 235 | 0.4607 | 0.3251 | 0.4924 |
| `strong_raw_text_fusion` | TimesNet | 235 | 0.5221 | 0.3526 | 0.4952 |

由于当前规则抽取几乎在测试窗口中广泛产生事件，event-window 的均值接近全测试集均值。这个结果暴露了一个问题：**当前 event-window 定义还不够稀疏，下一步需要区分 high-confidence external event、普通 observation 和文本趋势描述。**

### 7.2 High-volatility-window

| 方法 | Backbone | 平均窗口数 | MSE | MAE | Trend F1 |
|---|---|---:|---:|---:|---:|
| `strong_numerical_only` | TimesNet | 51.5 | 0.9209 | 0.5432 | 0.5495 |
| `strong_event_no_verifier` | TimesNet | 51.5 | 1.0864 | 0.5919 | 0.5363 |
| `strong_event_with_verifier` | TimesNet | 51.5 | 1.0990 | 0.5897 | 0.5237 |
| `strong_raw_text_fusion` | TimesNet | 51.5 | 1.1362 | 0.6109 | 0.5089 |
| `strong_event_with_verifier` | PatchTST | 51.5 | 1.1559 | 0.6186 | 0.5099 |
| `strong_numerical_only` | PatchTST | 51.5 | 1.1660 | 0.6213 | 0.5249 |

高波动窗口中 TimesNet numerical-only 仍最强。这说明当前事件特征并没有自动解决突变预测问题。事件特征需要更精确地识别外生冲击，而不是把大量文本 observation 都转成辅助变量。

## 8. Energy Shuffled-text Robustness

本轮对 Energy 的 `strong_event_with_verifier` 做 shuffled-text 对照。

| Backbone | Horizon | 原始 MSE | Shuffled-text MSE | Drop |
|---|---:|---:|---:|---:|
| TimesNet | 3 | 0.0360 | 0.0376 | +0.0016 |
| TimesNet | 6 | 0.0672 | 0.0663 | -0.0009 |
| PatchTST | 3 | 0.0492 | 0.0492 | 0.0000 |
| PatchTST | 6 | 0.0727 | 0.0727 | 0.0000 |

解释：

- Energy 中事件特征对 TimesNet 的总体 MSE 有小幅正收益。
- 但 shuffled-text robustness 的 drop 很小，说明当前强融合模型对事件特征的依赖还不强。
- PatchTST 原始与 shuffled 完全一致，说明当前规则事件特征在 PatchTST 中可能只是非常弱的辅助信号，或者 event_no_verifier / event_with_verifier 特征在当前 threshold 下没有形成足够差异。

## 9. 失败分析

### 9.1 简单拼接事件特征不足以稳定提升 strong backbone

第一阶段事件特征可以明显优于 raw text ridge，但 strong backbone 已经能从数值窗口中学习大量结构。此时简单拼接 `event_count / polarity / confidence / score` 不一定带来额外信息，反而可能增加噪声。

### 9.2 Verifier 过滤收益太小

本轮 `event_with_verifier` 与 `event_no_verifier` 很接近。尤其 PatchTST 上二者完全一致，说明当前 verifier threshold 产生的特征差异不足，或过滤后的事件仍过于密集。

下一步需要：

- 调高 verifier threshold。
- 使用 top-k verified event。
- 区分 external event 与 observation。
- 只保留对目标变量有显著 post-event effect 的事件。

### 9.3 Raw text fusion 仍不稳

`strong_raw_text_fusion + TimesNet` 的 MSE 为 `0.5221`，明显弱于 TimesNet numerical-only 的 `0.3986`。这再次支持 CEN-TS 的基本出发点：直接文本 embedding 拼接很容易向强模型引入噪声。

### 9.4 Persistence 仍是短期任务强对手

本轮同 split 下 persistence 平均 MSE 为 `0.2900`，仍低于所有学习模型。该结论与前两份报告一致：短 horizon 平滑预测不一定是事件方法最容易胜出的评价场景。

## 10. 当前科研判断

### 10.1 本轮是否完成 report_2 的下一步？

完成。对应关系如下：

| report_2 priority | 本轮状态 |
|---|---|
| 强 backbone + event feature fusion | 已实现并跑完 |
| Energy 事件窗口评价 | 已实现并输出 robustness |
| normalized metrics | 已实现 |
| GAMETime 无完整数据前不写主结果 | 已遵守 |
| LLM extraction 不抢在 strong fusion 前面 | 已遵守 |

### 10.2 是否证明 CEN-TS 已超过 strong numerical model？

尚未证明。

本轮最稳妥结论是：

> CEN-TS 的 verified event features 在 strong backbone setting 中出现了局部正信号，尤其是 Energy 上的 TimesNet 与整体 PatchTST 上的小幅提升；但它尚未稳定超过最强 numerical-only TimesNet，也未超过 persistence。因此当前事件融合方式还不能作为最终方法，只能作为下一轮 event-aware architecture 的依据。

### 10.3 是否继续支持“结构化事件优于 raw text”？

继续支持。

平均 MSE：

```text
strong_event_with_verifier = 0.4574
strong_raw_text_fusion     = 0.4900
```

结构化事件仍比 raw text embedding fusion 稳，尤其避免了 `TimesNet + raw text` 的明显退化。

## 11. 下一阶段实验计划

基于第三阶段结果，下一步 priority 应调整为以下方向。

### Priority 1：从简单拼接升级为 event-aware fusion

当前事件特征只是作为普通 covariate 拼入 backbone。下一步应实现：

- event-gated residual correction
- event attention mask
- event-conditioned adapter
- verified-event-only side channel
- numerical forecast + event residual two-stage model

核心目标不是让事件特征替代数值 backbone，而是让事件只在它有证据的窗口中修正强 numerical forecast。

### Priority 2：重做稀疏事件窗口

当前 event-window 太密集。下一步应把事件分为：

- external shock
- observation
- forecast statement
- policy / supply / demand / health intervention
- uncertain background

只用 high-confidence external shock 构造 event-triggered evaluation。

### Priority 3：校准 verifier threshold

需要系统扫描：

```text
verifier_threshold in {0.5, 0.6, 0.7, 0.8}
top_k events per window
event score weighted features
```

并报告 precision / coverage trade-off，而不是固定 `0.5`。

### Priority 4：LLM extraction 基于失败案例优化

现在 strong fusion 失败案例已经存在，LLM prompt 优化应围绕这些问题：

- 把 observation 与 causal event 分开。
- 要求判断是否外生。
- 对 Energy 给出 supply/demand/refinery/seasonality schema。
- 对 Health_US 给出 reporting delay / disease spread / policy intervention schema。
- 对抽出的事件直接运行 verifier，并用低分事件反向改 prompt。

### Priority 5：更长训练与 GPU 复现实验

本轮 CPU 矩阵只有 3 epochs。下一轮如有预算，应在同一脚本上增加：

- epochs
- patience
- model dimension
- repeated seeds
- GPU run

并检查 event fusion 的小幅收益是否稳定。

## 12. 结论

第三阶段完成了 `report_2` 指定的下一步实验：CEN-TS 事件/verifier 特征已经接入 TimesNet/PatchTST 强 backbone，并补充了 normalized metrics、event-window evaluation、high-volatility-window evaluation 和 Energy shuffled-text robustness。

本轮结果给出更严格也更真实的判断：

- `TimesNet numerical-only` 仍是最强学习模型，MSE 为 `0.3986`。
- `PatchTST + verified events` 有小幅正收益，MSE 从 `0.4607` 降到 `0.4572`。
- `Energy + TimesNet + verified events` 有局部正收益，MSE 从 `0.0539` 降到 `0.0516`。
- `TimesNet + raw text fusion` 明显退化，MSE 升到 `0.5221`。
- 事件结构仍比 raw text fusion 稳，但简单拼接还不足以稳定超过最强 numerical-only。
- persistence 仍是短期平滑预测的强基线。

因此，CEN-TS 的研究方向仍然成立，但方法形态需要升级：下一步不应继续证明“事件特征比 raw text 好”，而应实现真正的 **event-aware strong forecasting architecture**，让 verified event 只在事件窗口、转折窗口和高波动窗口中以门控或残差方式修正强数值预测。这样才有机会把“可验证事件因果叙事”的解释价值转化为稳定的预测增益。
