# CEN-TS 中文实验报告（二）：PyTorch/MM-TSFlib 强时间序列基线接入与第二阶段结果

生成日期：2026-07-09  
项目路径：`C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives`  
GitHub 仓库：`https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives`

## 1. 报告摘要

本报告严格承接 `reports/report_1.md` 的结论与下一阶段计划，优先执行其中的 **Priority 1：安装 PyTorch，并接入 MM-TSFlib 强时间序列 baseline**。

第一阶段的关键结论是：

- `event_with_verifier` 相比 `raw_text_embedding` 明显更稳，平均 MSE 从 `3.0621` 降到 `1.1167`。
- 但事件方法尚未超过 `numerical_only`，后者平均 MSE 为 `1.0038`。
- `persistence` 仍然最强，平均 MSE 为 `0.2631`。
- 因此下一步不能继续只看轻量 ridge，需要接入更强的时间序列 backbone。

本轮实验完成了以下工作：

- 没有绕过 `torch`：已在当前 Python310 环境安装 `torch==2.13.0+cpu`。
- 为 MM-TSFlib 注意力模型补齐 `reformer-pytorch` 依赖。
- 新增 `scripts/18_run_torch_strong_baselines.py`，直接调用 `external/MM-TSFlib/models/` 下模型文件训练。
- 在同一批 Time-MMD domain、同一 split、同一 history/horizon 设置上运行：
  - DLinear
  - PatchTST
  - iTransformer
  - TimesNet
  - TimeMixer
- GAMETime 完整 benchmark 数据仍未获得，因此本轮继续跳过 GAMETime 主实验，不伪造结果。

第二阶段最重要的新结论是：**强 numerical backbone 已经显著超过第一阶段 ridge numerical-only 和 event_with_verifier，但仍未超过 persistence。** 其中 TimesNet 平均 MSE 最低，为 `0.3986`；PatchTST 为 `0.4607`。这说明第一阶段的主要瓶颈确实包含 forecasting backbone，但也说明 CEN-TS 下一步必须把事件/verifier 特征接入强 backbone，而不是只停留在 ridge feature mode。

## 2. 第二阶段实验目标

根据 `report_1` 的第 11 节，本轮只执行当前可验证的下一步：

```text
Priority 1:
安装 PyTorch，并基于 MM-TSFlib 跑强时间序列 baseline。
```

本轮不做以下工作：

- 不运行 GAMETime 主 benchmark，因为完整数据仍未获得。
- 不把 missing torch 当作环境限制继续跳过。
- 不把强 baseline 替换成无 torch 的简化实现。
- 不声称 CEN-TS 已经超过强 backbone。

本轮核心问题是：

> 当使用更强的 MM-TSFlib/PyTorch numerical backbone 后，第一阶段 CEN-TS 结果的位置如何变化？

## 3. 环境与依赖更新

### 3.1 PyTorch 安装状态

当前环境已安装：

```text
torch==2.13.0+cpu
cuda_available=False
```

这说明本轮实验是真实 torch 训练，但运行在 CPU 上；因此训练轮数设置较小，用于快速完成第二阶段强 baseline 接入，而不是最终大规模调参。

### 3.2 MM-TSFlib 依赖修复

PatchTST 和 iTransformer 依赖 MM-TSFlib 的 attention layer，而该 layer 静态依赖：

```text
reformer_pytorch
```

本轮已安装：

```text
reformer-pytorch>=1.4
```

并将 `torch` 与 `reformer-pytorch` 写入 `requirements.txt` 和 `pyproject.toml`，避免只在当前环境中临时可用。

## 4. 代码改动

### 4.1 新增强 baseline 脚本

新增脚本：

```text
scripts/18_run_torch_strong_baselines.py
```

该脚本的作用是：

1. 读取 `configs/exp/main_timemmd.yaml`。
2. 使用 `data/processed/TimeMMD/` 中第一阶段同样的 processed CSV。
3. 使用同样的自动 domain 选择逻辑：
   - Energy
   - Health_US
   - Health_AFR
4. 使用同样的任务设置：
   - `history_length=24`
   - `horizons=[3, 6]`
   - `seeds=[2026, 2027, 2028]`
   - temporal split，不随机打乱时间。
5. 从 `external/MM-TSFlib/models/` 动态加载模型。
6. 使用 torch 训练并输出测试集 MSE/MAE/RMSE/MAPE/trend_f1。

输出文件：

```text
experiments/tables/torch_strong_baselines.csv
experiments/tables/torch_strong_baseline_failures.csv
```

最终强 baseline 结果共 `90` 行：

```text
5 models * 3 domains * 2 horizons * 3 seeds = 90
```

最终 failure 表为空，说明五个模型均已成功完成本轮矩阵。

## 5. 实验设置

### 5.1 数据集

本轮只使用 Time-MMD processed 数据：

```text
data/processed/TimeMMD/
```

本轮没有运行 GAMETime 主 benchmark。原因与第一阶段一致：当前 clone 的 `data/raw/GAMETime/` 不包含完整 benchmark 数据，因此只能保留 sanity check，不应产生 GAMETime 主结果。

### 5.2 模型矩阵

本轮强 baseline 包括：

| 方法 | 来源 | 说明 |
|---|---|---|
| `mmts_dlinear` | MM-TSFlib `DLinear.py` | 分解线性强基线 |
| `mmts_patchtst` | MM-TSFlib `PatchTST.py` | patch-based Transformer |
| `mmts_itransformer` | MM-TSFlib `iTransformer.py` | inverted Transformer |
| `mmts_timesnet` | MM-TSFlib `TimesNet.py` | period/FFT-based temporal model |
| `mmts_timemixer` | MM-TSFlib `TimeMixer.py` | multi-scale mixing model |

### 5.3 训练设置

为适配当前 CPU 环境，本轮使用小规模训练：

```text
epochs = 3
batch_size = 64
learning_rate = 1e-3
device = cpu
```

该设置的定位是“真实强 backbone 接入与初步比较”，不是最终调参结果。后续若有 GPU 或更长实验预算，应增加 epoch、patience、模型维度和搜索范围。

## 6. 第二阶段主结果

### 6.1 强 baseline 聚合结果

| 方法 | MSE | MAE | Trend F1 |
|---|---:|---:|---:|
| `mmts_timesnet` | 0.3986 | 0.3036 | 0.5076 |
| `mmts_patchtst` | 0.4607 | 0.3251 | 0.4924 |
| `mmts_itransformer` | 0.5487 | 0.3543 | 0.4909 |
| `mmts_dlinear` | 0.6571 | 0.3944 | 0.5422 |
| `mmts_timemixer` | 0.6611 | 0.3985 | 0.4999 |

最优模型是 `mmts_timesnet`，平均 MSE 为 `0.3986`。

与第一阶段主结果对比：

| 方法 | MSE | MAE |
|---|---:|---:|
| `persistence` | 0.2631 | 0.2396 |
| `mmts_timesnet` | 0.3986 | 0.3036 |
| `mmts_patchtst` | 0.4607 | 0.3251 |
| `numerical_only` | 1.0038 | 0.5651 |
| `event_with_verifier` | 1.1167 | 0.5950 |
| `raw_text_embedding` | 3.0621 | 0.9582 |

结论很清楚：

- 强 torch numerical backbone 明显超过第一阶段 ridge numerical-only。
- 强 torch numerical backbone 也明显超过当前 ridge-based event_with_verifier。
- 但强 torch numerical backbone 仍未超过 persistence。
- 当前 CEN-TS 的下一步必须是把 event/verifier 特征接入这些强 backbone，而不是继续和 ridge baseline 做主要比较。

## 7. 分 domain 与 horizon 结果

### 7.1 Energy

| 方法 | H=3 MSE | H=6 MSE |
|---|---:|---:|
| `mmts_timesnet` | 0.0379 | 0.0698 |
| `mmts_patchtst` | 0.0492 | 0.0728 |
| `mmts_itransformer` | 0.0539 | 0.0952 |
| `mmts_timemixer` | 0.0611 | 0.1045 |
| `mmts_dlinear` | 0.0972 | 0.1188 |

Energy 是本轮最支持强 backbone 的领域。第一阶段鲁棒性实验也显示 Energy 文本扰动会导致明显性能下降，因此 Energy 应作为下一步 CEN-TS strong-backbone fusion 的优先 domain。

### 7.2 Health_US

| 方法 | H=3 MSE | H=6 MSE |
|---|---:|---:|
| `mmts_timesnet` | 0.8761 | 1.4034 |
| `mmts_patchtst` | 1.0616 | 1.5759 |
| `mmts_itransformer` | 1.1411 | 1.9969 |
| `mmts_timemixer` | 1.3355 | 2.4600 |
| `mmts_dlinear` | 1.6979 | 2.0215 |

Health_US 仍然较难，尤其 H=6 明显变差。该结果与第一阶段中 Health_US 文本扰动影响不稳定的现象一致，说明该领域需要更细的事件抽取、lag 学习和目标子集分析。

### 7.3 Health_AFR

| 方法 | H=3 MSE | H=6 MSE |
|---|---:|---:|
| `mmts_timesnet` | 0.0017 | 0.0025 |
| `mmts_patchtst` | 0.0018 | 0.0026 |
| `mmts_itransformer` | 0.0020 | 0.0030 |
| `mmts_timemixer` | 0.0022 | 0.0036 |
| `mmts_dlinear` | 0.0030 | 0.0041 |

Health_AFR 的绝对误差很低，但这可能与目标变量尺度和序列平滑性有关，不能单独解释为方法优势。下一步需要加入 scale-normalized 指标或 per-domain normalized MSE，避免聚合均值被小尺度 domain 影响。

## 8. 与第一阶段结论的关系

### 8.1 第一阶段“backbone 太弱”的判断被证实

`report_1` 判断 ridge window predictor 不足以作为 A 会/顶刊级别 baseline。本轮结果支持这个判断：

```text
ridge numerical_only MSE = 1.0038
best torch baseline MSE = 0.3986
```

TimesNet 相比 ridge numerical-only 平均 MSE 降低约：

```text
(1.0038 - 0.3986) / 1.0038 = 60.3%
```

这说明继续只在 ridge 上调事件特征没有意义。事件特征必须进入强 backbone 才能判断 CEN-TS 对预测误差是否真的有增益。

### 8.2 “事件方法优于 raw text”仍成立，但标准提高了

第一阶段：

```text
event_with_verifier MSE = 1.1167
raw_text_embedding MSE = 3.0621
```

这仍然说明结构化事件比原始文本 embedding 更稳。

但第二阶段引入强 numerical backbone 后，新的主要比较对象变成：

```text
strong numerical-only backbone
vs
strong backbone + raw text
vs
strong backbone + verified event features
```

因此下一轮 CEN-TS 不能只证明“比 raw text ridge 好”，而必须回答：

> verified event features 能否在 TimesNet/PatchTST 这类强 numerical backbone 上继续提供增益？

### 8.3 Persistence 仍然是关键挑战

本轮最优模型 TimesNet 平均 MSE 为 `0.3986`，仍高于 persistence 的 `0.2631`。

这说明 `report_1` 中关于任务设置的判断仍然成立：

- 当前短 horizon 平滑预测对 persistence 有利。
- 如果只看普通 MSE，复杂模型未必能打败最后值延续。
- CEN-TS 的价值可能更容易体现在事件窗口、高波动窗口、转折点、OOD split 和解释可验证性上。

## 9. 失败与修复记录

本轮没有隐藏中间失败。主要问题与修复如下：

1. 初始环境没有 `torch`。  
   已安装 `torch==2.13.0+cpu`。

2. PatchTST/iTransformer 初次运行缺少 `reformer_pytorch`。  
   已安装 `reformer-pytorch>=1.4` 并写入依赖文件。

3. TimeMixer 初次 wrapper 配置中 `down_sampling_layers=0`，源码的 trend mixer 需要至少两个尺度。  
   已对 TimeMixer 单独设置 `down_sampling_layers=1` 和 `down_sampling_window=2`。

最终：

```text
experiments/tables/torch_strong_baseline_failures.csv
```

为空 failure 表，仅保留表头：

```text
seed,domain,model,horizon,error
```

## 10. 当前科研判断

### 10.1 本轮是否完成 report_1 的下一步？

完成了 Priority 1 的可执行部分。

具体包括：

- PyTorch 已安装。
- MM-TSFlib 模型已被真实调用。
- DLinear、PatchTST、iTransformer、TimesNet、TimeMixer 已在 Time-MMD 上完成同配置矩阵。
- 结果已保存为 CSV。
- GAMETime 因完整数据未获得而继续跳过，没有伪造 benchmark。

### 10.2 本轮是否改变第一阶段结论？

改变了 baseline 格局，但没有改变 CEN-TS 的核心判断。

改变的地方：

- 第一阶段 ridge numerical-only 不再是足够强的 numerical baseline。
- TimesNet/PatchTST 等 torch backbone 显著更强。
- 当前 `event_with_verifier` 的预测误差优势必须重新放到 strong-backbone setting 中验证。

未改变的地方：

- raw text embedding 仍是一个不稳定路线。
- event+verifier 仍有解释性和鲁棒性价值。
- persistence 仍是短期平滑预测的强对手。
- GAMETime 仍不能在无完整数据时声称完成主实验。

### 10.3 当前最稳妥结论

> 第二阶段实验确认：CEN-TS 的第一阶段轻量 ridge backbone 确实不足。接入 torch/MM-TSFlib 后，TimesNet 和 PatchTST 显著超过 ridge numerical-only 与 ridge event_with_verifier，但仍未超过 persistence。因此下一步研究重点应从“事件特征是否优于 raw text ridge”升级为“verified event features 能否在强 numerical backbone 上提升事件窗口、转折点和高波动区间预测，同时保持可验证解释”。

## 11. 下一阶段实验计划

严格基于本轮结果，下一步不应再重复跑 ridge ablation，而应做以下改进。

### Priority 1：强 backbone + event feature fusion

在 `scripts/18_run_torch_strong_baselines.py` 基础上增加事件输入：

```text
numerical window
verified event feature window
event consistency score
polarity/count/confidence features
```

至少比较：

| 组别 | 说明 |
|---|---|
| strong numerical-only | 当前 TimesNet/PatchTST |
| strong raw text fusion | torch backbone + TF-IDF/SVD text features |
| strong event_no_verifier | torch backbone + unfiltered event features |
| strong event_with_verifier | torch backbone + verifier-filtered event features |

### Priority 2：优先 Energy 做事件窗口评估

Energy 同时满足：

- 强 backbone 效果最好。
- 第一阶段文本扰动 drop 最大。
- case study 中 verifier 分数高。

因此下一轮应在 Energy 上增加：

- event-window MSE
- high-volatility-window MSE
- turning point F1
- directional accuracy
- robustness under shuffled text

### Priority 3：引入 normalized metrics

Health_AFR 绝对 MSE 很低，可能受尺度影响。因此下一轮应加入：

- normalized MSE
- sMAPE
- per-domain rank
- relative improvement over persistence

### Priority 4：继续跳过 GAMETime 主实验，直到完整数据可用

在完整 GAMETime benchmark 数据获得前，只保留：

- repository availability check
- synthetic verifier sanity check
- 不写 GAMETime 主 benchmark 数字

### Priority 5：LLM extraction 不应抢在 strong fusion 前面

第一阶段 LLM probe 已打通但小样本返回空事件。现在更紧急的是先证明 verified event features 能否和 strong backbone 结合。LLM prompt 优化应跟随 event-fusion 失败案例进行，而不是提前大规模调用。

## 12. 结论

本轮实验按 `report_1` 的下一步计划，完成了 torch 安装和 MM-TSFlib 强时间序列 baseline 接入。结果显示，TimesNet、PatchTST、iTransformer、DLinear 和 TimeMixer 都已能在当前 Time-MMD processed 数据上真实训练和评估。

第二阶段最强结果是：

```text
mmts_timesnet MSE = 0.3986
mmts_patchtst MSE = 0.4607
```

它们显著优于第一阶段：

```text
numerical_only MSE = 1.0038
event_with_verifier MSE = 1.1167
raw_text_embedding MSE = 3.0621
```

但仍未超过：

```text
persistence MSE = 0.2631
```

因此，CEN-TS 当前最正确的下一步不是继续调轻量事件 ridge，而是把“可验证事件因果叙事”接入 TimesNet/PatchTST 等强 backbone，并把评价重点扩展到 event-window、turning point、高波动窗口、鲁棒性和解释可追溯性。这样才能真正判断 CEN-TS 是否不仅比 raw text embedding 稳，而且能在强数值模型上提供额外、可验证、可解释的预测增益。
