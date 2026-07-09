# CEN-TS 中文实验报告：面向文本配对时间序列预测的可验证事件因果叙事框架

生成日期：2026-07-09  
项目路径：`C:\Users\Administrator\Desktop\TS\cen-ts-verifiable-event-causal-narratives`  
GitHub 仓库：`https://github.com/simple0828/cen-ts-verifiable-event-causal-narratives`

## 1. 报告摘要

本报告总结 CEN-TS 第一阶段实验系统的研究思想、方法设计、实验流程、当前结果与下一步改进方向。

CEN-TS 的核心问题是：对于同时包含数值时间序列和时间对齐文本的预测任务，直接把文本 embedding 拼接进预测模型是否足够？我们的研究假设是：**与其把原始文本当作黑箱向量，不如先把文本转化为结构化事件，再构建带方向、滞后和置信度的 event-variable causal graph，并用时间序列变化反向验证这些事件解释，最后只把通过验证的事件因果叙事用于预测。**

第一阶段已经完成一个可复现实验闭环：

- 自动下载并处理 Time-MMD 和 GAMETime。
- 实现数值-only、raw text embedding、事件特征、事件验证、完整 CEN-TS feature mode 等轻量模型。
- 实现事件抽取 schema、规则抽取 fallback、LLM 事件抽取 probe、lagged causal graph、Granger-lite 边验证、event consistency verifier。
- 跑通 smoke test、main experiment、ablation、robustness、GAMETime sanity check、MM-TSFlib 可用性扫描。
- 生成英文报告、中文报告、CSV 表格和版本化 GitHub 提交。

当前实验结论必须谨慎解释：**event+verifier 相比 raw text embedding 有明显优势，但尚未稳定超过 numerical-only 和 persistence。** 这说明“可验证事件因果叙事”方向有研究价值，尤其是在解释性与鲁棒性上；但要达到 A 会/顶刊标准，还需要更强的时间序列 backbone、更强的 LLM 事件抽取、更严格的 causal discovery 和更完整的统计显著性验证。

## 2. 研究背景与核心问题

### 2.1 文本配对时间序列预测的难点

Time-MMD 这类数据集不仅包含数值序列，还包含与时间戳对齐的文本，例如能源价格报告、健康事件描述、经济新闻或预测性叙述。直观上，这些文本可能提供数值序列之外的额外信息，例如：

- 能源领域：油价、供需、炼厂产能、季节性需求。
- 健康领域：疫情、疾病传播、医院压力、公共卫生政策。
- 交通领域：拥堵、事故、出行需求变化。
- 经济领域：政策、市场情绪、通胀、就业等宏观因素。

但直接使用文本存在三个问题：

1. **文本 embedding 不可解释。**  
   TF-IDF、BERT 或 sentence-transformer embedding 可以提升输入维度，但很难回答“模型到底用了哪条事件信息”。

2. **文本噪声和时间错位会破坏预测。**  
   如果文本和数值变化没有真实因果关系，或者文本时间戳与事件影响滞后不匹配，raw text fusion 可能引入噪声。

3. **普通 LLM prompt 容易产生幻觉解释。**  
   LLM 可以生成看似合理的事件叙述，但如果这些叙述不能被数值序列变化验证，就不应直接作为预测依据。

### 2.2 CEN-TS 的基本观点

CEN-TS 的观点是：文本不应该直接作为一个不可解释向量输入，而应该经过如下中间层：

```text
timestamped text
  -> structured event extraction
  -> event-variable lagged causal graph
  -> time-series consistency verification
  -> verified event features / causal narrative
  -> forecasting model
```

这个设计把“文本是否有用”拆成几个可检验问题：

- 文本中有没有可结构化的事件？
- 事件是否指向某个数值变量？
- 事件影响方向是正向、负向、中性还是不确定？
- 事件影响是否有合理滞后？
- 事件之后的数值序列是否真的发生方向一致的变化？
- 过滤掉低一致性事件后，预测是否更稳、更可解释？

这使得研究从“文本 embedding 是否提升 MSE”扩展为“文本事件解释是否可验证、可鲁棒、可用于预测”。

## 3. 与相关工作的关系

### 3.1 TaTS / Texts as Time Series

TaTS 类工作强调：时间对齐文本本身也有时间结构，可以被编码成随时间变化的辅助变量。CEN-TS 中的 `raw_text_embedding` baseline 对应这个思想：

- 每个时间点的 `fact/preds` 文本被编码为 TF-IDF + TruncatedSVD embedding。
- 文本 embedding 和数值窗口拼接后输入 ridge window predictor。
- 如果该 baseline 有效，说明 raw text-as-time-series 能为预测提供信号。

当前结果显示，raw text embedding 在本阶段实验中表现较差，平均 MSE 为 `3.0621`，明显高于 event+verifier 的 `1.1167`。这说明在轻量模型和当前处理方式下，原始文本向量融合非常容易引入噪声。

### 3.2 Augur

Augur 的核心是发现多元时间序列 covariates 之间的 directed causal associations，并把因果摘要用于预测。CEN-TS 做了两层扩展：

- 数值层：加入 lagged correlation 和 Granger-lite 边验证。
- 文本事件层：把 event node 与 variable node 连接，构成 event-variable causal graph。

目前 Granger-lite 在 Energy 和 Health_US 中发现了大量显著滞后边。例如 Energy 中多个区域汽油价格变量对 `OT` 的 best lag 为 1，p-value 显著。这说明 Time-MMD 的数值变量之间确实存在可利用的滞后结构。

### 3.3 Inferring Event Descriptions from Time Series / GAMETime

该方向启发我们：如果一个事件解释声称影响了时间序列，那么事件之后的序列变化应该能反向支持它。CEN-TS 中的 event consistency verifier 正是基于这个思想。

实现上，对于每个事件：

```text
pre_window = event 前的目标变量窗口
post_window = event 后 lag 范围内的目标变量窗口
delta = mean(post_window) - mean(pre_window)
```

然后根据事件 polarity 检查：

- positive 事件：希望 delta > 0
- negative 事件：希望 delta < 0
- neutral / uncertain：不强制方向，但检查异常波动

同时使用随机窗口作为 counterfactual baseline，计算观察效应相对随机效应的 lift。

GAMETime 仓库本身已克隆，但仓库 README 说明完整 benchmark 数据需要额外申请或下载，当前 clone 不包含完整 1.7M timestamp 数据。因此本阶段没有伪造 GAMETime benchmark 结果，而是做了 verifier 的 synthetic sanity check：

| check | passed | score | 说明 |
|---|---:|---:|---|
| repository_data_availability | False | 0.000 | clone 仓库未包含完整 benchmark 数据 |
| synthetic_positive_event_verifier | True | 0.854 | 正向事件后目标变量上升 |
| synthetic_negative_event_verifier | True | 0.779 | 负向事件后目标变量下降 |

### 3.4 APO / PromptWizard

APO 的思想是利用验证集错误反馈自动优化 prompt。CEN-TS 第一阶段实现了轻量 Narrative-APO 接口，但没有进行大规模 prompt 搜索。原因是当前更重要的是先验证完整事件-图-验证-预测闭环是否成立。

当前已有接口包括：

- event extraction instruction
- event selection rule
- lag rule
- narrative template
- failure case 接口
- validation score 设计接口

下一阶段可将验证集中的高误差窗口、低 consistency 事件、方向错误事件作为 textual gradient 的输入，让 LLM 自动提出 prompt program 修改建议。

## 4. 数据集与数据处理

### 4.1 Time-MMD

主数据集为 Time-MMD，已按要求直接克隆到：

```text
data/raw/Time-MMD/
```

处理后数据保存在：

```text
data/processed/TimeMMD/
```

数据准备脚本：

```bash
python scripts/01_prepare_timemmd.py --raw_dir data/raw/Time-MMD --out_dir data/processed/TimeMMD
```

处理流程包括：

1. 自动发现 `numerical/` 和 `textual/` 下共同存在的 domains。
2. 读取 numerical csv，识别 `date`、`OT`、其他 covariates。
3. 读取 textual csv，合并 `fact` 和 `preds` 字段。
4. 按 `start_date/date` 对齐文本与数值时间戳。
5. 清洗缺失值、无穷值和空文本。
6. 输出每个 domain 的 processed csv 和 stats json。

本阶段实际发现的可用领域包括：

- Agriculture
- Climate
- Economy
- Energy
- Environment
- Health_AFR
- Health_US
- Security
- SocialGood
- Traffic

实验优先选择文本覆盖较高、序列长度足够的领域。当前主实验自动选择了：

- Energy
- Health_US
- Health_AFR

### 4.2 GAMETime

GAMETime 已克隆到：

```text
data/raw/GAMETime/
```

但 clone 仓库未包含完整 benchmark 数据。项目 README 明确说明完整数据需要额外申请或下载。因此当前报告只做仓库可用性检查和 verifier synthetic sanity check，不声称完成 GAMETime 主 benchmark。

### 4.3 MM-TSFlib

MM-TSFlib 已克隆到：

```text
external/MM-TSFlib/
```

扫描发现其中包含大量强时间序列模型文件：

- DLinear
- PatchTST
- iTransformer
- TimesNet
- TimeMixer
- Autoformer
- Informer
- FEDformer
- Transformer
- TSMixer

但当前本地 Python 环境没有安装 `torch`，因此没有运行完整 MM-TSFlib training。本报告将其记录为环境限制，而不是方法失败。

## 5. 方法设计

### 5.1 Baseline 0：Persistence

Persistence 使用最后一个观测值作为未来所有 horizon 的预测：

```text
y_hat_{t+1:t+H} = y_t
```

这个 baseline 非常简单，但对于平滑时间序列可能很强。当前结果中 persistence 的平均 MSE 为 `0.2631`，是所有方法中最低的。这说明 Time-MMD 当前选中的某些序列存在较强短期自相关和平滑性。

这也提醒我们：如果研究方法不能超过 persistence，并不一定说明事件方法完全无效，而可能说明：

- 预测 horizon 较短；
- 目标序列非常平滑；
- 当前模型 backbone 太弱；
- 事件影响可能更多体现在方向、极端变化或鲁棒性，而不是短期 MSE。

### 5.2 Baseline 1：Numerical-only

Numerical-only 使用历史数值窗口，不使用文本。当前实现为轻量 ridge window predictor：

```text
Input: X_{t-L+1:t}
Output: y_{t+1:t+H}
```

平均结果：

| 方法 | MSE | MAE |
|---|---:|---:|
| numerical_only | 1.0038 | 0.5651 |

该 baseline 是判断文本和事件是否有增益的关键参照。

### 5.3 Baseline 2：Raw Text Embedding Fusion

Raw text embedding fusion 模拟 TaTS 思路：

```text
text_t -> TF-IDF -> TruncatedSVD -> embedding_t
[numerical_window, text_embedding_window] -> predictor
```

平均结果：

| 方法 | MSE | MAE |
|---|---:|---:|
| raw_text_embedding | 3.0621 | 0.9582 |

这是当前最差的主要方法，说明直接拼接文本 embedding 在轻量模型下非常不稳。可能原因包括：

- 文本维度与数值信号尺度不匹配。
- TF-IDF/SVD 不能捕捉真实事件结构。
- 文本中包含大量泛化背景或预测性话术。
- 文本时间戳与数值影响之间存在滞后，直接同时间拼接会错配。

### 5.4 Method 1：结构化事件特征

事件抽取 schema 包括：

```json
{
  "time": "...",
  "source_text_id": "...",
  "event_phrase": "...",
  "event_type": "...",
  "entities": ["..."],
  "target_variable": "OT or one covariate",
  "polarity": "positive | negative | neutral | uncertain",
  "expected_lag_min": 1,
  "expected_lag_max": 3,
  "magnitude": "weak | medium | strong | uncertain",
  "confidence": 0.0,
  "rationale": "short explanation"
}
```

当前主实验使用规则抽取 fallback，原因是第一阶段重点是验证实验闭环。规则抽取会从文本中识别：

- 方向词：increase、decrease、rise、drop 等。
- 领域词：oil、gasoline、health、traffic、market 等。
- 极性 polarity。
- 默认 lag：1 到 3 个时间步。
- 置信度 confidence。

LLM probe 已经接通 OpenAI API，检测到 API key 且调用成功。但在小样本 Energy 文本上，模型保守返回空事件。因此主实验仍使用 deterministic rule-based extractor。

### 5.5 Method 2：Event-variable lagged causal graph

图结构包括两类节点：

- event nodes
- numerical variable nodes

边包括：

- `event_to_var`
- `var_to_var`

每条边包含：

```json
{
  "source": "...",
  "target": "...",
  "edge_type": "event_to_var | var_to_var",
  "polarity": "positive | negative | uncertain",
  "lag": 1,
  "confidence": 0.0,
  "evidence_score": 0.0
}
```

图的作用不是单纯可视化，而是作为可验证叙事的中间表示。它将自然语言事件、目标变量、滞后关系和证据分数连接起来，为后续 narrative 和 groundedness 评价提供基础。

### 5.6 Method 3：Event consistency verifier

Verifier 是 CEN-TS 的核心贡献之一。它用于判断事件解释是否与时间序列变化一致。

当前 score 定义为：

```text
final_consistency_score =
  0.4 * direction_score
  + 0.3 * lag_score
  + 0.3 * counterfactual_score
```

其中：

- `direction_score`：事件极性与目标变量变化方向是否一致。
- `lag_score`：事件后 lag window 中变化幅度是否明显。
- `counterfactual_score`：观察到的变化是否超过随机窗口 baseline。

这种设计的意义是：LLM 或规则抽取产生的事件不能直接相信，必须通过数值序列变化进行反向验证。

### 5.7 Full CEN-TS Feature Mode

完整 feature mode 流程：

```text
text
  -> event extraction
  -> event graph
  -> verifier filtering
  -> event count / polarity / confidence / verified score features
  -> forecasting model
```

当前 `full_cents_feature` 与 `event_with_verifier` 使用同一组 verified event features，因此结果一致。下一阶段需要将 graph-level features、top-k event chain、LLM narrative embedding 等进一步加入 Full CEN-TS。

## 6. 实验设计

### 6.1 主任务

给定历史窗口：

```text
X_{t-L+1:t}, D_{t-L+1:t}
```

预测：

```text
y_{t+1:t+H}
```

其中：

- `X` 为数值序列。
- `D` 为时间对齐文本。
- `y` 默认为 `OT`。
- `L=24`。
- `H in {3, 6}`。
- 按时间顺序切分 train / validation / test，不随机打乱。

### 6.2 实验领域

主实验使用自动选择的 3 个文本覆盖较高领域：

- Energy
- Health_US
- Health_AFR

### 6.3 模型矩阵

当前已跑通：

| 方法 | 说明 |
|---|---|
| persistence | 最后观测值延续 |
| numerical_only | 数值窗口 ridge predictor |
| raw_text_embedding | TF-IDF/SVD 文本向量拼接 |
| event_no_verifier | 结构化事件特征，不过滤 |
| event_with_verifier | 结构化事件 + consistency verifier |
| full_cents_feature | 完整 CEN-TS feature mode |

### 6.4 消融实验

当前 ablation 聚合结果：

| 方法 | MSE | MAE | Event Consistency |
|---|---:|---:|---:|
| persistence | 0.2631 | 0.2396 | 0.0000 |
| numerical_only | 1.0038 | 0.5651 | 0.0000 |
| raw_text_embedding | 3.0621 | 0.9582 | 0.0000 |
| event_no_verifier | 1.1382 | 0.5844 | 0.4339 |
| event_with_verifier | 1.1167 | 0.5950 | 0.4339 |
| full_cents_feature | 1.1167 | 0.5950 | 0.4339 |

关键对比：

- event_with_verifier 相比 raw_text_embedding：MSE 降低约 **63.5%**。
- event_with_verifier 相比 numerical_only：MSE 高约 **11.2%**。
- persistence 仍明显更强，说明当前短期预测任务中序列自相关非常强。

### 6.5 鲁棒性实验

已实现扰动：

1. `shuffle_text_time`：打乱文本时间戳。
2. `random_text_injection`：注入无关文本。

平均 robustness drop：

```text
overall drop = 0.1616
```

按领域和 horizon 看：

| Domain | Horizon | Mean Drop |
|---|---:|---:|
| Energy | 3 | 0.3619 |
| Energy | 6 | 0.4503 |
| Health_AFR | 3 | -0.1073 |
| Health_AFR | 6 | -0.0929 |
| Health_US | 3 | 0.0058 |
| Health_US | 6 | -0.0515 |

解释：

- Energy 中文本扰动造成明显性能下降，说明 Energy 文本事件与价格变化具有较强关联。
- Health_AFR / Health_US 中扰动影响不稳定，说明文本信号较弱、事件抽取粗糙或目标序列本身结构不同。
- 这支持一个更细粒度结论：CEN-TS 不是所有领域都自动有效，它需要文本-数值关系足够强、事件抽取足够准、lag 设定足够合理。

## 7. 当前实验结果解读

### 7.1 Raw text embedding 的失败很有信息量

raw_text_embedding 的平均 MSE 为 `3.0621`，远高于 numerical-only 和 event methods。这个结果说明，在当前设置下，直接文本融合不是一个稳健方案。

这支持 CEN-TS 的出发点：文本需要先转化为更可控、更稀疏、更结构化的中间表示。否则模型容易把无关文本、泛化背景和预测性语言一并吸收，导致过拟合或噪声放大。

### 7.2 Event+verifier 相比 raw text 有明显优势

event_with_verifier 的平均 MSE 为 `1.1167`，比 raw_text_embedding 低约 `63.5%`。这说明，即使用非常简单的规则事件抽取，结构化事件特征也比原始文本向量更稳。

这并不意味着当前 CEN-TS 已经达到最终效果，而是说明“结构化事件 + 验证”这个方向有可继续投入的信号。

### 7.3 Event+verifier 还没有稳定超过 numerical-only

numerical_only 的平均 MSE 为 `1.0038`，略优于 event_with_verifier 的 `1.1167`。这说明当前事件特征还没有给数值预测带来稳定增益。

可能原因：

1. 当前规则事件抽取过粗，只能识别方向词，无法理解复杂因果机制。
2. lag 统一设为 1 到 3，未针对不同领域自适应。
3. event feature 过于简单，只包含 count、pos/neg、confidence、verified score。
4. ridge window predictor 表达能力有限。
5. 文本事件可能已经被数值历史窗口间接反映。

### 7.4 Persistence 很强，说明任务设置需要扩展

persistence 的 MSE 为 `0.2631`，显著优于其他方法。短期平滑序列中，最后值延续经常是很强 baseline。要证明 CEN-TS 的真正价值，下一阶段应考虑：

- 更长 horizon。
- 事件冲击窗口。
- 趋势转折点预测。
- OOD split。
- 高波动区间。
- 极端事件子集。
- event-triggered evaluation。

否则在短期平滑预测上，复杂文本方法很难击败 persistence。

### 7.5 Granger-lite 结果支持数值层滞后结构存在

Energy 中多个区域汽油价格变量对 `OT` 的 lag-1 Granger-lite 边显著，例如：

| Source | Target | Lag | p-value | Effect |
|---|---|---:|---:|---:|
| East Coast gasoline price | OT | 1 | 9.52e-08 | 0.993 |
| Lower Atlantic gasoline price | OT | 1 | 4.65e-07 | 0.990 |
| Central Atlantic gasoline price | OT | 1 | 2.05e-04 | 0.993 |

这说明未来可以把 Augur-lite 从简单 lagged correlation 升级到更正式的 Granger / PCMCI / transfer entropy。

### 7.6 LLM 事件抽取路径已经打通，但小样本未产生事件

OpenAI API key 被检测到，`gpt-4o-mini` 小样本调用成功：

```json
{
  "domain": "Energy",
  "target_variable": "OT",
  "max_rows": 1,
  "api_key_present": true,
  "model": "gpt-4o-mini",
  "attempted": true,
  "success": true,
  "llm_calls": 1,
  "events": [],
  "error": ""
}
```

模型在当前小样本上没有输出事件。这是一个真实负结果，可能说明：

- prompt 太保守。
- 示例文本更像数值报告，而非明确事件。
- schema 对“价格变化报告”是否算事件没有给出足够指引。
- 需要 few-shot examples。
- 需要领域特定事件类型和抽取规则。

下一阶段应重点优化 LLM extraction prompt，并使用验证集错误反馈做 Narrative-APO。

## 8. Case Study 解读

当前 case study 主要来自 Energy。典型事件是汽油价格在 2022 年 3 月快速上涨：

```text
The national average retail regular gasoline price increased to $4.102 per gallon on March 7, 2022...
```

规则抽取结果：

- polarity：positive
- target：OT
- lag：1-3
- confidence：约 0.7
- verifier score：约 0.897

这类案例说明，CEN-TS 的 verifier 能够识别“文本描述上涨，随后数值确实上涨”的一致模式。它的优势不是生成花哨叙事，而是让每条叙事都能追溯到：

- 原始文本
- 结构化事件
- event-to-variable edge
- lag window
- consistency score
- forecast context

这正是“verifiable causal event narrative”的核心。

## 9. 失败分析

### 9.1 规则事件抽取过粗

当前规则抽取依赖关键词和方向词，会把一些普通趋势描述也当作事件。例如“价格上涨”可能只是观测结果，而不是导致未来变化的外生事件。

改进方向：

- 区分 observation、forecast、external event、policy shock。
- 增加事件类别层次。
- 加入 few-shot LLM extraction。
- 使用领域 schema，例如 Energy 中区分 supply shock、demand shock、crude oil price、refinery capacity、seasonality。

### 9.2 Lag 判断粗糙

当前 lag 基本统一为 1 到 3。不同领域的事件影响周期显然不同：

- Energy：可能按周影响。
- Economy：可能按月或季度影响。
- Health：传播和报告延迟可能有多周。
- Traffic：事件可能即时或短期影响。

下一步应从数据频率、事件类型和历史验证结果中学习 lag distribution。

### 9.3 Forecast backbone 太弱

当前使用 ridge window predictor 是为了快速打通闭环，但它不是 A 会级别 baseline。下一步必须接入：

- DLinear
- PatchTST
- iTransformer
- TimesNet
- TimeMixer
- MM-TSFlib 原始设置

尤其需要比较：

```text
strong numerical-only backbone
vs
strong raw text fusion
vs
strong event-verified CEN-TS
```

### 9.4 GAMETime 数据未完整获得

GAMETime clone 仓库不包含完整 benchmark 数据。当前只做 sanity check，不能声称完成 GAMETime 实验。下一步需要按其 README 获取完整数据，再单独验证 event inference consistency。

### 9.5 Persistence 过强

当前短 horizon 平滑预测中 persistence 非常强。未来需要加入更能体现文本事件价值的评估：

- 事件冲击窗口。
- 转折点预测。
- 高波动窗口。
- OOD split。
- 长 horizon。
- policy shock / market shock 子集。

## 10. 当前科研判断

### 10.1 当前结果是否支持假设？

部分支持。

支持点：

- event+verifier 明显优于 raw text embedding，MSE 降低约 63.5%。
- Energy 领域文本扰动会导致性能下降，说明文本事件信息确实被模型利用。
- event consistency verifier 能在 case study 中筛出方向一致事件。
- Granger-lite 发现数值变量之间存在强滞后结构，支持因果图方向。
- LLM extraction、GAMETime sanity、MM-TSFlib scanning 都已接入实验报告。

不支持或尚未证明的点：

- event+verifier 尚未稳定超过 numerical-only。
- persistence 在当前任务上仍远强于所有学习模型。
- LLM extraction 当前未产生有效事件。
- GAMETime 完整 benchmark 尚未获得。
- 统计显著性和强 baseline 尚不足。

因此，当前结论应写成：

> 第一阶段实验支持“可验证事件结构优于原始文本向量融合”这一子假设，但尚未证明完整 CEN-TS 在预测 MSE 上超过强数值模型。该方向最明确的价值目前体现在鲁棒性、解释可追溯性和事件一致性验证上。

### 10.2 是否值得继续？

值得继续，但下一阶段必须补强实验标准。

如果目标是 A 会/顶刊，不能只停留在轻量 ridge baseline 和规则事件抽取。需要把当前系统升级成：

- 强 backbone；
- 强 LLM event extraction；
- 严格 causal edge verification；
- 更好的 event-triggered evaluation；
- 多领域统计显著性；
- 人工或 LLM-as-judge 解释评价。

## 11. 下一阶段实验计划

### Priority 1：接入强 baseline

安装 PyTorch，并基于 MM-TSFlib 跑：

- DLinear
- PatchTST
- iTransformer
- TimesNet / TimeMixer

对齐：

- 相同 train/val/test split。
- 相同 horizon。
- 相同 seeds。
- 相同 domains。

### Priority 2：强化 LLM 事件抽取

改进 prompt：

- 加 few-shot examples。
- 区分 observation 和 causal event。
- 加领域 schema。
- 要求输出 “是否外生事件”。
- 要求给出可检验变量和 lag 依据。

增加小规模人工检查集，评估：

- extraction precision
- polarity accuracy
- target variable accuracy
- lag plausibility

### Priority 3：强化 causal discovery

当前 Granger-lite 是简化版。下一步应加入：

- statsmodels Granger causality
- PCMCI
- transfer entropy
- bootstrap confidence interval
- multiple testing correction

### Priority 4：强化 verifier

当前 verifier 使用简单 pre/post mean delta。下一步应加入：

- matched counterfactual windows
- seasonal matched windows
- bootstrap significance
- placebo event test
- shuffled event timestamp test
- event-type-specific lag test

### Priority 5：扩展评价任务

除了普通 MSE/MAE，还应重点评估：

- directional accuracy
- trend turning point F1
- event-window MSE
- high-volatility-window MSE
- robustness under text shuffle
- robustness under irrelevant text injection
- explanation groundedness
- narrative faithfulness

### Priority 6：GAMETime 完整验证

按照 GAMETime README 获取完整数据后，单独跑：

- event inference consistency
- curve-to-event sanity
- event-to-curve verification
- CEN-TS verifier calibration

### Priority 7：论文图表准备

需要准备：

- 方法总览图。
- event-variable causal graph 示例。
- verifier scoring 示意图。
- main benchmark 表。
- ablation 表。
- robustness 表。
- case study 图。
- failure taxonomy。

## 12. 结论

CEN-TS 第一阶段已经完成了一个可复现、可运行、可扩展的实验系统。它目前最重要的成果不是“已经打败所有 baseline”，而是建立了一条可验证研究路径：

```text
文本不是直接拼接，而是先变成事件；
事件不是直接相信，而是进入因果图；
因果图不是只做解释，而是用时间序列变化验证；
通过验证的事件再进入预测与叙事。
```

当前结果表明：

- 原始文本 embedding 融合在该设置下不稳。
- 结构化事件 + verifier 明显优于 raw text embedding。
- 但事件方法尚未稳定超过 numerical-only 和 persistence。
- 因此下一阶段应聚焦强 baseline、强事件抽取和更严格 verifier。

从科研判断上看，这个方向仍然值得继续推进。它的潜在论文价值不只在预测误差，而在于把 text-paired time series forecasting 从黑箱文本融合推进到**可验证、可追溯、可反事实检验的事件因果叙事预测框架**。

