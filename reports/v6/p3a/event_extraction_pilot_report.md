# CEN-TaTS v6 P3A：真实 LLM 事件抽取 Pilot 报告

## 1. 执行摘要

P3A 使用真实 OpenAI-compatible API、严格 `event_schema_v1`、本地缓存、证据与时间验证以及确定性 renderer。未运行完整 Environment 抽取、Event-TaTS 训练或 M3-M5。

## 2. P3A_STATUS

`P3A_STATUS=PARTIAL`。硬失败：无。PARTIAL 原因：['default API budget limited pilot to fewer than 100 unique texts', 'only 8 unique multi-event texts were available for the requested 10-row queue category', 'quality gate not met: temporal_valid_rate']。

## 3. Git 分支与 commit

- 分支：`feature/cen-event-extractor-pilot-v6`
- P2 基座：`2bd8b526cf5d0069da842628382847d9680987df`
- 报告生成时 HEAD：`2bd8b526cf5d0069da842628382847d9680987df`
- P3A 最终提交 SHA 由提交完成后的 `git rev-parse HEAD` 记录（commit 不能自包含自身 SHA）。

## 4. P2 parity 继承情况

P2 状态 `PASS`，parity `PASS`；本阶段未重跑正式训练。

## 5. TaTS 预测基座是否未修改

`third_party/TaTS clean=True`；`tats_cen/models/iTransformer.py unchanged=True`。

## 6. API 配置与脱敏状态

- provider: openai_compatible
- base URL: `https://api.qingyuntop.top/v1`
- model: `gpt-4.1-mini`
- API key 仅做非空检查，未写入报告、manifest 或日志。
- 调用预算：80 calls / 400000 input tokens / 120000 output tokens。

## 7. API probe

Probe 成功：`True`；样例数 2；token usage 可用：`True`；request ID 可用：`True`。

## 8. Environment 文本审计

总行数 15248，非空率 0.984654，标准化唯一文本 2080，平均字符 152.22，平均 GPT-2 token 28.847275875849206。

## 9. 数据和 target 语义

`OT` 已由本地 Time-MMD 材料解析为日频 Air Quality Index（AQI index points）。没有联网补充语义。

## 10. train_core/prompt_dev 边界

边界：`{'official_test': [12199, 15248], 'official_validation': [10673, 12199], 'prompt_dev': [9072, 10673], 'train_core': [0, 9072]}`。LLM 调用仅来自 train_core/prompt_dev；official validation/test 调用均为 0。

## 11. EventRecord schema

Schema `event_schema_v1`，SHA256 `070454eb731441c712d73f9844c30b6d6f932a386c0ec2261aec0a730c80afa5`。强制枚举、confidence 0-1、lag 顺序、非空 evidence 与确定性 event ID。

## 12. Prompt v1

Prompt `p_extract_v1`，SHA256 `acbee070ae8a954b3bf5cf4b19bfd6a3e2d885200c2618987cf7f8f86500fc19`；严格 JSON、原文证据、禁止外部知识，最多 5 个事件。

## 13. Prompt 修改记录

未创建 v2；未运行 APO。详见 `reports/v6/p3a/prompt_change_log.md`。

## 14. API client

标准库 HTTP 实现支持 chat/completions 与 responses 风格、timeout、至多 3 次可恢复错误重试、指数退避和一次 schema/JSON repair；无规则 fallback。

## 15. 缓存机制

cache hit/miss = 2/52。Key 覆盖 provider/base URL hash/model/temperature/prompt/schema/source/report time/target description。完整原文和 API body 只保存在 ignored 本地 cache。

## 16. Pilot 采样

候选样本 200 个唯一文本；受默认 API 预算限制，实际 primary pilot 为 54 个唯一文本，split 分布 `{'prompt_dev': 11, 'train_core': 43}`。

## 17. 调用统计

LLM calls=73，retries=0，input/output tokens=69046/12622，平均/P50/P95 latency=4.044635927443867/3.6023230000864714/8.526319700002205 秒。

## 18. JSON 和 schema 结果

JSON parse rate=1.000000；schema validity rate=1.000000；失败分布 `{'internal_budget_ledger_error': 1}`。

## 19. evidence grounding

exact=1.000000，normalized-exact=0.000000，valid=1.000000，unsupported=0.000000。unsupported 不进入 narrative。

## 20. temporal consistency

Temporal validity rate=0.738095；未来 event 被标 observed 时本地验证会置 invalid。

## 21. 事件类型与 factuality 分布

平均事件数=0.792453，no-event rate=0.396226，forecast=0.047619，observed=0.952381，unknown direction=0.571429，unknown lag=1.000000。

## 22. 稳定性诊断

完成 20/20 对重复请求；event count consistency=1.000000，event type Jaccard=0.800000，action match=0.800000，evidence overlap=1.000000。

## 23. event narrative renderer

Renderer 不调用 LLM，固定英文模板，明确 unknown，最多 5 个事件，只接收 grounding/temporal 有效事件。

## 24. Pilot CSV 一致性

`data/v6/p3a/Environment_event_pilot.csv` 的原始全部列（含 date、OT、prior_history_avg、fact）hash identity=`True`；非处理行 event_fact 为空且 status=not_processed。

## 25. 人工 annotation queue

`results/v6/p3a/manual_annotation_queue.csv` 已生成 40 个唯一候选。高置信、相对低置信与失败/边界候选均达到 10；primary pilot 仅有 8 个真正多事件文本，因此多事件类别短缺 2 条，未用单事件样本冒充。

## 26. 无人工标注的限制

人工 review 列保持空白，因此本阶段不能声称 precision、recall、F1 或正式事件抽取准确率。

## 27. 全量调用和 token 估计

Full dataset 预计 unique requests=2003，API calls=2003，input/output tokens=1895428/368474，费用=unavailable。

## 28. 已知失败案例

失败记录位于 `results/v6/p3a/pilot_failures.jsonl`，retry queue 位于 `results/v6/p3a/retry_queue.jsonl`；没有以规则或原文替代失败结果。

## 29. 自动测试

`tests_passed=True`；测试命令为固定解释器执行 `python -m pytest tests/v6 -q`。

## 30. 是否允许进入 P3B 全量事件抽取

`否`。本阶段不会自动进入 P3B。

## 31. P3B 前需要用户确认的预算

需用户明确确认至少 2003 次预计 API 调用、1895428 input tokens、368474 output tokens；费用因未提供本地单价而为 `unavailable`。
