from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tats_cen"))

from cen_ts.api_config import APIConfig
from cen_ts.p3a_pipeline import write_json


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()
    p2 = read_json("results/v6/p2/p2_status.json")
    probe = read_json("results/v6/p3a/api_probe.json")
    usage = read_json("results/v6/p3a/api_usage.json")
    audit = read_json("results/v6/p3a/input_audit.json")
    sample = read_json("results/v6/p3a/pilot_sample_manifest.json")
    stability = read_json("results/v6/p3a/extraction_stability.json")
    estimate = read_json("results/v6/p3a/full_extraction_estimate.json")
    queue_manifest = read_json("results/v6/p3a/manual_annotation_queue_manifest.json")
    api = APIConfig.from_env()
    upstream_status = subprocess.check_output(
        ["git", "-c", f"safe.directory={(ROOT / 'third_party/TaTS').resolve()}", "-C", str(ROOT / "third_party/TaTS"), "status", "--short"],
        text=True, encoding="utf-8",
    ).strip()
    upstream_clean = not upstream_status
    expected_hash = read_json("results/v6/p2/p1b_parity.json")["tats_cen_itransformer_sha256"]
    itransformer_unchanged = sha256(ROOT / "tats_cen/models/iTransformer.py") == expected_hash
    prompt_hash = sha256(ROOT / "prompts/v6/event_extraction/p_extract_v1.txt")
    schema_hash = sha256(ROOT / "prompts/v6/event_extraction/schema_v1.json")
    hard_failures = []
    if not probe["api_probe_success"]:
        hard_failures.append("api_probe_failed")
    if usage["api_calls_total_ledger"] <= 0:
        hard_failures.append("zero_llm_calls")
    if usage["official_validation_calls"] or usage["official_test_calls"]:
        hard_failures.append("forbidden_split_calls")
    if not upstream_clean or not itransformer_unchanged:
        hard_failures.append("forecast_backbone_modified")
    partial_reasons = []
    if usage["pilot_unique_texts"] < 100:
        partial_reasons.append("default API budget limited pilot to fewer than 100 unique texts")
    if queue_manifest["multi_event_shortfall"]:
        partial_reasons.append("only 8 unique multi-event texts were available for the requested 10-row queue category")
    gates = {
        "json_parse_rate": usage["json_parse_rate"] >= 0.95,
        "schema_valid_rate": usage["schema_valid_rate"] >= 0.95,
        "grounding_valid_rate": usage["grounding_valid_rate"] >= 0.90,
        "temporal_valid_rate": usage["temporal_valid_rate"] >= 0.95,
    }
    partial_reasons.extend(f"quality gate not met: {name}" for name, passed in gates.items() if not passed)
    if audit["target_semantics"]["status"] != "resolved":
        partial_reasons.append("target semantics unresolved")
    if not args.tests_passed:
        partial_reasons.append("automatic tests not passed")
    status = "FAIL" if hard_failures else "PARTIAL" if partial_reasons else "PASS"
    full = estimate["estimates"]["full_dataset"]
    status_payload = {
        "p2_commit": "2bd8b526cf5d0069da842628382847d9680987df",
        "upstream_clean": upstream_clean,
        "itransformer_unchanged": itransformer_unchanged,
        "api_probe_success": probe["api_probe_success"],
        "extractor_model": api.model,
        "llm_calls": usage["api_calls_total_ledger"],
        "cache_hits": usage["cache_hits"],
        "cache_misses": usage["cache_misses"],
        "pilot_unique_texts": usage["pilot_unique_texts"],
        "json_parse_rate": usage["json_parse_rate"],
        "schema_valid_rate": usage["schema_valid_rate"],
        "grounding_valid_rate": usage["grounding_valid_rate"],
        "temporal_valid_rate": usage["temporal_valid_rate"],
        "average_events_per_text": usage["average_events_per_text"],
        "no_event_rate": usage["no_event_rate"],
        "unknown_direction_rate": usage["unknown_direction_rate"],
        "unknown_lag_rate": usage["unknown_lag_rate"],
        "stability_event_type_jaccard": stability["event_type_raw_jaccard"],
        "input_tokens": usage["total_input_tokens"],
        "output_tokens": usage["total_output_tokens"],
        "api_retries": usage["api_retries_total_ledger"],
        "estimated_full_unique_calls": full["estimated_unique_requests"],
        "estimated_full_input_tokens": full["estimated_input_tokens"],
        "estimated_full_output_tokens": full["estimated_output_tokens"],
        "estimated_full_cost": full["estimated_cost"],
        "manual_annotation_completed": False,
        "official_validation_calls": usage["official_validation_calls"],
        "official_test_calls": usage["official_test_calls"],
        "event_tats_training_runs": usage["event_tats_training_runs"],
        "tests_passed": bool(args.tests_passed),
        "quality_gates": gates,
        "hard_failures": hard_failures,
        "partial_reasons": partial_reasons,
        "p3a_status": status,
    }
    write_json(ROOT / "results/v6/p3a/p3a_status.json", status_payload)
    branch = git("branch", "--show-current")
    base_commit = git("rev-parse", "HEAD")
    report = f"""# CEN-TaTS v6 P3A：真实 LLM 事件抽取 Pilot 报告

## 1. 执行摘要

P3A 使用真实 OpenAI-compatible API、严格 `event_schema_v1`、本地缓存、证据与时间验证以及确定性 renderer。未运行完整 Environment 抽取、Event-TaTS 训练或 M3-M5。

## 2. P3A_STATUS

`P3A_STATUS={status}`。硬失败：{hard_failures or '无'}。PARTIAL 原因：{partial_reasons or '无'}。

## 3. Git 分支与 commit

- 分支：`{branch}`
- P2 基座：`2bd8b526cf5d0069da842628382847d9680987df`
- 报告生成时 HEAD：`{base_commit}`
- P3A 最终提交 SHA 由提交完成后的 `git rev-parse HEAD` 记录（commit 不能自包含自身 SHA）。

## 4. P2 parity 继承情况

P2 状态 `{p2['p2_status']}`，parity `{p2['p2_parity']}`；本阶段未重跑正式训练。

## 5. TaTS 预测基座是否未修改

`third_party/TaTS clean={upstream_clean}`；`tats_cen/models/iTransformer.py unchanged={itransformer_unchanged}`。

## 6. API 配置与脱敏状态

- provider: openai_compatible
- base URL: `{api.base_url}`
- model: `{api.model}`
- API key 仅做非空检查，未写入报告、manifest 或日志。
- 调用预算：{api.max_calls} calls / {api.max_input_tokens} input tokens / {api.max_output_tokens} output tokens。

## 7. API probe

Probe 成功：`{probe['api_probe_success']}`；样例数 {probe['sample_count']}；token usage 可用：`{probe['token_usage_available']}`；request ID 可用：`{probe['request_ids_available']}`。

## 8. Environment 文本审计

总行数 {audit['total_rows']}，非空率 {audit['text_nonempty_rate']:.6f}，标准化唯一文本 {audit['normalized_unique_nonempty_texts']}，平均字符 {audit['average_characters']:.2f}，平均 GPT-2 token {audit['average_gpt2_tokens']}。

## 9. 数据和 target 语义

`OT` 已由本地 Time-MMD 材料解析为日频 Air Quality Index（AQI index points）。没有联网补充语义。

## 10. train_core/prompt_dev 边界

边界：`{audit['split_boundaries']}`。LLM 调用仅来自 train_core/prompt_dev；official validation/test 调用均为 0。

## 11. EventRecord schema

Schema `event_schema_v1`，SHA256 `{schema_hash}`。强制枚举、confidence 0-1、lag 顺序、非空 evidence 与确定性 event ID。

## 12. Prompt v1

Prompt `p_extract_v1`，SHA256 `{prompt_hash}`；严格 JSON、原文证据、禁止外部知识，最多 5 个事件。

## 13. Prompt 修改记录

未创建 v2；未运行 APO。详见 `reports/v6/p3a/prompt_change_log.md`。

## 14. API client

标准库 HTTP 实现支持 chat/completions 与 responses 风格、timeout、至多 3 次可恢复错误重试、指数退避和一次 schema/JSON repair；无规则 fallback。

## 15. 缓存机制

cache hit/miss = {usage['cache_hits']}/{usage['cache_misses']}。Key 覆盖 provider/base URL hash/model/temperature/prompt/schema/source/report time/target description。完整原文和 API body 只保存在 ignored 本地 cache。

## 16. Pilot 采样

候选样本 {sample['selected']['total']} 个唯一文本；受默认 API 预算限制，实际 primary pilot 为 {usage['pilot_unique_texts']} 个唯一文本，split 分布 `{usage['selected_split_counts']}`。

## 17. 调用统计

LLM calls={usage['api_calls_total_ledger']}，retries={usage['api_retries_total_ledger']}，input/output tokens={usage['total_input_tokens']}/{usage['total_output_tokens']}，平均/P50/P95 latency={usage['average_latency_seconds']}/{usage['p50_latency_seconds']}/{usage['p95_latency_seconds']} 秒。

## 18. JSON 和 schema 结果

JSON parse rate={usage['json_parse_rate']:.6f}；schema validity rate={usage['schema_valid_rate']:.6f}；失败分布 `{usage['failure_type_distribution']}`。

## 19. evidence grounding

exact={usage['grounding_exact_rate']:.6f}，normalized-exact={usage['grounding_normalized_exact_rate']:.6f}，valid={usage['grounding_valid_rate']:.6f}，unsupported={usage['unsupported_evidence_rate']:.6f}。unsupported 不进入 narrative。

## 20. temporal consistency

Temporal validity rate={usage['temporal_valid_rate']:.6f}；未来 event 被标 observed 时本地验证会置 invalid。

## 21. 事件类型与 factuality 分布

平均事件数={usage['average_events_per_text']:.6f}，no-event rate={usage['no_event_rate']:.6f}，forecast={usage['forecast_rate']:.6f}，observed={usage['observed_rate']:.6f}，unknown direction={usage['unknown_direction_rate']:.6f}，unknown lag={usage['unknown_lag_rate']:.6f}。

## 22. 稳定性诊断

完成 {stability['completed_pairs']}/{stability['requested_samples']} 对重复请求；event count consistency={stability['event_count_consistency_rate']:.6f}，event type Jaccard={stability['event_type_raw_jaccard']:.6f}，action match={stability['action_normalized_match_rate']:.6f}，evidence overlap={stability['evidence_span_overlap']:.6f}。

## 23. event narrative renderer

Renderer 不调用 LLM，固定英文模板，明确 unknown，最多 5 个事件，只接收 grounding/temporal 有效事件。

## 24. Pilot CSV 一致性

`data/v6/p3a/Environment_event_pilot.csv` 的原始全部列（含 date、OT、prior_history_avg、fact）hash identity=`{usage['pilot_csv_numeric_identity']}`；非处理行 event_fact 为空且 status=not_processed。

## 25. 人工 annotation queue

`results/v6/p3a/manual_annotation_queue.csv` 已生成 40 个唯一候选。高置信、相对低置信与失败/边界候选均达到 10；primary pilot 仅有 8 个真正多事件文本，因此多事件类别短缺 2 条，未用单事件样本冒充。

## 26. 无人工标注的限制

人工 review 列保持空白，因此本阶段不能声称 precision、recall、F1 或正式事件抽取准确率。

## 27. 全量调用和 token 估计

Full dataset 预计 unique requests={full['estimated_unique_requests']}，API calls={full['estimated_api_calls']}，input/output tokens={full['estimated_input_tokens']}/{full['estimated_output_tokens']}，费用={full['estimated_cost']}。

## 28. 已知失败案例

失败记录位于 `results/v6/p3a/pilot_failures.jsonl`，retry queue 位于 `results/v6/p3a/retry_queue.jsonl`；没有以规则或原文替代失败结果。

## 29. 自动测试

`tests_passed={bool(args.tests_passed)}`；测试命令为固定解释器执行 `python -m pytest tests/v6 -q`。

## 30. 是否允许进入 P3B 全量事件抽取

`{'否' if status != 'PASS' else '仅在用户明确确认预算后允许'}`。本阶段不会自动进入 P3B。

## 31. P3B 前需要用户确认的预算

需用户明确确认至少 {full['estimated_api_calls']} 次预计 API 调用、{full['estimated_input_tokens']} input tokens、{full['estimated_output_tokens']} output tokens；费用因未提供本地单价而为 `{full['estimated_cost']}`。
"""
    path = ROOT / "reports/v6/p3a/event_extraction_pilot_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(json.dumps({"p3a_status": status, "partial_reasons": partial_reasons, "hard_failures": hard_failures}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
