from __future__ import annotations


def textual_gradient(failures: list[dict]) -> str:
    issues = []
    if any(f.get("schema_errors") for f in failures):
        issues.append("当前提示仍会产生 schema 或字段缺失错误。")
    if any(f.get("ungrounded") for f in failures):
        issues.append("当前提示需要更强地要求 evidence_span 逐字来自输入文本。")
    if any(f.get("cycle_mismatch") for f in failures):
        issues.append("当前提示在事件影响尚不可观察时过度依赖序列一致性。")
    if any(f.get("forecast_error_high") for f in failures):
        issues.append("当前提示应优先保留与目标变量和预测窗口 lag 相交的事件。")
    if not issues:
        issues.append("当前提示表现稳定，只需收紧事实性和叙事忠实性。")
    return "\n".join(issues)
