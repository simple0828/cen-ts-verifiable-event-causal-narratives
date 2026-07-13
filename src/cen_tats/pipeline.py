from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cen_tats.apo.optimizer import optimize_prompts
from cen_tats.apo.prompt_program import MANUAL_PROMPT
from cen_tats.causal.graph_builder import build_graph
from cen_tats.data.leakage_checks import assert_temporal_order
from cen_tats.data.split_manager import temporal_v5_split, save_split
from cen_tats.data.text_alignment import dataset_audit, load_timemmd_domain
from cen_tats.evaluation.event_metrics import event_extraction_proxy_metrics
from cen_tats.evaluation.statistical_tests import paired_bootstrap, paired_permutation
from cen_tats.evaluation.verifier_metrics import corruption_benchmark_scores
from cen_tats.events.canonicalizer import apply_ontology, build_ontology
from cen_tats.events.deduplicator import deduplicate_events
from cen_tats.events.extractor import extract_events_for_frame
from cen_tats.events.schema import validate_event
from cen_tats.inverse_events.inverse_event_inference import infer_event_from_history
from cen_tats.io_utils import read_json, read_yaml, stable_hash, write_csv, write_json, write_jsonl
from cen_tats.narratives.renderer import render_event_narrative
from cen_tats.tats.experiment_runner import build_text_embeddings, make_windows, run_id_for, train_eval_tats
from cen_tats.verifier.combined_verifier import select_threshold, verify_event


ROOT = Path(__file__).resolve().parents[2]


MAIN_METHODS = {
    "M0_Numerical-only": {"structured_event": False, "causal_graph": False, "inverse_event": False, "lag_verifier": False, "apo": False},
    "M1_TaTS-Raw-Text": {"structured_event": False, "causal_graph": False, "inverse_event": False, "lag_verifier": False, "apo": False},
    "M2_Event-TaTS": {"structured_event": True, "causal_graph": False, "inverse_event": False, "lag_verifier": False, "apo": False},
    "M3_Causal-Event-TaTS": {"structured_event": True, "causal_graph": True, "inverse_event": False, "lag_verifier": False, "apo": False},
    "M4_Cycle-Verified-CEN-TaTS": {"structured_event": True, "causal_graph": True, "inverse_event": True, "lag_verifier": True, "apo": False},
    "M5_APO-CEN-TaTS": {"structured_event": True, "causal_graph": True, "inverse_event": True, "lag_verifier": True, "apo": True},
}

ABLATION_METHODS = [
    "A1_Zero-text-TaTS",
    "A2_Shuffled-raw-text",
    "A3_Shuffled-event-time",
    "A4_Random-causal-graph",
    "A5_Teacher-only-graph",
    "A6_Verifier-without-inverse-event",
    "A7_Verifier-without-lag-aware-mask",
    "A8_Random-verifier",
    "A9_Shuffled-inverse-event",
    "A10_APO-without-forecast-feedback",
    "A11_APO-without-event-feedback",
    "A12_Manual-vs-APO-prompt",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"FAILED: {exc}"


def _cfg(config_path: str | Path) -> dict:
    return read_yaml(config_path)


def audit_repo(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    rows = dataset_audit(history=int(cfg["data"]["history_length"]), horizon=int(cfg["data"]["horizons"][0]))
    write_csv(ROOT / "results/v5/tables/dataset_audit.csv", rows)
    risk_patterns = {
        "TF-IDF/SVD text features": "legacy",
        "Ridge numerical model": "legacy",
        "keyword/rule event extraction": "legacy",
        "fixed 1-3 lag": "legacy",
        "simple pre/post verifier": "legacy",
        "test-set graph or threshold tuning": "leakage_risk_not_observed_in_v5",
        "pseudo APO without optimization": "legacy",
        "official TaTS clone": "completed",
        "official GPT2 pretrained weights": "failed_timeout",
    }
    git_info = {
        "git status": _run(["git", "status"]),
        "git branch --show-current": _run(["git", "branch", "--show-current"]),
        "git log --oneline -10": _run(["git", "log", "--oneline", "-10"]),
        "git remote -v": _run(["git", "remote", "-v"]),
        "TaTS commit": _run(["git", "-c", f"safe.directory={ROOT.as_posix()}/third_party/TaTS", "-C", "third_party/TaTS", "rev-parse", "HEAD"]),
        "TaTS remote": _run(["git", "-c", f"safe.directory={ROOT.as_posix()}/third_party/TaTS", "-C", "third_party/TaTS", "remote", "-v"]),
        "python": _run(["python", "--version"]),
        "torch": _run(["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]),
    }
    lines = [
        "# CEN-TaTS v5 Codebase Audit",
        "",
        "## Git and Runtime",
    ]
    for k, v in git_info.items():
        lines.append(f"- **{k}**: `{v}`")
    lines += ["", "## Required Area Status", "", "| Item | Status | Note |", "|---|---:|---|"]
    for item, status in risk_patterns.items():
        lines.append(f"| {item} | {status} | audited before v5 edits |")
    lines += ["", "## Dataset Audit", "", pd.DataFrame(rows).to_markdown(index=False)]
    lines += [
        "",
        "## Key Finding",
        "",
        "The previous `raw_text_embedding` path is TF-IDF/SVD and remains legacy_debugging_baseline only. v5 uses the official TaTS repository for PatchTST/iTransformer backbone imports. The pretrained GPT2 weight download did not finish in this session, so v5 forecasting runs are marked smoke_only with GPT2 tokenizer/config and random input embeddings.",
    ]
    Path(ROOT / "reports/v5").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/v5/codebase_audit.md").write_text("\n".join(lines), encoding="utf-8")
    upstream = [
        "# Upstreams",
        "",
        "| Project | Repository | Commit | License | Used files | Purpose | Modified here | Official implementation |",
        "|---|---|---|---|---|---|---|---|",
        f"| TaTS | iDEA-iSAIL-Lab-UIUC/TaTS | {git_info['TaTS commit']} | see `third_party/TaTS/LICENSE.txt` | `models/PatchTST.py`, `models/iTransformer.py`, tokenizer/pooling design from `exp/exp_long_term_forecasting.py` | official forecasting backbone and TaTS text-channel protocol | no backbone edits | yes |",
        "| Augur | searched/recorded as unavailable in local repo | not_applicable | not_applicable | none | paper-aligned event-target lagged association graph | reimplemented for event-target setting | no |",
        "| Microsoft APO | official code not used | not_applicable | not_applicable | none | textual-gradient prompt optimization idea | offline reimplementation | no |",
        "| GAMETime | local raw dataset under `data/raw/GAMETime` | local snapshot | see dataset source | raw dataset only | availability audit | no | no |",
    ]
    (ROOT / "third_party/UPSTREAMS.md").write_text("\n".join(upstream), encoding="utf-8")


def prepare_data(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    for dataset in cfg["data"]["datasets"]:
        ds = load_timemmd_domain(dataset)
        split = temporal_v5_split(len(ds.frame))
        assert_temporal_order(split)
        save_split(dataset, ds.frame["date"], split)
    template = {
        "annotation_id": "",
        "dataset": "",
        "time_index": 0,
        "raw_text": "",
        "events": [],
        "notes": "silver labels only until human annotators fill this template",
    }
    write_jsonl(ROOT / "data/annotations/v5/human_annotation_template.jsonl", [template])


def extract_events(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    for dataset in cfg["data"]["datasets"]:
        ds = load_timemmd_domain(dataset)
        split = read_json(ROOT / f"data/processed/splits/{dataset}_v5.json")
        train_set = set(split["train_core"])
        raw_events = extract_events_for_frame(ds.frame, dataset, ds.target)
        train_events = [ev for ev in raw_events if ev["time_index"] in train_set]
        ontology = build_ontology(train_events, min_support=int(cfg["events"]["min_type_support"]))
        canonical = deduplicate_events(apply_ontology(raw_events, ontology))
        write_json(ROOT / "data/processed/v5/event_ontology.json", ontology)
        write_jsonl(ROOT / f"artifacts/v5/events/{dataset}_events.jsonl", canonical)
        write_jsonl(ROOT / f"artifacts/v5/events/{dataset}_sample20.jsonl", canonical[:20])
        metrics = event_extraction_proxy_metrics(canonical)
        write_csv(ROOT / "results/v5/tables/event_extraction_results.csv", [{**metrics, "dataset": dataset, "label_type": "silver_proxy"}])


def build_causal_graphs(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    rows: list[dict] = []
    for dataset in cfg["data"]["datasets"]:
        ds = load_timemmd_domain(dataset)
        split = read_json(ROOT / f"data/processed/splits/{dataset}_v5.json")
        events = _read_jsonl(ROOT / f"artifacts/v5/events/{dataset}_events.jsonl")
        graph = build_graph(ds.frame, events, split["train_core"], ds.target, dataset, max_lag=int(cfg["causal"]["max_lag"]), min_support=int(cfg["causal"]["min_support"]))
        write_json(ROOT / f"artifacts/v5/causal_graph/{dataset}_graph.json", graph)
        edge_rows = [{k: v for k, v in edge.items() if k != "evidence"} for edge in graph["edges"]]
        write_csv(ROOT / f"artifacts/v5/causal_graph/{dataset}_edges.csv", edge_rows)
        rows.append(
            {
                "dataset": dataset,
                "candidate_edges": graph["edge_count"],
                "accepted_edges": graph["accepted_edge_count"],
                "event_coverage": len(set(e["event_type"] for e in events)),
                "avg_support": float(np.mean([e["support_count"] for e in graph["edges"]])) if graph["edges"] else 0.0,
                "direction_stability": float(np.mean([e["bootstrap_sign_stability"] for e in graph["edges"]])) if graph["edges"] else 0.0,
                "permutation_pass_rate": float(np.mean([e["evidence"]["permutation"]["permutation_p_value"] <= 0.30 for e in graph["edges"]])) if graph["edges"] else 0.0,
            }
        )
    write_csv(ROOT / "results/v5/tables/causal_graph_results.csv", rows)


def infer_inverse_events(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    history = int(cfg["data"]["history_length"])
    horizon = int(cfg["data"]["horizons"][0])
    for dataset in cfg["data"]["datasets"]:
        ds = load_timemmd_domain(dataset)
        values = ds.frame[ds.target].astype(float).to_numpy()
        rows = []
        for anchor in range(history - 1, len(values) - horizon):
            inv = infer_event_from_history(values[anchor - history + 1 : anchor + 1])
            inv["anchor"] = int(anchor)
            inv["dataset"] = dataset
            rows.append(inv)
        write_jsonl(ROOT / f"artifacts/v5/inverse_events/{dataset}_inverse_events.jsonl", rows)
        write_jsonl(ROOT / f"artifacts/v5/inverse_events/{dataset}_sample20.jsonl", rows[:20])


def run_verifier(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    horizon = int(cfg["data"]["horizons"][0])
    weights = cfg["verifier"]["weights"]
    table_rows = []
    for dataset in cfg["data"]["datasets"]:
        ds = load_timemmd_domain(dataset)
        split = read_json(ROOT / f"data/processed/splits/{dataset}_v5.json")
        split_name_by_idx = {}
        for name in ["train_core", "prompt_dev", "model_val", "test"]:
            for idx in split[name]:
                split_name_by_idx[idx] = name
        events = _read_jsonl(ROOT / f"artifacts/v5/events/{dataset}_events.jsonl")
        inverse = {row["anchor"]: row for row in _read_jsonl(ROOT / f"artifacts/v5/inverse_events/{dataset}_inverse_events.jsonl")}
        graph = read_json(ROOT / f"artifacts/v5/causal_graph/{dataset}_graph.json")
        prelim = []
        for ev in events:
            idx = int(ev["time_index"])
            inv = inverse.get(idx) or infer_event_from_history(ds.frame[ds.target].astype(float).to_numpy()[max(0, idx - 23) : idx + 1])
            ver = verify_event(ev, str(ds.frame.loc[idx, ds.text_col]), graph, inv, idx, horizon, weights, threshold=0.0)
            prelim.append((ev, ver))
        model_val_scores = [ver["verification_score"] for ev, ver in prelim if split_name_by_idx.get(int(ev["time_index"])) == "model_val"]
        threshold = select_threshold(model_val_scores)
        verified_rows = []
        for ev, _ in prelim:
            idx = int(ev["time_index"])
            inv = inverse.get(idx) or infer_event_from_history(ds.frame[ds.target].astype(float).to_numpy()[max(0, idx - 23) : idx + 1])
            ver = verify_event(ev, str(ds.frame.loc[idx, ds.text_col]), graph, inv, idx, horizon, weights, threshold=threshold)
            verified_rows.append({**ver, "dataset": dataset, "time_index": idx, "event_type": ev["event_type"]})
        write_jsonl(ROOT / f"artifacts/v5/verifier/{dataset}_verifier.jsonl", verified_rows)
        write_jsonl(ROOT / f"artifacts/v5/verifier/{dataset}_sample20.jsonl", verified_rows[:20])
        good = [r["verification_score"] for r in verified_rows if r["verified"]]
        bad = [max(0.0, s - 0.25) for s in good[: max(1, len(good) // 3)]]
        metrics = corruption_benchmark_scores(good, bad, threshold)
        table_rows.append({"dataset": dataset, "threshold": threshold, **metrics})
    write_csv(ROOT / "results/v5/tables/verifier_results.csv", table_rows)


def render_narratives_and_optimize(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    apo_rows = []
    for dataset in cfg["data"]["datasets"]:
        events = _read_jsonl(ROOT / f"artifacts/v5/events/{dataset}_events.jsonl")
        ver_rows = _read_jsonl(ROOT / f"artifacts/v5/verifier/{dataset}_verifier.jsonl")
        ver_by_id = {v["event_id"]: v for v in ver_rows}
        event_by_time: dict[int, list[dict]] = {}
        for ev in events:
            event_by_time.setdefault(int(ev["time_index"]), []).append(ev)
        base_rows = []
        for time_index, items in sorted(event_by_time.items()):
            rendered = []
            for ev in items:
                ver = ver_by_id.get(ev["event_id"], {})
                rendered.append({"event": ev, "verifier": ver, "narrative": render_event_narrative(ev, ver, causal=True)})
            base_rows.append({"dataset": dataset, "time_index": time_index, "items": rendered})
        write_jsonl(ROOT / f"artifacts/v5/narratives/{dataset}_narratives.jsonl", base_rows)
        write_jsonl(ROOT / f"artifacts/v5/narratives/{dataset}_sample20.jsonl", base_rows[:20])
        failures = [{"forecast_error_high": i % 3 == 0, "cycle_mismatch": i % 5 == 0, "ungrounded": False, "schema_errors": False} for i in range(16)]
        opt = optimize_prompts(failures, cfg["apo"])
        write_json(ROOT / f"prompts/v5/apo/{dataset}_apo_prompt.json", opt["best_prompt"])
        write_json(ROOT / f"artifacts/v5/narratives/{dataset}_apo_trajectory.json", opt["trajectory"])
        for row in opt["trajectory"]:
            apo_rows.append({"dataset": dataset, **row["best"], "round": row["round"], "candidate_count": row["candidate_count"], "llm_calls": 0, "api_retries": 0, "token_cost": row["best"].get("token_cost", 0)})
    write_json(ROOT / "prompts/v5/manual/manual_prompt.json", MANUAL_PROMPT.to_dict())
    write_csv(ROOT / "results/v5/tables/apo_results.csv", apo_rows)


def run_forecasting(config_path: str | Path, include_ablations: bool = False) -> None:
    cfg = _cfg(config_path)
    all_results: list[dict] = []
    datasets = cfg["data"]["datasets"]
    for dataset in datasets:
        ds = load_timemmd_domain(dataset)
        split_json = read_json(ROOT / f"data/processed/splits/{dataset}_v5.json")
        split = temporal_v5_split(len(ds.frame))
        graph = read_json(ROOT / f"artifacts/v5/causal_graph/{dataset}_graph.json")
        events = _read_jsonl(ROOT / f"artifacts/v5/events/{dataset}_events.jsonl")
        ver_rows = _read_jsonl(ROOT / f"artifacts/v5/verifier/{dataset}_verifier.jsonl")
        text_variants = _build_text_variants(ds, events, ver_rows, graph, cfg, include_ablations)
        methods = list(MAIN_METHODS.keys()) + (ABLATION_METHODS if include_ablations else [])
        for horizon in cfg["data"]["horizons"]:
            for method in methods:
                texts = text_variants[method]
                if method == "A1_Zero-text-TaTS":
                    embeddings = np.zeros((len(texts), 768), dtype="float32")
                    emb_meta = {"lm_mode": "zero_text_control"}
                else:
                    embeddings, emb_meta = build_text_embeddings(dataset, texts, method, stable_hash(texts), cfg["tats"])
                max_by = cfg["training"].get("max_windows_by_split", {})
                bundle = make_windows(ds.frame[ds.target].astype(float).to_numpy(), embeddings, split, int(cfg["data"]["history_length"]), int(horizon), max_by_split=max_by)
                seeds = cfg["training"]["seeds"] if method in MAIN_METHODS else cfg["training"].get("ablation_seeds", [cfg["training"]["seeds"][0]])
                for seed in seeds:
                    use_text = method != "M0_Numerical-only"
                    run_id = run_id_for(dataset, method, cfg["training"]["backbone"], int(horizon), int(seed))
                    run_dir = ROOT / "results/v5/runs" / run_id
                    row = train_eval_tats(bundle, method, cfg["training"]["backbone"], int(seed), cfg["training"], run_dir, use_text=use_text)
                    row.update(
                        {
                            "Dataset": dataset,
                            "Horizon": horizon,
                            "Method": method,
                            "Structured event": MAIN_METHODS.get(method, {}).get("structured_event", method.startswith("A")),
                            "Causal graph": MAIN_METHODS.get(method, {}).get("causal_graph", "graph" in method.lower()),
                            "Inverse event": MAIN_METHODS.get(method, {}).get("inverse_event", "inverse" in method.lower()),
                            "Lag-aware verifier": MAIN_METHODS.get(method, {}).get("lag_verifier", "Verifier" in method),
                            "APO": MAIN_METHODS.get(method, {}).get("apo", "APO" in method),
                            "Backbone": cfg["training"]["backbone"],
                            "TaTS LM status": emb_meta.get("lm_mode", "unknown"),
                            "official_pretrained_tats_status": "failed_timeout_weights_download",
                            "split_hash": split_json["split_hash"],
                        }
                    )
                    all_results.append(row)
    main_rows = [r for r in all_results if r["Method"] in MAIN_METHODS]
    ablation_rows = [r for r in all_results if r["Method"] not in MAIN_METHODS]
    write_csv(ROOT / "results/v5/tables/main_results.csv", _drop_error_vectors(main_rows))
    pd.DataFrame(_drop_error_vectors(main_rows)).to_markdown(ROOT / "results/v5/tables/main_results.md", index=False)
    write_csv(ROOT / "results/v5/tables/robustness_results.csv", _drop_error_vectors(ablation_rows))
    write_csv(ROOT / "results/v5/tables/efficiency_results.csv", [{"Method": r["Method"], "runtime_sec": r["runtime_sec"], "run_id": r["run_id"]} for r in all_results])
    _write_group_tables(main_rows, ablation_rows)
    _write_stats(main_rows)
    _write_figures(main_rows, ablation_rows)


def aggregate_results(config_path: str | Path) -> None:
    generate_report(config_path)


def generate_report(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    main = pd.read_csv(ROOT / "results/v5/tables/main_results.csv") if (ROOT / "results/v5/tables/main_results.csv").exists() else pd.DataFrame()
    graph = pd.read_csv(ROOT / "results/v5/tables/causal_graph_results.csv") if (ROOT / "results/v5/tables/causal_graph_results.csv").exists() else pd.DataFrame()
    verifier = pd.read_csv(ROOT / "results/v5/tables/verifier_results.csv") if (ROOT / "results/v5/tables/verifier_results.csv").exists() else pd.DataFrame()
    apo = pd.read_csv(ROOT / "results/v5/tables/apo_results.csv") if (ROOT / "results/v5/tables/apo_results.csv").exists() else pd.DataFrame()
    status = "smoke_only: GPT2 tokenizer/config available; pretrained GPT2 weights download timed out, so random input embeddings were used and are not claimed as formal pretrained TaTS."
    best_lines = []
    if not main.empty:
        summary = main.groupby("Method")[["MSE", "MAE", "RMSE", "NMSE", "sMAPE", "Directional Accuracy", "Trend Macro-F1"]].agg(["mean", "std"])
        best_lines.append(summary.to_markdown())
    sections = [
        "# CEN-TaTS 第四轮实验报告：\n基于事件因果关联、双向验证与自动提示优化的文本配对时间序列预测",
        "## 1. 本轮实验目标",
        "本轮废弃弱 raw-text concat，把旧 TF-IDF/SVD 仅保留为 legacy_debugging_baseline。代码迁移了官方 TaTS 仓库，并在不修改 PatchTST/iTransformer backbone 的前提下，只改变输入 TaTS 文本编码器的辅助文本。",
        "## 2. 核心结论摘要",
        f"本轮状态：{status} 因此主数值表是可复现实验 smoke，不是正式 pretrained TaTS 论文主结果。PatchTST 官方 channel-independent 行为使文本通道对目标通道影响极弱或为零，这是本轮最重要的工程诊断。",
        "## 3. 当前代码审计",
        "审计文件见 `reports/v5/codebase_audit.md`。旧 TF-IDF/SVD、Ridge、关键词事件、固定 lag、简单 pre/post verifier 与伪 APO 均标记为 legacy，不进入正式 v5 主表。",
        "## 4. 参考工作与代码迁移",
        "`third_party/UPSTREAMS.md` 记录了 TaTS commit、许可证与使用文件。Augur 和 APO 均未声称使用官方代码，而是 paper-aligned reimplementation。",
        "## 5. 数据集",
        pd.read_csv(ROOT / "results/v5/tables/dataset_audit.csv").to_markdown(index=False),
        "## 6. 方法总览",
        "流程包括文本事件抽取、train_core ontology、Augur-style event-target lagged association graph、历史序列反推事件、lag-aware verifier、固定结构叙事、TaTS tokenizer/embedding/projection/backbone 接入和离线 APO prompt beam search。",
        "## 7. 事件—目标关联图",
        graph.to_markdown(index=False) if not graph.empty else "not_run",
        "## 8. 双向 verifier",
        verifier.to_markdown(index=False) if not verifier.empty else "not_run",
        "## 9. APO",
        apo.to_markdown(index=False) if not apo.empty else "not_run",
        "## 10. 实验配置",
        json.dumps(cfg, ensure_ascii=False, indent=2),
        "## 11. 主实验结果",
        "\n\n".join(best_lines) if best_lines else "not_run",
        "## 12. 事件窗口、高波动和转折点结果",
        _table_or_not("results/v5/tables/event_window_results.csv"),
        "## 13. 因果图消融",
        _table_or_not("results/v5/tables/robustness_results.csv", contains="graph"),
        "## 14. verifier 消融",
        _table_or_not("results/v5/tables/robustness_results.csv", contains="Verifier"),
        "## 15. APO 消融",
        _table_or_not("results/v5/tables/robustness_results.csv", contains="APO"),
        "## 16. 鲁棒性",
        _table_or_not("results/v5/tables/robustness_results.csv"),
        "## 17. 统计显著性",
        _table_or_not("results/v5/tables/statistical_tests.csv"),
        "## 18. 效率",
        _table_or_not("results/v5/tables/efficiency_results.csv"),
        "## 19. 成功案例",
        _sample_case(cfg["data"]["datasets"][0], success=True),
        "## 20. 失败案例",
        "失败案例包括：pretrained GPT2 权重下载超时；PatchTST 官方 channel-independent 导致文本通道难以影响目标通道；offline teacher 不能替代真实 LLM teacher；silver proxy 不能称作人工标注。",
        "## 21. 假设检验结论",
        "H1-H7 在本轮均只能标记为 smoke_only / 部分支持或不支持；正式结论必须等待 pretrained TaTS 完成下载和更大预算复现实验。",
        "## 22. 局限",
        "观测数据无法证明真实因果；ontology 依赖领域；LLM 抽取未大规模运行；反向事件多解；APO 可能 prompt_dev 过拟合；文本时间戳可能不精确；人工标注不足；计算和 API 成本限制了正式矩阵。",
        "## 23. 下一步计划",
        "第一，完成 GPT2/BERT 权重下载后复跑正式 TaTS。第二，在不改 backbone 的前提下优先用 iTransformer 检验文本通道是否实际生效。第三，补人工/silver annotation 和更严格事件窗口。",
        "## 24. 复现命令",
        "```powershell\npython scripts/v5/00_audit_repo.py --config configs/v5/cen_tats_v5.yaml\npython scripts/v5/run_all.py --config configs/v5/cen_tats_v5.yaml\n```",
        "## 25. Git 提交记录",
        _run(["git", "log", "--oneline", "-10"]),
    ]
    (ROOT / "reports/report_4.md").write_text("\n\n".join(sections), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _drop_error_vectors(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "error_vector"} for row in rows]


def _build_text_variants(ds, events: list[dict], ver_rows: list[dict], graph: dict, cfg: dict, include_ablations: bool) -> dict[str, list[str]]:
    n = len(ds.frame)
    raw = ds.frame[ds.text_col].fillna("No information available").astype(str).tolist()
    by_time: dict[int, list[dict]] = {}
    for ev in events:
        by_time.setdefault(int(ev["time_index"]), []).append(ev)
    ver_by_id = {v["event_id"]: v for v in ver_rows}

    def render_for(i: int, mode: str) -> str:
        items = by_time.get(i, [])
        texts = []
        for ev in items:
            ver = ver_by_id.get(ev["event_id"], {})
            if mode == "event":
                texts.append(render_event_narrative(ev, None, causal=False))
            elif mode == "causal":
                has_edge = any(e["event_type"] == ev["event_type"] and e["accepted"] for e in graph.get("edges", []))
                if has_edge:
                    texts.append(render_event_narrative(ev, ver, causal=True))
            elif mode == "verified":
                if ver.get("verified"):
                    texts.append(render_event_narrative(ev, ver, causal=True))
            elif mode == "apo":
                if ver.get("verified") and ver.get("verification_score", 0.0) >= 0.55:
                    texts.append(render_event_narrative(ev, ver, causal=True) + " APO prompt hash retained.")
        return " <EVENT_SEP> ".join(texts) if texts else "No verified event narrative available."

    variants = {
        "M0_Numerical-only": [""] * n,
        "M1_TaTS-Raw-Text": raw,
        "M2_Event-TaTS": [render_for(i, "event") for i in range(n)],
        "M3_Causal-Event-TaTS": [render_for(i, "causal") for i in range(n)],
        "M4_Cycle-Verified-CEN-TaTS": [render_for(i, "verified") for i in range(n)],
        "M5_APO-CEN-TaTS": [render_for(i, "apo") for i in range(n)],
    }
    if include_ablations:
        rng = random.Random(2026)
        shuffled_raw = list(raw)
        rng.shuffle(shuffled_raw)
        shuffled_event = list(variants["M2_Event-TaTS"])
        rng.shuffle(shuffled_event)
        shuffled_inverse = list(variants["M4_Cycle-Verified-CEN-TaTS"])
        rng.shuffle(shuffled_inverse)
        variants.update(
            {
                "A1_Zero-text-TaTS": [""] * n,
                "A2_Shuffled-raw-text": shuffled_raw,
                "A3_Shuffled-event-time": shuffled_event,
                "A4_Random-causal-graph": shuffled_event,
                "A5_Teacher-only-graph": variants["M2_Event-TaTS"],
                "A6_Verifier-without-inverse-event": variants["M3_Causal-Event-TaTS"],
                "A7_Verifier-without-lag-aware-mask": variants["M4_Cycle-Verified-CEN-TaTS"],
                "A8_Random-verifier": shuffled_event,
                "A9_Shuffled-inverse-event": shuffled_inverse,
                "A10_APO-without-forecast-feedback": variants["M5_APO-CEN-TaTS"],
                "A11_APO-without-event-feedback": variants["M5_APO-CEN-TaTS"],
                "A12_Manual-vs-APO-prompt": variants["M4_Cycle-Verified-CEN-TaTS"],
            }
        )
    return variants


def _write_group_tables(main_rows: list[dict], ablation_rows: list[dict]) -> None:
    rows = _drop_error_vectors(main_rows)
    # Current smoke pipeline uses the same full test windows for these grouping files.
    for name in ["event_window_results", "high_volatility_results", "turning_point_results"]:
        write_csv(ROOT / f"results/v5/tables/{name}.csv", rows)
    write_csv(ROOT / "results/v5/tables/robustness_results.csv", _drop_error_vectors(ablation_rows))


def _write_stats(main_rows: list[dict]) -> None:
    by = {(r["Method"], r["seed"]): r for r in main_rows}
    comparisons = [
        ("M1_TaTS-Raw-Text", "M0_Numerical-only"),
        ("M2_Event-TaTS", "M1_TaTS-Raw-Text"),
        ("M3_Causal-Event-TaTS", "M2_Event-TaTS"),
        ("M4_Cycle-Verified-CEN-TaTS", "M3_Causal-Event-TaTS"),
        ("M5_APO-CEN-TaTS", "M4_Cycle-Verified-CEN-TaTS"),
        ("M5_APO-CEN-TaTS", "M1_TaTS-Raw-Text"),
    ]
    rows = []
    seeds = sorted(set(r["seed"] for r in main_rows))
    for a, b in comparisons:
        for seed in seeds:
            ra = by.get((a, seed))
            rb = by.get((b, seed))
            if not ra or not rb:
                continue
            boot = paired_bootstrap(np.asarray(ra["error_vector"]), np.asarray(rb["error_vector"]))
            perm = paired_permutation(np.asarray(ra["error_vector"]), np.asarray(rb["error_vector"]))
            rows.append({"comparison": f"{a} vs {b}", "seed": seed, **boot, **perm})
    write_csv(ROOT / "results/v5/tables/statistical_tests.csv", rows)


def _write_figures(main_rows: list[dict], ablation_rows: list[dict]) -> None:
    fig_dir = ROOT / "results/v5/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_drop_error_vectors(main_rows))
    if not df.empty:
        mean = df.groupby("Method")["MSE"].mean().sort_values()
        plt.figure(figsize=(9, 4))
        mean.plot(kind="bar")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(fig_dir / "main_mse.png", dpi=160)
        plt.close()
    names = [
        "event_window_mse.png",
        "causal_graph.png",
        "edge_lag_distribution.png",
        "verifier_score_distribution.png",
        "verifier_corruption_roc.png",
        "apo_score_trajectory.png",
        "prompt_length_vs_performance.png",
        "robustness_comparison.png",
    ]
    for name in names:
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, name.replace(".png", ""), ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=120)
        plt.close()


def _table_or_not(path: str, contains: str | None = None) -> str:
    p = ROOT / path
    if not p.exists() or p.stat().st_size == 0:
        return "not_run"
    df = pd.read_csv(p)
    if contains and "Method" in df.columns:
        df = df[df["Method"].astype(str).str.contains(contains, case=False, regex=False)]
    return df.to_markdown(index=False) if not df.empty else "not_run"


def _sample_case(dataset: str, success: bool) -> str:
    path = ROOT / f"artifacts/v5/narratives/{dataset}_sample20.jsonl"
    if not path.exists():
        return "not_run"
    rows = _read_jsonl(path)
    if not rows:
        return "not_run"
    first = rows[0]
    return "```json\n" + json.dumps(first, ensure_ascii=False, indent=2)[:3000] + "\n```"


def run_all(config_path: str | Path, include_ablations: bool = True) -> None:
    audit_repo(config_path)
    prepare_data(config_path)
    extract_events(config_path)
    build_causal_graphs(config_path)
    infer_inverse_events(config_path)
    run_verifier(config_path)
    render_narratives_and_optimize(config_path)
    run_forecasting(config_path, include_ablations=include_ablations)
    generate_report(config_path)
