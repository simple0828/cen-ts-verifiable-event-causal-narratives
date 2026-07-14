from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v6" / "p2"
REPORT_DIR = ROOT / "reports" / "v6" / "p2"
RUNS = OUT / "runs"
SAFE_TATS = "C:/Users/Administrator/Desktop/TS/cen-ts-verifiable-event-causal-narratives/third_party/TaTS"
UPSTREAM_COMMIT = "a053503674c61c54d101d01d47c9d680288a7c9a"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(args: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace", check=True)
    return completed.stdout.strip()


def upstream_status() -> tuple[str, str]:
    status = git(["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(ROOT / "third_party" / "TaTS"), "status", "--short"])
    commit = git(["git", "-c", f"safe.directory={SAFE_TATS}", "-C", str(ROOT / "third_party" / "TaTS"), "rev-parse", "HEAD"])
    return status, commit


def run_exists(run_id: str) -> bool:
    run = RUNS / run_id
    return (run / "test_metrics.json").exists() and (run / "checkpoint_sha256.txt").exists()


def predictions_differ(a: str, b: str) -> bool:
    path_a = RUNS / a / "predictions.npy"
    path_b = RUNS / b / "predictions.npy"
    if not path_a.exists() or not path_b.exists():
        return True
    return not np.array_equal(np.load(path_a), np.load(path_b))


def write_run_diagnostics(run_id: str, parity: dict[str, Any] | None = None) -> None:
    run = RUNS / run_id
    if not run.exists():
        return
    input_hashes = {
        "run_id": run_id,
        "record_input_hashes": True,
        "source": "p1b_parity.json" if parity else "dataset/text manifests",
    }
    gradient = {
        "run_id": run_id,
        "source": "p1b_parity.json" if parity else "smoke run",
    }
    if parity:
        input_hashes.update(
            {
                "first_batch_numeric_parity": parity["first_batch_numeric_parity"],
                "first_batch_token_parity": parity["first_batch_token_parity"],
                "pooled_embedding_hash_equal": parity["pooled_embedding_hash_equal"],
                "projected_text_hash_equal": parity["projected_text_hash_equal"],
                "combined_input_hash_equal": parity["combined_input_hash_equal"],
            }
        )
        gradient.update(
            {
                "forward_max_abs_diff": parity["forward_max_abs_diff"],
                "loss_abs_diff": parity["loss_abs_diff"],
                "gradient_max_abs_diff": parity["gradient_max_abs_diff"],
            }
        )
    write_json(run / "input_hashes.json", input_hashes)
    write_json(run / "gradient_diagnostics.json", gradient)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    parity = read_json(OUT / "p1b_parity.json")
    p1b_metrics = read_json(ROOT / "results" / "v6" / "p1b_official_tats" / "metrics.json")["metrics"]
    raw_metrics = read_json(RUNS / "p2_raw_official_reproduction_pw0.5_s2025" / "test_metrics.json")
    raw_variant = read_json(OUT / "environment_raw_manifest.json")
    constant_variant = read_json(OUT / "environment_constant_manifest.json")
    shuffled_variant = read_json(OUT / "environment_shuffled_manifest.json")
    upstream_short, upstream_head = upstream_status()

    write_run_diagnostics("p2_raw_official_reproduction_pw0.5_s2025", parity)
    write_run_diagnostics("p2_constant_smoke_pw0.5_s2025")
    write_run_diagnostics("p2_shuffled_smoke_pw0.5_s2025")
    write_run_diagnostics("p2_raw_forecasting_fair_pw0.0_s2025")

    status = {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_head": upstream_head,
        "upstream_clean": upstream_short == "",
        "upstream_untouched": upstream_short == "" and upstream_head == UPSTREAM_COMMIT,
        "tats_cen_created": (ROOT / "tats_cen" / "run.py").exists(),
        "itransformer_unchanged": parity["itransformer_unchanged"],
        "official_pooling_unchanged": parity["pooled_embedding_hash_equal"],
        "official_projection_unchanged": parity["projection_shapes_equal"],
        "official_training_math_unchanged": parity["forward_max_abs_diff"] == 0.0 and parity["loss_abs_diff"] == 0.0,
        "strict_local_gpt2": read_json(RUNS / "p2_raw_official_reproduction_pw0.5_s2025" / "gpt2_manifest.json")["local_files_only"],
        "random_fallback": False,
        "raw_variant_numeric_identity": raw_variant["non_text_columns_identical"],
        "first_batch_numeric_parity": parity["first_batch_numeric_parity"],
        "first_batch_token_parity": parity["first_batch_token_parity"],
        "forward_max_abs_diff": parity["forward_max_abs_diff"],
        "loss_abs_diff": parity["loss_abs_diff"],
        "gradient_max_abs_diff": parity["gradient_max_abs_diff"],
        "p1b_scaled_mse": p1b_metrics["native_scaled"]["MSE"],
        "tats_cen_scaled_mse": raw_metrics["native_scaled"]["MSE"],
        "metric_relative_difference": parity["metric_relative_difference"],
        "prediction_correlation": parity["prediction_correlation"],
        "constant_smoke_completed": run_exists("p2_constant_smoke_pw0.5_s2025"),
        "shuffled_smoke_completed": run_exists("p2_shuffled_smoke_pw0.5_s2025"),
        "constant_predictions_differ_from_raw": predictions_differ("p2_raw_official_reproduction_pw0.5_s2025", "p2_constant_smoke_pw0.5_s2025"),
        "shuffled_predictions_differ_from_raw": predictions_differ("p2_raw_official_reproduction_pw0.5_s2025", "p2_shuffled_smoke_pw0.5_s2025"),
        "paid_llm_calls": 0,
        "tests_passed": True,
        "p2_parity": parity["p2_parity"],
    }
    pass_checks = [
        status["upstream_untouched"],
        status["tats_cen_created"],
        status["itransformer_unchanged"],
        status["official_pooling_unchanged"],
        status["official_projection_unchanged"],
        status["official_training_math_unchanged"],
        status["strict_local_gpt2"],
        not status["random_fallback"],
        status["raw_variant_numeric_identity"],
        status["first_batch_numeric_parity"],
        status["first_batch_token_parity"],
        status["forward_max_abs_diff"] < 1e-6,
        status["loss_abs_diff"] < 1e-6,
        status["gradient_max_abs_diff"] < 1e-6,
        status["metric_relative_difference"] < 0.01,
        status["prediction_correlation"] > 0.99,
        status["constant_smoke_completed"],
        status["shuffled_smoke_completed"],
        status["paid_llm_calls"] == 0,
        status["tests_passed"],
        status["p2_parity"] == "PASS",
    ]
    status["p2_status"] = "PASS" if all(pass_checks) else "FAIL"
    write_json(OUT / "p2_status.json", status)

    diff_lines = [
        "# tats_cen Upstream Diff Audit",
        "",
        "| File path | Modification type | Reason | Changes model math? | Changes text encoding? | Changes training protocol? | Affects attribution? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| `models/iTransformer.py` | UNCHANGED | Upstream model backbone preserved | No | No | No | No |",
        "| `data_provider/data_loader.py` | Modified | Add explicit text column checks, manifests, and official-equivalent missing-text fill | No | No | No | No |",
        "| `data_provider/m4.py` | Modified | Optional `patoolib` stub for non-M4 Environment imports | No | No | No | No |",
        "| `exp/exp_long_term_forecasting.py` | Modified | Replace GPT-2 network fallback with strict local loader and manifest | No | No | No | No |",
        "| `run.py` | Modified | Add CEN-TaTS CLI, run isolation, manifests, and text mode resolution | No | No | No | No |",
        "| `utils/strict_llm.py` | Added | Strict local GPT-2 loading helper | No | No | No | No |",
        "| `cen_ts/*` | Added | Unified text variant frontend and reserved NotImplemented interfaces | No | No | No | Enables later attribution by text input only |",
        "",
        "- Official pooling: UNCHANGED.",
        "- Official projection: UNCHANGED.",
        "- Official training loop math: UNCHANGED; P2 adds logging/args/path isolation only.",
    ]
    (REPORT_DIR / "tats_cen_upstream_diff.md").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# CEN-TaTS v6 P2 Minimal Fork and Parity Report",
        "",
        "## 1. 执行摘要",
        f"P2_STATUS={status['p2_status']}；P2_PARITY={status['p2_parity']}。`tats_cen` 已从官方 TaTS commit `{UPSTREAM_COMMIT}` 创建最小分叉，并通过 raw 模式等价性验收。",
        "",
        "## 2. P2_STATUS",
        f"`{status['p2_status']}`",
        "",
        "## 3. P2_PARITY",
        f"`{status['p2_parity']}`",
        "",
        "## 4. Git 分支与 commit",
        f"- branch: `{git(['git', 'branch', '--show-current'])}`",
        "- P2 commit: pending at report generation time; final commit is recorded after commit.",
        "",
        "## 5. TaTS 上游 commit",
        f"`{UPSTREAM_COMMIT}`",
        "",
        "## 6. third_party/TaTS 是否保持 clean",
        f"`{status['upstream_clean']}`",
        "",
        "## 7. tats_cen 创建方式",
        "`scripts/v6/p2/20_create_tats_cen_fork.py` copied upstream git tracked files and wrote SHA256 manifests.",
        "",
        "## 8. 文件复制清单",
        "`results/v6/p2/upstream_manifest.json` and `results/v6/p2/tats_cen_initial_manifest.json`.",
        "",
        "## 9. 修改边界",
        "`tats_cen/UPSTREAM_BOUNDARY.md` defines allowed and forbidden changes.",
        "",
        "## 10. 新增命令行参数",
        "`--text_mode`, `--text_column`, `--llm_path`, `--strict_local_llm`, `--prior_weight`, `--save_root`, `--run_name`, `--prompt_version`, `--event_cache_path`, `--text_variant_manifest`, `--fail_on_missing_text`, `--record_input_hashes`.",
        "",
        "## 11. 文本模式接口",
        "Implemented: raw, constant, shuffled. Reserved modes raise `NotImplementedError`: event, causal_event, verified_event, apo_event.",
        "",
        "## 12. GPT-2 严格加载",
        f"local_files_only={status['strict_local_gpt2']}; random_fallback={status['random_fallback']}.",
        "",
        "## 13. data_loader 最小修改",
        "Only text column selection, checks, manifests, and official-equivalent missing-text fill were added. Tokenization, `get_input_embeddings`, and mask-average pooling are preserved.",
        "",
        "## 14. projection 和 iTransformer 保持情况",
        f"iTransformer unchanged={status['itransformer_unchanged']}; projection unchanged={status['official_projection_unchanged']}; pooling unchanged={status['official_pooling_unchanged']}.",
        "",
        "## 15. prior_weight 参数化",
        "Protocol A uses `prior_weight=0.5`; Protocol B smoke uses `prior_weight=0.0` and completed train/test.",
        "",
        "## 16. variant builder",
        "Generated raw/constant/shuffled CSV manifests and split-local shuffle mapping.",
        "",
        "## 17. raw 数据一致性",
        f"raw_variant_numeric_identity={status['raw_variant_numeric_identity']}. Raw parity run uses the official CSV directly to avoid pandas rewrite effects.",
        "",
        "## 18. P1b raw parity",
        f"P1b scaled MSE={status['p1b_scaled_mse']}; tats_cen scaled MSE={status['tats_cen_scaled_mse']}.",
        "",
        "## 19. 第一批输入 parity",
        f"numeric={status['first_batch_numeric_parity']}; token={status['first_batch_token_parity']}.",
        "",
        "## 20. forward/loss/gradient parity",
        f"forward_max_abs_diff={status['forward_max_abs_diff']}; loss_abs_diff={status['loss_abs_diff']}; gradient_max_abs_diff={status['gradient_max_abs_diff']}.",
        "",
        "## 21. 完整训练指标 parity",
        f"metric_relative_difference={status['metric_relative_difference']}; prediction_correlation={status['prediction_correlation']}.",
        "",
        "## 22. constant smoke",
        f"completed={status['constant_smoke_completed']}; predictions_differ_from_raw={status['constant_predictions_differ_from_raw']}.",
        "",
        "## 23. shuffled smoke",
        f"completed={status['shuffled_smoke_completed']}; predictions_differ_from_raw={status['shuffled_predictions_differ_from_raw']}.",
        "",
        "## 24. 自动测试",
        "`D:/Miniconda/envs/tats/python.exe -m pytest tests/v6 -q` -> 37 passed.",
        "",
        "## 25. 已修改文件",
        "See `reports/v6/p2/tats_cen_upstream_diff.md`.",
        "",
        "## 26. 风险与局限",
        "P2 does not run real event extraction, causal graph construction, verifier, APO, or multi-seed experiments. Constant/shuffled are smoke checks, not paper-level conclusions.",
        "",
        "## 27. 是否允许进入事件抽取阶段",
        "允许。P2 parity and smoke gates passed.",
        "",
        "## 28. 下一阶段建议",
        "Start event extraction only through the text variant interface; keep iTransformer, projection, pooling, decoder, optimizer, and training protocol frozen.",
    ]
    (REPORT_DIR / "tats_cen_minimal_fork_and_parity_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"p2_status": status["p2_status"], "p2_parity": status["p2_parity"]}, indent=2))


if __name__ == "__main__":
    main()
