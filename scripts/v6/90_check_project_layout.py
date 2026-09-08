"""Audit this layout migration against its recorded v6 baseline (offline)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cen_ts.paths import TATS_ROOT, UPSTREAM_ROOT, UPSTREAM_COMMIT, project_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination(old: str) -> str | None:
    prefixes = (
        ("tats_cen/cen_ts/", "src/cen_ts/"),
        ("src/cen_tats/runtime/", "src/cen_ts/runtime/"),
        ("src/cen_tats/evaluation/forecast_metrics.py", "src/cen_ts/evaluation/forecast_metrics.py"),
        ("src/cen_tats/", "archive/legacy/src/cen_tats/"),
        ("src/cents/", "archive/legacy/src/cents/"),
        ("tats_cen/", "vendor/tats/"),
        ("scripts/v5/", "archive/legacy/scripts/v5/"),
        ("tests/v5/", "archive/legacy/tests/v5/"),
        ("legacy/", "archive/legacy/pre_v5/"),
    )
    for before, after in prefixes:
        if old.startswith(before):
            return after + old[len(before):]
    if old.startswith("configs/") and not old.startswith("configs/v6/"):
        return "archive/legacy/" + old
    if old.startswith("scripts/") and old.count("/") == 1 and old != "scripts/_bootstrap.py":
        return "archive/legacy/" + old
    if old.startswith("tests/") and old.count("/") == 1:
        return "archive/legacy/" + old
    return {
        "scripts/v6/p2/20_create_tats_cen_fork.py": "scripts/v6/p2/20_prepare_vendor.py",
        "scripts/v6/p2/22_run_tats_cen.py": "scripts/v6/p2/22_run_tats.py",
    }.get(old)


def audit() -> dict:
    baseline = json.loads((ROOT / "results/refactor/project-layout/baseline.json").read_text(encoding="utf-8"))
    commit = baseline["base_commit"]
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True).splitlines()
    mappings, unexpected_changes = [], []
    for old in tracked:
        new = destination(old)
        if new is None:
            continue
        path = ROOT / new
        original = subprocess.check_output(["git", "show", f"{commit}:{old}"], cwd=ROOT)
        same = path.exists() and path.read_bytes().replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n")
        mappings.append({"from": old, "to": new, "exists": path.exists(), "content_equal_lf": same})
        must_preserve = (new.startswith("archive/") or old.startswith("tats_cen/cen_ts/")
                         or any(old.startswith("tats_cen/" + part + "/") for part in ("models", "data", "data_provider", "exp", "layers", "utils"))
                         or old.endswith("evaluation/forecast_metrics.py"))
        if not path.exists() or (must_preserve and not same):
            unexpected_changes.append(new)
    protected_changes = [name for name, expected in baseline["protected_files"].items()
                         if not (ROOT / name).exists() or sha256(ROOT / name) != expected]
    model_manifest = json.loads((TATS_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    tracked_now = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    models = {}
    for relative, expected in model_manifest["model_blob_sha256"].items():
        path = TATS_ROOT / relative
        models[relative] = {
            "tracked": path.relative_to(ROOT).as_posix() in tracked_now,
            "pinned_source_equal": path.exists() and hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected,
        }
    import_violations, absolute_paths, syntax_errors = [], [], []
    # Legacy imports/path strings in this audit's mapping table are provenance.
    for folder in ("src/cen_ts", "scripts", "tests", "vendor/tats", "configs/v6"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in {".py", ".yaml", ".sh"} or ".local." in path.name or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            rel = path.relative_to(ROOT).as_posix()
            if re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]", source):
                absolute_paths.append(rel)
            if path.suffix == ".py":
                try:
                    tree = ast.parse(source, filename=rel)
                except SyntaxError as exc:
                    syntax_errors.append(f"{rel}:{exc.lineno}: {exc.msg}")
                    continue
                for node in ast.walk(tree):
                    names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                             else [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
                    if any(name.split(".")[0] in {"cen_tats", "cents"} for name in names):
                        import_violations.append(rel)
    git = ["git", "-c", f"safe.directory={UPSTREAM_ROOT.resolve().as_posix()}", "-C", str(UPSTREAM_ROOT)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    upstream_status = subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    obsolete_dirs = [name for name in ("tats_cen", "src/cents", "src/cen_tats", "vendor/tats/cen_ts") if (ROOT / name).exists()]
    result = {
        "base_commit": commit,
        "protected_file_count": len(baseline["protected_files"]),
        "protected_files_changed": protected_changes,
        "migration_mapping": mappings,
        "unexpected_migration_changes": unexpected_changes,
        "models": models,
        "old_active_directories": obsolete_dirs,
        "old_package_imports": import_violations,
        "hardcoded_machine_paths": absolute_paths,
        "syntax_errors": syntax_errors,
        "upstream_commit": head,
        "upstream_clean": not upstream_status,
    }
    result["passed"] = (not any((protected_changes, unexpected_changes, obsolete_dirs, import_violations, absolute_paths, syntax_errors))
                        and all(all(checks.values()) for checks in models.values())
                        and head == UPSTREAM_COMMIT and not upstream_status)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/refactor/project-layout/migration_audit.json")
    args = parser.parse_args()
    result = audit()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {"migration_mapping", "models"}}
    summary.update(migrated_files=len(result["migration_mapping"]), model_files=len(result["models"]))
    print(json.dumps(summary, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
