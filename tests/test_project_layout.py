import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from cen_ts.paths import ROOT, TATS_ROOT, UPSTREAM_ROOT, UPSTREAM_COMMIT, gpt2_path


def load_initializer():
    spec = importlib.util.spec_from_file_location("prepare_vendor", ROOT / "scripts/v6/p2/20_prepare_vendor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_business_package_has_one_active_home():
    import cen_ts
    assert ROOT / "src/cen_ts/__init__.py" == Path(cen_ts.__file__)
    assert not (ROOT / "src/cen_tats").exists()
    assert not (ROOT / "src/cents").exists()
    assert not (ROOT / "tats_cen").exists()
    assert not (TATS_ROOT / "cen_ts").exists()


def test_all_vendored_models_match_fixed_upstream():
    manifest = json.loads((TATS_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == UPSTREAM_COMMIT
    assert "models/iTransformer.py" in manifest["model_blob_sha256"]
    for relative, expected in manifest["model_blob_sha256"].items():
        data = (TATS_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected, relative


def test_repeated_initializer_preserves_existing_business_and_adapters(tmp_path):
    manifest = json.loads((TATS_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    files = ["cen_ts/custom.py", "run.py", *manifest["model_blob_sha256"]]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing user code\n")
    before = {name: (tmp_path / name).read_bytes() for name in files}
    initializer = load_initializer()
    for _ in range(2):
        assert initializer.prepare_vendor(tmp_path, tmp_path / "absent-upstream")["restored"] == []
    assert {name: (tmp_path / name).read_bytes() for name in files} == before


def test_initializer_restores_models_from_pinned_commit(tmp_path):
    import pytest
    if not (UPSTREAM_ROOT / ".git").exists():
        pytest.skip("Fixed upstream checkout is unavailable")
    result = load_initializer().prepare_vendor(tmp_path)
    manifest = json.loads((TATS_ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    assert set(result["restored"]) == set(manifest["model_blob_sha256"])
    for name, digest in manifest["model_blob_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert load_initializer().prepare_vendor(tmp_path)["restored"] == []


def test_gpt2_path_works_outside_project_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CEN_TS_GPT2_PATH", "models/local-gpt2")
    assert gpt2_path() == ROOT / "models/local-gpt2"
    assert gpt2_path("models/explicit") == ROOT / "models/explicit"


def test_stage_dispatch_works_outside_project_cwd(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "p2-run", "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--llm_path" in result.stdout
