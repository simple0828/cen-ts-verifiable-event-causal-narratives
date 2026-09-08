"""Canonical entry point for v6 stages; forwards stage arguments unchanged."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "preflight": "00_preflight.py",
    "verify-gpt2": "01_verify_gpt2.py",
    "verify-cuda": "02_verify_cuda.py",
    "smoke-tats": "03_smoke_tats_itransformer.py",
    "p1-audit": "10_audit_p1_environment_data.py",
    "p1-inputs": "11_diagnose_p1_inputs.py",
    "p1-run": "12_run_p1_official_tats.py",
    "p1-status": "13_refresh_p1_status_after_tests.py",
    "p1b-run": "14_run_p1b_exact_official_tats.py",
    "p2-prepare": "p2/20_prepare_vendor.py",
    "p2-variant": "p2/21_build_text_variant.py",
    "p2-run": "p2/22_run_tats.py",
    "p2-parity": "p2/23_compare_p1b_parity.py",
    "p2-report": "p2/24_generate_p2_report.py",
    "p3a-audit": "p3a/30_audit_event_extraction_inputs.py",
    "p3a-probe": "p3a/31_probe_extractor_api.py",
    "p3a-sample": "p3a/32_build_event_pilot_sample.py",
    "p3a-run": "p3a/33_run_event_extraction_pilot.py",
    "p3a-cost": "p3a/34_estimate_full_extraction_cost.py",
    "p3a-report": "p3a/35_finalize_p3a_report.py",
    "layout-check": "90_check_project_layout.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return subprocess.call([sys.executable, str(ROOT / "scripts/v6" / STAGES[args.stage]), *args.args], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
