from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_debate_script_help_runs():
    proc = _run_help("debate-v-majority")
    assert proc.returncode == 0
    assert "--dataset" in proc.stdout
    assert "--mode" in proc.stdout
    assert "--judge_runtime_model" in proc.stdout
    assert "--hle_variant" in proc.stdout
    assert "--exclude_ids" in proc.stdout
    assert "--dataset_local_mirror" in proc.stdout
    assert "--one" in proc.stdout
    assert "--output" in proc.stdout
    assert "--emit_trace_level" in proc.stdout
    assert "--final_manifest" in proc.stdout
    assert "--final_run" in proc.stdout
    assert "--token_ledger_path" in proc.stdout
    assert "--token_ledger_path" in proc.stdout
    assert "--max_run_cost_usd" in proc.stdout
    assert "--max_total_cost_usd" in proc.stdout


def test_analyze_script_help_runs():
    proc = _run_help("analyze-results")
    assert proc.returncode == 0
    assert "--results-dir" in proc.stdout
    assert "--out-dir" in proc.stdout


def test_trace2txt_script_help_runs():
    proc = _run_help("trace2txt")
    assert proc.returncode == 0
    assert "--input" in proc.stdout
    assert "--item" in proc.stdout


def test_token_ledger_cost_script_help_runs():
    proc = _run_help("token-ledger-cost")
    assert proc.returncode == 0
    assert "--ledger-path" in proc.stdout
    assert "--summary-path" in proc.stdout

