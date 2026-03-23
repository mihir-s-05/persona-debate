from __future__ import annotations

import json
from pathlib import Path

from debate_v_majority.tools import token_ledger_cost


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_total_cost_sums_estimated_cost_usd(tmp_path: Path) -> None:
    ledger_path = tmp_path / "test-ledger.jsonl"
    _write_lines(
        ledger_path,
        [
            json.dumps({"estimated_cost_usd": 0.10}),
            json.dumps({"estimated_cost_usd": "0.025"}),
            json.dumps({"estimated_cost_usd": None}),
            json.dumps({"other": 123}),
            "not json",
        ],
    )

    summary = token_ledger_cost.compute_total_cost(ledger_path)

    assert summary.lines_read == 5
    assert summary.rows_parsed == 4
    assert summary.rows_with_cost == 2
    assert summary.rows_missing_or_invalid_cost == 3
    assert summary.total_cost_usd == token_ledger_cost.Decimal("0.125")
    assert len(summary.file_breakdown) == 1


def test_compute_total_cost_combines_two_ledgers(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_lines(
        a,
        [
            json.dumps({"estimated_cost_usd": 0.1}),
        ],
    )
    _write_lines(
        b,
        [
            json.dumps({"estimated_cost_usd": 0.2}),
        ],
    )

    summary = token_ledger_cost.compute_total_cost([a, b])

    assert summary.total_cost_usd == token_ledger_cost.Decimal("0.3")
    assert summary.rows_with_cost == 2
    assert len(summary.file_breakdown) == 2
    paths = {row["ledger_path"] for row in summary.file_breakdown}
    assert str(a) in paths and str(b) in paths


def test_main_writes_summary_file(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(token_ledger_cost, "REPO_ROOT", tmp_path)
    summary_path = tmp_path / "token_ledger_total.json"
    _write_lines(
        tmp_path / "out" / "token_ledger.jsonl",
        [
            json.dumps({"estimated_cost_usd": 0.005}),
            json.dumps({"estimated_cost_usd": 0.0075}),
        ],
    )

    exit_code = token_ledger_cost.main(["--summary-path", str(summary_path)])

    assert exit_code == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rows_with_estimated_cost"] == 2
    assert payload["total_cost_usd_str"] == "0.01250000"
    assert payload["ledger_paths"] == [str(tmp_path / "out" / "token_ledger.jsonl")]
    assert len(payload["per_ledger"]) == 1

    stdout = capsys.readouterr().out
    assert "Updated total cost" in stdout


def test_main_merges_token_and_shared_ledgers(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(token_ledger_cost, "REPO_ROOT", tmp_path)
    summary_path = tmp_path / "token_ledger_total.json"
    _write_lines(
        tmp_path / "out" / "token_ledger.jsonl",
        [json.dumps({"estimated_cost_usd": 0.01})],
    )
    _write_lines(
        tmp_path / "out" / "shared_token_ledger.jsonl",
        [json.dumps({"estimated_cost_usd": 0.02})],
    )

    exit_code = token_ledger_cost.main(["--summary-path", str(summary_path)])

    assert exit_code == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_cost_usd_str"] == "0.03000000"
    assert payload["rows_with_estimated_cost"] == 2
    assert len(payload["ledger_paths"]) == 2
    assert len(payload["per_ledger"]) == 2

