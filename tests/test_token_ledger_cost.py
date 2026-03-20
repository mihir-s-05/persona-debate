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


def test_main_writes_summary_file(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "shared_token_ledger.jsonl"
    summary_path = tmp_path / "shared_token_ledger_total.json"
    _write_lines(
        ledger_path,
        [
            json.dumps({"estimated_cost_usd": 0.005}),
            json.dumps({"estimated_cost_usd": 0.0075}),
        ],
    )

    exit_code = token_ledger_cost.main(
        [
            "--ledger-path",
            str(ledger_path),
            "--summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rows_with_estimated_cost"] == 2
    assert payload["total_cost_usd_str"] == "0.01250000"

    stdout = capsys.readouterr().out
    assert "Updated total cost" in stdout

