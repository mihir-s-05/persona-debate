#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER_PATH = REPO_ROOT / "out" / "shared_token_ledger.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "out" / "shared_token_ledger_total.json"


@dataclass
class LedgerCostSummary:
    lines_read: int = 0
    rows_parsed: int = 0
    rows_with_cost: int = 0
    rows_missing_or_invalid_cost: int = 0
    total_cost_usd: Decimal = Decimal("0")


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_decimal_cost(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return amount


def compute_total_cost(ledger_path: Path) -> LedgerCostSummary:
    summary = LedgerCostSummary()
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            summary.lines_read += 1
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                summary.rows_missing_or_invalid_cost += 1
                continue

            if not isinstance(row, dict):
                summary.rows_missing_or_invalid_cost += 1
                continue

            summary.rows_parsed += 1
            cost = _to_decimal_cost(row.get("estimated_cost_usd"))
            if cost is None:
                summary.rows_missing_or_invalid_cost += 1
                continue

            summary.rows_with_cost += 1
            summary.total_cost_usd += cost
    return summary


def _payload(ledger_path: Path, summary: LedgerCostSummary) -> dict[str, Any]:
    return {
        "generated_at_utc": _now_iso_utc(),
        "ledger_path": str(ledger_path),
        "lines_read": summary.lines_read,
        "rows_parsed": summary.rows_parsed,
        "rows_with_estimated_cost": summary.rows_with_cost,
        "rows_missing_or_invalid_cost": summary.rows_missing_or_invalid_cost,
        "total_cost_usd": float(summary.total_cost_usd),
        "total_cost_usd_str": f"{summary.total_cost_usd:.8f}",
    }


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate estimated_cost_usd from a shared token ledger JSONL and write a summary "
            "file with the running total."
        )
    )
    parser.add_argument(
        "--ledger-path",
        default=str(DEFAULT_LEDGER_PATH),
        help="Path to shared_token_ledger.jsonl.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Where to write the total cost summary JSON.",
    )
    args = parser.parse_args(argv)

    ledger_path = Path(args.ledger_path)
    summary_path = Path(args.summary_path)
    if not ledger_path.exists():
        print(f"Ledger file not found: {ledger_path}", file=sys.stderr)
        return 1

    summary = compute_total_cost(ledger_path)
    payload = _payload(ledger_path, summary)
    _write_summary(summary_path, payload)

    print(
        f"Updated total cost: ${summary.total_cost_usd:.8f} "
        f"({summary.rows_with_cost} rows with estimated_cost_usd)."
    )
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())