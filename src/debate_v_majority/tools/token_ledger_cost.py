#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_PATH = REPO_ROOT / "out" / "token_ledger_total.json"


def _ledger_path_candidates() -> list[Path]:
    """Fixed set of JSONL ledgers to merge (each included only if the file exists)."""
    return [
        REPO_ROOT / "out" / "token_ledger.jsonl",
        REPO_ROOT / "out" / "shared_token_ledger.jsonl",
        REPO_ROOT / "token_ledger.jsonl",
    ]


@dataclass
class LedgerCostSummary:
    lines_read: int = 0
    rows_parsed: int = 0
    rows_with_cost: int = 0
    rows_missing_or_invalid_cost: int = 0
    total_cost_usd: Decimal = Decimal("0")
    file_breakdown: list[dict[str, Any]] = field(default_factory=list)


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


def _compute_total_cost_one_file(ledger_path: Path) -> LedgerCostSummary:
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


def compute_total_cost(ledger_paths: Path | Sequence[Path]) -> LedgerCostSummary:
    """Sum `estimated_cost_usd` across one or more token ledger JSONL files."""
    paths = [ledger_paths] if isinstance(ledger_paths, Path) else list(ledger_paths)
    if not paths:
        raise ValueError("ledger_paths must contain at least one path")

    combined = LedgerCostSummary()
    for ledger_path in paths:
        one = _compute_total_cost_one_file(ledger_path)
        combined.lines_read += one.lines_read
        combined.rows_parsed += one.rows_parsed
        combined.rows_with_cost += one.rows_with_cost
        combined.rows_missing_or_invalid_cost += one.rows_missing_or_invalid_cost
        combined.total_cost_usd += one.total_cost_usd
        combined.file_breakdown.append(
            {
                "ledger_path": str(ledger_path),
                "lines_read": one.lines_read,
                "rows_parsed": one.rows_parsed,
                "rows_with_estimated_cost": one.rows_with_cost,
                "rows_missing_or_invalid_cost": one.rows_missing_or_invalid_cost,
                "total_cost_usd": float(one.total_cost_usd),
                "total_cost_usd_str": f"{one.total_cost_usd:.8f}",
            }
        )
    return combined


def _dedupe_ledger_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolved_ledger_paths() -> list[Path]:
    """All configured ledger files that exist (shared + local, deduped)."""
    existing = [p for p in _ledger_path_candidates() if p.exists()]
    return _dedupe_ledger_paths(existing)


def _payload(ledger_paths: list[Path], summary: LedgerCostSummary) -> dict[str, Any]:
    return {
        "generated_at_utc": _now_iso_utc(),
        "ledger_paths": [str(p) for p in ledger_paths],
        "ledger_path": str(ledger_paths[0]) if len(ledger_paths) == 1 else None,
        "lines_read": summary.lines_read,
        "rows_parsed": summary.rows_parsed,
        "rows_with_estimated_cost": summary.rows_with_cost,
        "rows_missing_or_invalid_cost": summary.rows_missing_or_invalid_cost,
        "total_cost_usd": float(summary.total_cost_usd),
        "total_cost_usd_str": f"{summary.total_cost_usd:.8f}",
        "per_ledger": summary.file_breakdown,
    }


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate estimated_cost_usd from the repo token ledgers (out/token_ledger.jsonl, "
            "out/shared_token_ledger.jsonl, and repo-root token_ledger.jsonl when each exists) and "
            "write a combined total summary JSON."
        )
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Where to write the total cost summary JSON.",
    )
    args = parser.parse_args(argv)

    ledger_paths = _resolved_ledger_paths()
    if not ledger_paths:
        print(
            "No token ledger JSONL files found. Expected at least one of:\n"
            + "\n".join(f"  - {p}" for p in _ledger_path_candidates()),
            file=sys.stderr,
        )
        return 1

    summary_path = Path(args.summary_path)

    summary = compute_total_cost(ledger_paths)
    payload = _payload(ledger_paths, summary)
    _write_summary(summary_path, payload)

    parts = [f"${row['total_cost_usd_str']}" for row in summary.file_breakdown]
    joined = " + ".join(parts) if len(parts) > 1 else parts[0]
    print(
        f"Updated total cost: ${summary.total_cost_usd:.8f} ({joined}) "
        f"— {summary.rows_with_cost} rows with estimated_cost_usd across {len(ledger_paths)} file(s)."
    )
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())