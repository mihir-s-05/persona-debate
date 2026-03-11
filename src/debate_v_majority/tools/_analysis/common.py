from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ...shared import majority_vote_details


FINDINGS_MD_SECTIONS: list[str] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_findings_md(md: str) -> None:
    if md:
        FINDINGS_MD_SECTIONS.append(str(md).rstrip())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{fmt_pct(k/n if n else 0.0)} [{fmt_pct(lo)}, {fmt_pct(hi)}]"


def entropy_from_counts(counts: Counter[str | None]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p, 2)
    return h


def strict_majority_vote(answers: list[str | None]) -> str | None:
    return majority_vote_details(answers)["strict_majority_answer"]


def plurality_vote(answers: list[str | None]) -> str | None:
    counts = Counter(answers)
    if not counts:
        return None
    top_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else None


def plurality_vote_ignore_none(answers: list[str | None]) -> str | None:
    return plurality_vote([a for a in answers if a is not None])


def mean(values: Iterable[float]) -> float | None:
    nums = [float(v) for v in values]
    if not nums:
        return None
    return sum(nums) / len(nums)


def median(values: Iterable[float]) -> float | None:
    nums = sorted(float(v) for v in values)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2 == 1:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def count_none(xs: Iterable[Any]) -> int:
    return sum(1 for x in xs if x is None)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])

