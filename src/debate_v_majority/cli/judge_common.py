from __future__ import annotations

from typing import Any

from ..engines import InferenceEngine
from ..shared import PromptTokenCounter


def _parse_csv_ints(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_judge_rounds(rounds_str: str | None, n_rounds: int) -> list[int]:
    if not rounds_str:
        return [n_rounds]
    rounds = _parse_csv_ints(rounds_str)
    if not rounds:
        return [n_rounds]
    invalid = [r for r in rounds if r < 1 or r > n_rounds]
    if invalid:
        raise ValueError(f"Invalid judge rounds {invalid}: must be between 1 and {n_rounds}")
    return sorted(set(rounds))


def _count_prompt_tokens(
    *,
    engine: Any,
    counter: PromptTokenCounter | None,
    messages: list[dict[str, Any]],
) -> int | None:
    count_fn = getattr(engine, "count_prompt_tokens", None)
    base_count_fn = getattr(InferenceEngine, "count_prompt_tokens", None)
    if callable(count_fn) and getattr(count_fn, "__func__", None) is not base_count_fn:
        return int(count_fn(messages))
    if counter is None:
        return None
    try:
        return int(counter.count_chat_tokens(messages))
    except (ModuleNotFoundError, OSError, ValueError):
        return None


__all__ = [
    "_count_prompt_tokens",
    "_parse_csv_ints",
    "_parse_judge_rounds",
]
