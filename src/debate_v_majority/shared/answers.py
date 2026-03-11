from __future__ import annotations

import re
from collections import Counter
from typing import Any


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} from a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    """
    Remove a \\boxed{...} or \\fbox{...} wrapper and return inner content.
    Tolerates whitespace between the command and the opening brace (e.g. \\boxed {A}).
    """
    if not isinstance(s, str):
        return None
    if not s.endswith("}"):
        return None
    m = re.match(r"^\\(?:boxed|fbox)\s*{", s)
    if not m:
        return None
    return s[m.end() : -1]


def parse_math(text: str) -> str | None:
    """Parse math answer from \\boxed{} format."""
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def normalize_numeric_string(s: str | None) -> str | None:
    """Normalize a numeric string (remove commas, leading zeros, etc.)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    m = re.fullmatch(r"-?\d+", s)
    if m:
        sign = ""
        if s.startswith("-"):
            sign = "-"
            s = s[1:]
        s = s.lstrip("0") or "0"
        return sign + s
    return s


def normalize_freeform_string(s: str | None) -> str | None:
    """Normalize a freeform string for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.strip("$")
    s = " ".join(s.split())
    s = s.strip().strip(".")
    return s.lower()


def most_frequent_answer(answers: list[str | None] | None) -> str | None:
    """Return the unique repeated mode answer, else None."""
    if not answers:
        return None
    filtered = [a for a in answers if a is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    top_count = max(counts.values(), default=0)
    if top_count <= 1:
        return None
    top = [ans for ans, count in counts.items() if count == top_count]
    return top[0] if len(top) == 1 else None


def vote_counts(answers: list[str | None] | None) -> dict[str, int]:
    """Count parsed answers while preserving unparsed votes explicitly."""
    counts: dict[str, int] = {}
    for answer in answers or []:
        key = "<unparsed>" if answer is None else str(answer)
        counts[key] = counts.get(key, 0) + 1
    return counts


def strict_majority_answer(answers: list[str | None] | None) -> str | None:
    """Return an answer only when it wins with more than half of all votes."""
    if not answers:
        return None
    filtered = [str(answer) for answer in answers if answer is not None]
    if not filtered:
        return None
    threshold = len(answers) / 2
    counts = Counter(filtered)
    for answer, count in counts.items():
        if count > threshold:
            return answer
    return None


def plurality_answer(answers: list[str | None] | None) -> str | None:
    """Return the unique non-None plurality answer, else None."""
    if not answers:
        return None
    filtered = [str(answer) for answer in answers if answer is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    top_count = max(counts.values(), default=0)
    winners = [answer for answer, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else None


def majority_vote_details(answers: list[str | None] | None) -> dict[str, Any]:
    """Return vote diagnostics plus the harness-selected mode-style winner."""
    answer_list = list(answers or [])
    return {
        "vote_counts": vote_counts(answer_list),
        "strict_majority_answer": strict_majority_answer(answer_list),
        "plurality_answer": plurality_answer(answer_list),
        "majority_answer": most_frequent_answer(answer_list),
    }


__all__ = [
    "last_boxed_only_string",
    "majority_vote_details",
    "most_frequent_answer",
    "normalize_freeform_string",
    "normalize_numeric_string",
    "parse_math",
    "plurality_answer",
    "remove_boxed",
    "strict_majority_answer",
    "vote_counts",
]
