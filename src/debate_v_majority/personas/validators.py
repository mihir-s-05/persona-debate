from __future__ import annotations

import re
from typing import Any, Iterable

from .schema import PersonaCard, PersonaDescriptor, ValidationResult


ANSWER_LEAK_PATTERNS = [
    re.compile(r"\\boxed\s*{", re.IGNORECASE),
    re.compile(r"\boption\s+[ABCD]\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*[ABCD0-9-]", re.IGNORECASE),
    re.compile(r"\blikely\s+answer\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\b", re.IGNORECASE),
    re.compile(r"\bthe\s+answer\s+is\b", re.IGNORECASE),
    re.compile(r"\beliminate option\b|\brule out option\b", re.IGNORECASE),
    re.compile(r"\boption\s+[ABCD]\s+is\s+(?:wrong|incorrect|right|correct)\b", re.IGNORECASE),
]

QUESTION_HINT_PATTERNS = [
    re.compile(r"\bhidden trap\b|\btrap in this question\b|\bwatch out for\b", re.IGNORECASE),
    re.compile(r"\bcareful about\s+(?:the\s+)?(?:indexing|off.by.one|boundary|edge case)\b", re.IGNORECASE),
]

STYLE_ONLY_PATTERNS = [
    re.compile(r"\bbackstory\b", re.IGNORECASE),
    re.compile(r"\bpersona\b.*\bfrom childhood\b", re.IGNORECASE),
]

DUPLICATE_SIMILARITY_THRESHOLD = 0.82

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give", "find", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to",
    "use", "using", "with", "what", "which", "who", "why", "how", "when", "where", "problem",
    "question", "answer", "final", "carefully", "return", "form", "following", "task", "dataset",
    "option", "options", "correct", "incorrect", "value", "values", "real", "positive", "negative",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _normalize_text(text)))


def _question_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", _normalize_text(text))
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _question_ngrams(text: str, *, n: int) -> set[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9_]+", _normalize_text(text))
        if len(token) >= 3 and token not in _STOPWORDS
    ]
    return {" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def _collect_task_text(raw_task: dict[str, Any] | None) -> str:
    if not raw_task:
        return ""
    parts: list[str] = []
    for value in raw_task.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(x) for x in value if isinstance(x, str))
    return " ".join(parts)


def _jaccard_similarity(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def validate_text_for_leakage(text: str) -> ValidationResult:
    norm = _normalize_text(text)
    if any(pattern.search(text) for pattern in ANSWER_LEAK_PATTERNS):
        return ValidationResult(status="reject_hard", reasons=["contains answer-oriented leakage indicators"])
    if any(pattern.search(text) for pattern in QUESTION_HINT_PATTERNS):
        return ValidationResult(status="retry", reasons=["contains question-specific hint language"])
    if any(pattern.search(text) for pattern in STYLE_ONLY_PATTERNS):
        return ValidationResult(status="retry", reasons=["contains style-only or backstory-heavy language"])
    if len(norm.split()) < 5:
        return ValidationResult(status="retry", reasons=["too short to be operational"])
    return ValidationResult(status="accept", reasons=[])


def validate_descriptor(descriptor: PersonaDescriptor) -> ValidationResult:
    text = " ".join(
        [
            descriptor.name,
            descriptor.short_rule,
            descriptor.reasoning_summary,
            " ".join(descriptor.axis_interpretation.values()),
        ]
    )
    res = validate_text_for_leakage(text)
    return res


def validate_descriptor_against_task(
    descriptor: PersonaDescriptor,
    *,
    question: str | None = None,
    raw_task: dict[str, Any] | None = None,
    context_texts: list[str] | None = None,
) -> ValidationResult:
    base = validate_descriptor(descriptor)
    if base.status != "accept":
        return base

    descriptor_text = " ".join(
        [
            descriptor.name,
            descriptor.short_rule,
            descriptor.reasoning_summary,
            " ".join(descriptor.axis_interpretation.values()),
        ]
    )
    task_text = " ".join(
        part for part in [question or "", _collect_task_text(raw_task), *(context_texts or [])] if part
    ).strip()
    if not task_text:
        return base

    descriptor_tokens = _question_tokens(descriptor_text)
    task_tokens = _question_tokens(task_text)
    shared = descriptor_tokens & task_tokens
    shared_specific = {token for token in shared if len(token) >= 8}

    task_bigrams = _question_ngrams(task_text, n=2)
    task_trigrams = _question_ngrams(task_text, n=3)
    descriptor_norm = _normalize_text(descriptor_text)
    copied_bigrams = sum(1 for ngram in task_bigrams if len(ngram) >= 12 and ngram in descriptor_norm)
    if len(shared_specific) >= 3 or (len(shared_specific) >= 2 and copied_bigrams >= 1):
        return ValidationResult(
            status="retry",
            reasons=["repeats too many question-specific terms instead of describing reusable reasoning policy"],
        )
    if any(ngram in descriptor_norm for ngram in task_trigrams if len(ngram) >= 18):
        return ValidationResult(
            status="retry",
            reasons=["copies question-specific phrasing into the descriptor"],
        )
    if copied_bigrams >= 3:
        return ValidationResult(
            status="retry",
            reasons=["copies question-specific phrasing into the descriptor"],
        )

    return base


def validate_card(card: PersonaCard) -> ValidationResult:
    text = " ".join(
        [
            card.title,
            card.core_reasoning_strategy,
            " ".join(card.priorities),
            " ".join(card.distrusts),
            card.decomposition_style,
            card.revision_policy,
            card.confidence_policy,
            card.failure_mode_to_avoid,
            card.system_prompt,
        ]
    )
    res = validate_text_for_leakage(text)
    if len(_normalize_text(card.system_prompt)) > 1200:
        return ValidationResult(status="retry", reasons=["system_prompt too long for compact operational card"])
    return res


def duplicate_diagnostics(texts: Iterable[str], *, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> list[dict[str, float | int]]:
    items = [str(x) for x in texts]
    out: list[dict[str, float | int]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = _jaccard_similarity(items[i], items[j])
            if sim >= threshold:
                out.append({"left": i, "right": j, "similarity": round(sim, 4)})
    return out
