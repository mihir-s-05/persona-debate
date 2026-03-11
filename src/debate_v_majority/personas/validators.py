from __future__ import annotations

import re
from typing import Iterable

from .schema import PersonaCard, PersonaDescriptor, ValidationResult


LEAK_PATTERNS = [
    re.compile(r"\\boxed\s*{", re.IGNORECASE),
    re.compile(r"\boption\s+[ABCD]\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*[ABCD0-9-]", re.IGNORECASE),
    re.compile(r"\blikely\s+answer\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\b", re.IGNORECASE),
    re.compile(r"\bthe\s+answer\s+is\b", re.IGNORECASE),
    re.compile(r"\btheorem\b|\blemma\b|\bcorollary\b", re.IGNORECASE),
    re.compile(r"\buse\s+(?:bayes|pigeonhole|fermat|stoichiometry|conservation of|induction|l'h[oô]pital|cauchy|lagrange|euler)\b", re.IGNORECASE),
    re.compile(r"\bchain\s+rule\b|\bproduct\s+rule\b|\bquotient\s+rule\b", re.IGNORECASE),
    re.compile(r"\bhidden trap\b|\btrap in this question\b|\bwatch out for\b", re.IGNORECASE),
    re.compile(r"\bcareful about\s+(?:the\s+)?(?:indexing|off.by.one|boundary|edge case)\b", re.IGNORECASE),
    re.compile(r"\beliminate option\b|\brule out option\b", re.IGNORECASE),
    re.compile(r"\boption\s+[ABCD]\s+is\s+(?:wrong|incorrect|right|correct)\b", re.IGNORECASE),
    re.compile(r"\bapply\s+(?:integration by parts|substitution|partial fractions)\b", re.IGNORECASE),
]

STYLE_ONLY_PATTERNS = [
    re.compile(r"\bbackstory\b", re.IGNORECASE),
    re.compile(r"\bpersona\b.*\bfrom childhood\b", re.IGNORECASE),
]

DUPLICATE_SIMILARITY_THRESHOLD = 0.82


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _normalize_text(text)))


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
    if any(pattern.search(text) for pattern in LEAK_PATTERNS):
        return ValidationResult(status="reject_hard", reasons=["contains answer-oriented leakage indicators"])
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
    return validate_text_for_leakage(text)


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
