from __future__ import annotations

import re
from typing import Any, Iterable

from .schema import PersonaCard, PersonaDescriptor, ValidationResult


ANSWER_LEAK_PATTERNS = [
    re.compile(r"\\boxed\s*{", re.IGNORECASE),
    re.compile(r"\boption\s+[A-E]\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*[A-E0-9-]", re.IGNORECASE),
    re.compile(r"\blikely\s+answer\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+answer\b", re.IGNORECASE),
    re.compile(r"\bthe\s+answer\s+is\b", re.IGNORECASE),
    re.compile(r"\beliminate option\b|\brule out option\b", re.IGNORECASE),
    re.compile(r"\boption\s+[A-E]\s+is\s+(?:wrong|incorrect|right|correct)\b", re.IGNORECASE),
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

_CARD_ACTION_VERBS = (
    "switch",
    "revise",
    "rebuild",
    "restart",
    "abandon",
    "defend",
    "maintain",
    "retain",
    "hold",
    "keep",
    "patch",
    "narrow",
    "re-evaluate",
)

_REVISION_TRIGGER_CUES = (
    "contradiction",
    "contradictions",
    "missing case",
    "broken",
    "cannot repair",
    "ranking",
    "changes your view",
    "better-supported",
    "fails",
    "counter-example",
    "counterexample",
    "edge case",
    "exception",
    "invalidated",
    "identified",
    "demonstrated",
    "demonstrates",
    "rebuild",
    "re-derivation",
    "framework",
    "mismatch",
    "unhandled",
    "fatal structural flaw",
    "lower-energy",
    "more plausible",
    "more probable",
    "foundational block",
    "dead end",
    "different outcome",
    "superior",
    "proven impossible",
    "proved impossible",
    "proves that",
    "violation",
    "violates",
    "forbidden zone",
    "excluded zone",
    "global law",
    "symmetry",
    "specific link",
    "concrete contradiction",
)

_REVISION_EVIDENCE_CUES = (
    "evidence",
    "counterevidence",
    "counter-evidence",
    "counterexample",
    "counter-example",
    "objection",
    "objections",
    "critique",
    "critiques",
)

_CRITIQUE_ATTACK_CUES = (
    "attack",
    "attacks",
    "critique",
    "expose",
    "exposes",
    "audit",
    "audits",
    "scrutinize",
    "scrutinizes",
    "probe",
    "probes",
    "check",
    "checks",
    "identify",
    "identifies",
    "demand",
    "demands",
    "test",
    "tests",
    "hunt",
    "hunts",
    "flag",
    "flags",
    "question",
    "questions",
    "pressure-test",
    "pressure tests",
)

_CRITIQUE_FAILURE_CUES = (
    "unsupported",
    "assumption",
    "constraint",
    "constraints",
    "break",
    "category",
    "ranking",
    "drift",
    "link",
    "contradiction",
    "contradictions",
    "counter-example",
    "counterexample",
    "edge case",
    "exception",
    "invalid",
    "gap",
    "gaps",
    "leak",
    "leaks",
    "stability",
    "connect",
    "connectivity",
    "justify",
    "justification",
    "final claim",
    "conclusion",
    "derive",
    "derives",
    "support",
    "supports",
    "bind",
    "architecture",
    "framework",
    "mapping",
    "rule",
    "rules",
    "symmetry",
    "conservation law",
    "verification",
    "step-by-step",
    "transition",
    "transitions",
    "impossible state",
    "impossible states",
    "high-level leap",
    "high-level leaps",
    "local dependency",
    "local dependencies",
    "forbidden zone",
)

_CRITIQUE_STRUCTURAL_CUES = (
    "whether",
    "when",
    "fails",
    "violate",
    "violates",
    "bypass",
    "single rule",
    "specific",
)


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


def _compat_attr(obj: Any, *names: str) -> str:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return str(value)
    return ""


def _compat_list_attr(obj: Any, *names: str) -> list[str]:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, list):
            return [str(x) for x in value]
    return []


def _compat_map_attr(obj: Any, *names: str) -> dict[str, str]:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    return {}


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _descriptor_text(descriptor: PersonaDescriptor) -> str:
    return " ".join(
        [
            descriptor.name,
            _compat_attr(descriptor, "solve_style", "short_rule"),
            _compat_attr(descriptor, "critique_style", "reasoning_summary"),
            _compat_attr(descriptor, "revision_policy"),
            _compat_attr(descriptor, "failure_mode_to_watch"),
            " ".join(_compat_map_attr(descriptor, "axis_signature", "axis_interpretation").values()),
        ]
    )


def _card_text(card: PersonaCard) -> str:
    return " ".join(
        [
            card.title,
            _compat_attr(card, "core_policy", "core_reasoning_strategy"),
            " ".join(_compat_list_attr(card, "priorities")),
            " ".join(_compat_list_attr(card, "distrusts")),
            " ".join(_compat_map_attr(card, "axis_signature").values()),
            _compat_attr(card, "critique_policy", "decomposition_style"),
            _compat_attr(card, "revision_policy"),
            _compat_attr(card, "round_reminder", "confidence_policy"),
            _compat_attr(card, "failure_mode_to_watch", "failure_mode_to_avoid"),
            card.system_prompt,
        ]
    )


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
    return validate_text_for_leakage(_descriptor_text(descriptor))


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

    descriptor_text = _descriptor_text(descriptor)
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
    critique_policy = _compat_attr(card, "critique_policy", "decomposition_style")
    revision_policy = _compat_attr(card, "revision_policy")
    axis_signature = _compat_map_attr(card, "axis_signature")
    res = validate_text_for_leakage(_card_text(card))
    if res.status != "accept":
        return res
    if len(_normalize_text(card.system_prompt)) > 1200:
        return ValidationResult(status="retry", reasons=["system_prompt too long for compact operational card"])
    if axis_signature and not (2 <= len(axis_signature) <= 4):
        return ValidationResult(status="retry", reasons=["axis_signature must contain between 2 and 4 entries"])

    revision_norm = _normalize_text(revision_policy)
    wrapped_revision = f" {revision_norm} "
    has_trigger_keyword = _contains_any(revision_norm, _REVISION_TRIGGER_CUES)
    has_conditional_structure = (
        any(marker in wrapped_revision for marker in (" if ", " unless ", " when ", " only if "))
        and _contains_any(revision_norm, _CARD_ACTION_VERBS)
    )
    has_compact_evidence_trigger = (
        _contains_any(revision_norm, _CARD_ACTION_VERBS)
        and _contains_any(revision_norm, _REVISION_EVIDENCE_CUES)
    )
    has_compact_revision_label = (
        len(revision_norm.split()) <= 2
        and revision_norm in _CARD_ACTION_VERBS
    )
    if not (
        has_trigger_keyword
        or has_conditional_structure
        or has_compact_evidence_trigger
        or has_compact_revision_label
    ):
        return ValidationResult(status="retry", reasons=["revision_policy must name a concrete switch trigger"])

    critique_norm = _normalize_text(critique_policy)
    has_attack_verb = _contains_any(critique_norm, _CRITIQUE_ATTACK_CUES)
    has_failure_target = _contains_any(critique_norm, _CRITIQUE_FAILURE_CUES)
    has_structural_specificity = _contains_any(critique_norm, _CRITIQUE_STRUCTURAL_CUES)
    has_compact_critique_label = (
        len(critique_norm.split()) <= 3
        and (
            critique_norm in {"branch-and-test", "stepwise", "step"}
            or critique_norm.endswith("decomposition")
        )
    )
    if not (
        has_compact_critique_label
        or ((has_attack_verb or has_structural_specificity) and (has_failure_target or has_structural_specificity))
    ):
        return ValidationResult(status="retry", reasons=["critique_policy is too generic"])
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
