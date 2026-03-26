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
SEMANTIC_REDUNDANCY_THRESHOLD = 0.92
COVERAGE_AUDIT_VERSION = "phase1.coverage_audit.v1"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give", "find", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to",
    "use", "using", "with", "what", "which", "who", "why", "how", "when", "where", "problem",
    "question", "answer", "final", "carefully", "return", "form", "following", "task", "dataset",
    "option", "options", "correct", "incorrect", "value", "values", "real", "positive", "negative",
}

_SEMANTIC_AUDIT_STOPWORDS = _STOPWORDS | {
    "agent",
    "agents",
    "answering",
    "branch",
    "branches",
    "candidate",
    "candidates",
    "check",
    "checks",
    "claim",
    "claims",
    "common",
    "confidence",
    "constraint",
    "constraints",
    "critique",
    "decomposition",
    "evidence",
    "explanation",
    "explain",
    "first",
    "general",
    "line",
    "model",
    "operate",
    "operational",
    "path",
    "paths",
    "persona",
    "personas",
    "policy",
    "reason",
    "reasoning",
    "revision",
    "round",
    "solve",
    "step",
    "steps",
    "support",
    "system",
    "trace",
    "verify",
    "verification",
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

_DEBATE_JOB_TITLE_TOKENS = ("auditor", "validator", "sentinel", "deconstructor")
_COVERAGE_BUCKET_AXES = ("commitment_style", "abstraction_preference", "search_strategy")


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
    visible_keys = ("question", "image", "image_preview")
    for key in visible_keys:
        value = raw_task.get(key)
        if isinstance(value, str):
            parts.append(value)

    choice_map = raw_task.get("choice_map")
    if isinstance(choice_map, dict):
        parts.extend(str(value) for value in choice_map.values() if isinstance(value, str))

    choice_labels = raw_task.get("choice_labels")
    if isinstance(choice_labels, list):
        parts.extend(str(value) for value in choice_labels if isinstance(value, str))
    return " ".join(parts)


def _jaccard_similarity(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _semantic_overlap(a: str, b: str) -> dict[str, Any]:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    a_tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", a_norm)
        if len(token) >= 4 and token not in _SEMANTIC_AUDIT_STOPWORDS
    }
    b_tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", b_norm)
        if len(token) >= 4 and token not in _SEMANTIC_AUDIT_STOPWORDS
    }
    shared_tokens = sorted(a_tokens & b_tokens)
    token_jaccard = _jaccard_similarity(a_norm, b_norm)

    a_bigrams = _question_ngrams(a_norm, n=2)
    b_bigrams = _question_ngrams(b_norm, n=2)
    shared_bigrams = sorted(a_bigrams & b_bigrams)
    a_trigrams = _question_ngrams(a_norm, n=3)
    b_trigrams = _question_ngrams(b_norm, n=3)
    shared_trigrams = sorted(a_trigrams & b_trigrams)

    bigram_score = len(shared_bigrams) / max(1, min(len(a_bigrams), len(b_bigrams)))
    trigram_score = len(shared_trigrams) / max(1, min(len(a_trigrams), len(b_trigrams)))
    shared_term_score = len(shared_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    similarity = min(1.0, round((0.5 * token_jaccard) + (0.2 * bigram_score) + (0.15 * trigram_score) + (0.15 * shared_term_score), 4))

    return {
        "similarity": similarity,
        "token_jaccard": round(token_jaccard, 4),
        "shared_terms": shared_tokens,
        "shared_bigrams": shared_bigrams,
        "shared_trigrams": shared_trigrams,
        "shared_term_count": len(shared_tokens),
        "shared_bigram_count": len(shared_bigrams),
        "shared_trigram_count": len(shared_trigrams),
    }


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
            _compat_attr(descriptor, "question_approach_summary"),
            _compat_attr(descriptor, "disagreement_profile"),
            _compat_attr(descriptor, "revision_profile"),
            _compat_attr(descriptor, "solve_style", "short_rule"),
            _compat_attr(descriptor, "critique_style", "reasoning_summary"),
            _compat_attr(descriptor, "revision_policy"),
            _compat_attr(descriptor, "failure_mode_to_watch"),
            " ".join(_compat_map_attr(descriptor, "round1_solver_profile").values()),
            " ".join(_compat_map_attr(descriptor, "debate_temperament_profile").values()),
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
            " ".join(_compat_map_attr(card, "round1_solver_policy").values()),
            " ".join(_compat_map_attr(card, "round2_critique_policy").values()),
            " ".join(_compat_map_attr(card, "round3_revision_policy").values()),
            " ".join(_compat_map_attr(card, "runtime_prompts").values()),
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
    prompt_text = " ".join(part for part in [question or "", _collect_task_text(raw_task)] if part).strip()
    context_text = " ".join(context_texts or []).strip()
    task_text = " ".join(part for part in [prompt_text, context_text] if part).strip()
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
    copied_trigrams = sum(1 for ngram in task_trigrams if len(ngram) >= 18 and ngram in descriptor_norm)
    # Prompt-conditioned descriptors should be allowed to reuse visible task language,
    # especially on dense technical prompts where a useful approach summary naturally
    # repeats domain terms. Treat overlap as suspicious only when it is both very high
    # and accompanied by substantial copied phrasing.
    if len(shared_specific) >= 20 and copied_bigrams >= 5:
        return ValidationResult(
            status="retry",
            reasons=["repeats too many question-specific terms instead of describing reusable reasoning policy"],
        )
    if copied_trigrams >= 1:
        return ValidationResult(
            status="retry",
            reasons=["copies question-specific phrasing into the descriptor"],
        )
    if copied_bigrams >= 4:
        return ValidationResult(
            status="retry",
            reasons=["copies question-specific phrasing into the descriptor"],
        )
    if context_text:
        context_tokens = _question_tokens(context_text)
        context_shared_specific = {token for token in descriptor_tokens & context_tokens if len(token) >= 8}
        context_bigrams = _question_ngrams(context_text, n=2)
        context_trigrams = _question_ngrams(context_text, n=3)
        copied_context_bigrams = sum(1 for ngram in context_bigrams if len(ngram) >= 12 and ngram in descriptor_norm)
        copied_context_trigrams = sum(1 for ngram in context_trigrams if len(ngram) >= 18 and ngram in descriptor_norm)
        if len(context_shared_specific) >= 3 or copied_context_bigrams >= 1 or copied_context_trigrams >= 1:
            return ValidationResult(
                status="retry",
                reasons=["repeats too many question-specific terms instead of describing reusable reasoning policy"],
            )

    return base


def validate_card(card: PersonaCard) -> ValidationResult:
    round2_policy = _compat_map_attr(card, "round2_critique_policy")
    round3_policy = _compat_map_attr(card, "round3_revision_policy")
    critique_policy = (
        round2_policy.get("primary_attack_rule")
        or _compat_attr(card, "critique_policy", "decomposition_style")
    )
    revision_policy = (
        round3_policy.get("switch_triggers")
        or round3_policy.get("default_stance")
        or _compat_attr(card, "revision_policy")
    )
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
    has_structured_revision_policy = (
        len(_normalize_text(round3_policy.get("switch_triggers", "")).split()) >= 5
        and (
            len(_normalize_text(round3_policy.get("default_stance", "")).split()) >= 4
            or len(_normalize_text(round3_policy.get("patch_vs_rebuild_rule", "")).split()) >= 4
        )
    )
    if not (
        has_trigger_keyword
        or has_conditional_structure
        or has_compact_evidence_trigger
        or has_compact_revision_label
        or has_structured_revision_policy
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
    has_structured_critique_policy = (
        len(_normalize_text(round2_policy.get("primary_attack_rule", "")).split()) >= 5
        and (
            len(_normalize_text(round2_policy.get("preferred_target_type", "")).split()) >= 3
            or len(_normalize_text(round2_policy.get("what_to_ignore", "")).split()) >= 3
        )
    )
    if not (
        has_compact_critique_label
        or has_structured_critique_policy
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


def semantic_redundancy_diagnostics(
    texts: Iterable[str],
    *,
    threshold: float = SEMANTIC_REDUNDANCY_THRESHOLD,
) -> list[dict[str, Any]]:
    items = [str(x) for x in texts]
    out: list[dict[str, Any]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            details = _semantic_overlap(items[i], items[j])
            if details["similarity"] >= threshold:
                out.append(
                    {
                        "left": i,
                        "right": j,
                        **details,
                    }
                )
    return out


def semantic_redundancy_audit(
    texts: Iterable[str],
    *,
    threshold: float = SEMANTIC_REDUNDANCY_THRESHOLD,
    label: str = "item",
) -> dict[str, Any]:
    items = [str(x) for x in texts]
    pairs = semantic_redundancy_diagnostics(items, threshold=threshold)
    max_similarity = max((float(pair.get("similarity") or 0.0) for pair in pairs), default=0.0)
    return {
        "label": label,
        "threshold": threshold,
        "n_items": len(items),
        "status": "retry" if pairs else "accept",
        "max_similarity": round(max_similarity, 4),
        "pairs": pairs,
    }


def _bucket(value: float) -> str:
    if value <= 0.33:
        return "low"
    if value >= 0.67:
        return "high"
    return "balanced"


def _normalized_map_value(mapping: dict[str, str], key: str) -> str:
    return _normalize_text(mapping.get(key, ""))


def descriptor_coverage_audit(descriptors: list[PersonaDescriptor]) -> dict[str, Any]:
    issues: list[str] = []
    redundant_indices: set[int] = set()
    axis_bucket_counts: dict[str, dict[str, int]] = {axis_id: {} for axis_id in _COVERAGE_BUCKET_AXES}
    for descriptor in descriptors:
        for axis_id in _COVERAGE_BUCKET_AXES:
            if axis_id not in descriptor.axis_values:
                continue
            bucket = _bucket(float(descriptor.axis_values.get(axis_id, 0.5)))
            counts = axis_bucket_counts[axis_id]
            counts[bucket] = counts.get(bucket, 0) + 1
    for axis_id, counts in axis_bucket_counts.items():
        if not counts:
            continue
        for bucket, count in counts.items():
            if len(descriptors) >= 5 and bucket in {"low", "high"} and count > 2:
                issues.append(f"too many personas share {axis_id}={bucket}")
        if len(descriptors) >= 5:
            if counts.get("low", 0) == 0:
                issues.append(f"missing low bucket on {axis_id}")
            if counts.get("high", 0) == 0:
                issues.append(f"missing high bucket on {axis_id}")

    patterns: dict[tuple[str, str, str], list[int]] = {}
    for idx, descriptor in enumerate(descriptors):
        profile = descriptor.round1_solver_profile
        pattern = (
            _normalized_map_value(profile, "candidate_generation_policy"),
            _normalized_map_value(profile, "evidence_priority_policy"),
            _normalized_map_value(profile, "verification_policy"),
        )
        patterns.setdefault(pattern, []).append(idx)
    for indices in patterns.values():
        if len(descriptors) >= 4 and len(indices) >= 3:
            redundant_indices.update(indices)
            issues.append("three or more personas imply the same round-1 strategy pattern")

    approach_groups: dict[str, list[int]] = {}
    for idx, descriptor in enumerate(descriptors):
        approach = _normalize_text(descriptor.question_approach_summary or descriptor.short_rule)
        approach_groups.setdefault(approach, []).append(idx)
    for indices in approach_groups.values():
        if len(descriptors) >= 4 and len(indices) >= max(2, len(descriptors) - 2):
            redundant_indices.update(indices)
            issues.append("question_approach_summary collapsed into near-duplicate approaches")

    discouraged_name_count = sum(
        1
        for descriptor in descriptors
        if any(token in _normalize_text(descriptor.name) for token in _DEBATE_JOB_TITLE_TOKENS)
    )
    if len(descriptors) >= 5 and discouraged_name_count >= 3:
        issues.append("too many personas are named like debate staff roles instead of stable personas")

    return {
        "label": "descriptor_coverage",
        "status": "retry" if issues else "accept",
        "issues": issues,
        "redundant_indices": sorted(redundant_indices),
        "axis_bucket_counts": axis_bucket_counts,
        "version": COVERAGE_AUDIT_VERSION,
    }


def card_coverage_audit(cards: list[PersonaCard]) -> dict[str, Any]:
    issues: list[str] = []
    redundant_indices: set[int] = set()
    signatures: dict[tuple[str, str, str, str], list[int]] = {}
    for idx, card in enumerate(cards):
        policy = card.round1_solver_policy
        signature = (
            _normalized_map_value(policy, "candidate_generation_order"),
            _normalized_map_value(policy, "hypothesis_retention_rule"),
            _normalized_map_value(policy, "early_disqualifiers"),
            _normalized_map_value(policy, "verification_trigger"),
        )
        signatures.setdefault(signature, []).append(idx)
    distinct_policy_pairs = 0
    for signature, indices in signatures.items():
        distinct_fields = sum(1 for field in signature if field)
        if len(cards) >= 4 and len(indices) >= 2:
            redundant_indices.update(indices)
            issues.append("round1_solver_policy signatures are duplicated across multiple cards")
        if distinct_fields >= 2:
            distinct_policy_pairs += 1
    if len(cards) >= 4 and distinct_policy_pairs < max(1, len(cards) - 1):
        issues.append("not enough cards differ meaningfully on round1_solver_policy fields")

    return {
        "label": "card_coverage",
        "status": "retry" if issues else "accept",
        "issues": issues,
        "redundant_indices": sorted(redundant_indices),
        "version": COVERAGE_AUDIT_VERSION,
    }
