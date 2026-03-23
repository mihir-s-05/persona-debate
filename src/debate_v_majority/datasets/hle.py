"""
HLE-Verified dataset handling: loading metadata, prompt rendering, extraction, and scoring.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Literal

from ..shared import normalize_freeform_string, normalize_numeric_string, parse_math
from .base import build_structured_debate_message


HLE_DATASET_ID = "skylenage/HLE-Verified"
HLE_DATASET_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
HLE_PAPER_VERSION = "arXiv:2602.13964v3"
HLE_SCORER_PROVENANCE = "hle_verified.scorer.v1"
HLE_EXTRACTOR_PROVENANCE = "hle_verified.extractor.v1"

HLEVariant = Literal["verified", "revised", "verified_full"]
HLEModality = Literal["any", "text_only", "image_only"]
HLEAnswerFormat = Literal["multiple_choice", "numeric_exact", "freeform_exact"]
HLE_IMAGE_KEYS = ("image", "image_preview", "rationale_image")

VARIANT_TO_SUBSETS: dict[HLEVariant, set[str]] = {
    "verified": {"Gold subset"},
    "revised": {"Revision subset"},
    "verified_full": {"Gold subset", "Revision subset", "Uncertain subset"},
}

AGENT_PROMPT = {
    "question": (
        "Solve the following HLE-Verified task carefully.\n"
        "Return your final answer in the required strict format.\n"
        "If you report confidence, use `Confidence: <number between 0 and 1>`.\n\n"
        "{question}"
    ),
    "debate": [
        "These are the prior-round outputs from other agents:",
        (
            "\n\nCritique the strongest and weakest parts of the other agents' reasoning, "
            "decide whether to revise, and then give one strict final answer."
        ),
    ],
}

JUDGE_PROMPT = {
    "user_prompt_suffix": (
        "\n\nRead the agents' responses and select the answer best supported by the transcript. "
        "Do not solve from scratch if the transcript already settles the item. "
        "Return one strict final answer at the end, and optionally a `Confidence:` line."
    )
}


def allowed_subset_labels(variant: HLEVariant) -> set[str]:
    return set(VARIANT_TO_SUBSETS[variant])


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, dict):
        for key in ("path", "url", "uri", "file_uri", "src"):
            candidate = value.get(key)
            if candidate:
                s = str(candidate).strip()
                if s:
                    return s
    s = str(value).strip()
    return s or None


def inflate_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Recover the full HLE-Verified record from either a flat HF row or a saved mirror row."""
    if "answer_type" in sample and "question" in sample:
        raw_task = dict(sample)
    else:
        payload = sample.get("json")
        if isinstance(payload, str) and payload.strip():
            raw_task = json.loads(payload)
        else:
            raw_task = dict(sample)

    for key in (
        "id",
        "Verified_Classes",
        "category",
        "raw_subject",
        "problem_is_valid",
        "problem_error_type",
        "answer_is_valid",
        "answer_error_type",
        "rationale_is_valid",
        "rationale_error_type",
        "question",
        "answer",
    ):
        value = sample.get(key)
        if value is not None and raw_task.get(key) in (None, ""):
            raw_task[key] = value

    raw_task.setdefault("answer_type", "exactMatch")
    return raw_task


def _extract_choice_map(question: str) -> dict[str, str]:
    choice_map: dict[str, str] = {}
    for line in str(question or "").splitlines():
        m = re.match(r"^\s*(?:\\item\s+|[-*]\s+|•\s+)?([A-Z])(?:[.)]|:)\s*(.+?)\s*$", line)
        if not m:
            continue
        label = m.group(1).upper()
        text = m.group(2).strip()
        if len(label) == 1 and text:
            choice_map[label] = text
    return choice_map


def _normalize_choice_text(value: str | None) -> str | None:
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value))
    s = s.strip().strip(".")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s)
    return s.casefold() or None


def _replace_frac_tokens(text: str) -> str:
    out = str(text)
    while True:
        new = re.sub(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1/\2)", out)
        if new == out:
            return new
        out = new


def _normalize_exact_math(value: str | None) -> str | None:
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip()
    if not s:
        return None
    s = s.strip("$")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "")
    s = _replace_frac_tokens(s)
    s = re.sub(r"\\(?:text|mathrm|mathbf)\s*{([^{}]+)}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    return s


def _is_numeric_exact_value(value: str | None) -> bool:
    s = _normalize_exact_math(value)
    if s is None:
        return False
    return bool(
        re.fullmatch(
            r"[()\[\],./+\-0-9]+",
            s.replace("−", "-"),
        )
    )


def infer_answer_format_type(task_info: dict[str, Any]) -> HLEAnswerFormat:
    answer_type = str(task_info.get("answer_type") or "exactMatch").strip()
    if answer_type == "multipleChoice":
        return "multiple_choice"
    if _is_numeric_exact_value(task_info.get("answer")):
        return "numeric_exact"
    return "freeform_exact"


def infer_domain_family(task_info: dict[str, Any]) -> str:
    category = str(task_info.get("category") or "").strip().lower()
    if category in {"math", "mathematics"}:
        return "math"
    if category in {"physics", "chemistry"}:
        return "physical_sciences"
    if category in {"biology/medicine", "biology", "medicine"}:
        return "medicine"
    if category in {"computer science/ai", "computer science", "artificial intelligence", "engineering"}:
        return "computer_science"
    if category in {"humanities/social science", "humanities", "social science"}:
        return "humanities"
    return "applied_professional_reasoning"


def prepare_task(task_info: dict[str, Any]) -> dict[str, Any]:
    raw_task = inflate_record(task_info)
    raw_task["choice_map"] = _extract_choice_map(str(raw_task.get("question") or ""))
    raw_task["choice_labels"] = sorted(raw_task["choice_map"])
    raw_task["answer_format_type"] = infer_answer_format_type(raw_task)
    raw_task["domain_family"] = infer_domain_family(raw_task)
    return raw_task


def _answer_rule_and_gt(raw_task: dict[str, Any], *, question_text: str) -> tuple[str, str]:
    format_type = raw_task["answer_format_type"]
    if format_type == "multiple_choice":
        labels = raw_task.get("choice_labels") or sorted(_extract_choice_map(question_text))
        if labels:
            answer_rule = f"Put your final answer in the form \\boxed{{{labels[0]}}} using exactly one choice label from {', '.join(labels)}."
        else:
            answer_rule = "Put your final answer in the form \\boxed{A} using exactly one choice label."
        gt = normalize_multiple_choice_answer(str(raw_task.get("answer") or ""), raw_task)
        if gt is None:
            raise ValueError("HLE multiple-choice record did not normalize to a valid choice label")
        return answer_rule, gt
    if format_type == "numeric_exact":
        answer_rule = "Put your final answer in the form \\boxed{answer} and preserve exact numeric structure."
        gt = normalize_numeric_exact_answer(str(raw_task.get("answer") or ""))
        if gt is None:
            raise ValueError("HLE numeric record did not normalize to a numeric answer")
        return answer_rule, gt
    answer_rule = "Put your final answer in the form \\boxed{answer}."
    gt = normalize_freeform_exact_answer(str(raw_task.get("answer") or ""))
    if gt is None:
        raise ValueError("HLE freeform record did not normalize to a comparable answer")
    return answer_rule, gt


def _image_part_specs(task_info: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for key in HLE_IMAGE_KEYS:
        value = _nonempty_string(task_info.get(key))
        if value:
            specs.append({"type": "image", "image_uri": value, "source_key": key})
    return specs


def has_images(task_info: dict[str, Any]) -> bool:
    return bool(_image_part_specs(prepare_task(task_info)))


def _image_part_specs_with_fallback(task_info: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for spec in _image_part_specs(task_info):
        specs.append(
            {
                **spec,
                "fallback_text": (
                    f"[Image unavailable: {spec['source_key']}={spec['image_uri']}. "
                    "Continue from the text prompt and any visible image references.]"
                ),
            }
        )
    return specs


def _render_image_note(task_info: dict[str, Any]) -> str:
    refs = []
    for spec in _image_part_specs(task_info):
        refs.append(f"{spec['source_key']}={spec['image_uri']}")
    if not refs:
        return ""
    return "\n\nImage references present (not attached in this text-only run):\n- " + "\n- ".join(refs)


def _build_prompt_text(*, raw_task: dict[str, Any], attach_images: bool) -> tuple[str, str]:
    question_text = str(raw_task.get("question") or "").strip()
    if not question_text:
        raise KeyError("HLE record missing question")
    answer_rule, gt = _answer_rule_and_gt(raw_task, question_text=question_text)
    if attach_images and _image_part_specs(raw_task):
        question_block = f"{question_text}\n\nRelevant images are attached with this prompt.\n\n{answer_rule}"
    else:
        question_block = f"{question_text}{_render_image_note(raw_task)}\n\n{answer_rule}"
    return AGENT_PROMPT["question"].format(question=question_block), gt


def build_judge_question(raw_task: dict[str, Any]) -> str:
    """Return the raw question text for the judge, without agent solving instructions."""
    task = prepare_task(raw_task)
    question_text = str(task.get("question") or "").strip()
    if not question_text:
        raise KeyError("HLE record missing question")
    answer_rule, _ = _answer_rule_and_gt(task, question_text=question_text)
    return f"{question_text}{_render_image_note(task)}\n\n{answer_rule}"


def build_initial_message(task_info: dict[str, Any], *, attach_images: bool = True) -> dict[str, Any]:
    raw_task = prepare_task(task_info)
    prompt_text, _gt = _build_prompt_text(raw_task=raw_task, attach_images=attach_images)
    return {"role": "user", "content": build_prompt_content(prompt_text, raw_task, attach_images=attach_images)}


def build_prompt_content(
    prompt_text: str,
    task_info: dict[str, Any],
    *,
    attach_images: bool = True,
    attachment_notice: str | None = None,
) -> str | list[dict[str, str]]:
    raw_task = prepare_task(task_info)
    image_specs = _image_part_specs_with_fallback(raw_task) if attach_images else []
    if not image_specs:
        return str(prompt_text)
    text = str(prompt_text)
    if attachment_notice:
        text = f"{text}\n\n{attachment_notice}"
    return [{"type": "text", "text": text}, *image_specs]


def parse_question_answer(sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    raw_task = prepare_task(sample)
    prompt, gt = _build_prompt_text(raw_task=raw_task, attach_images=False)
    return prompt, gt, raw_task


def normalize_multiple_choice_answer(value: str | None, task_info: dict[str, Any]) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    boxed = parse_math(s)
    if boxed is not None:
        s = boxed.strip()

    m = re.fullmatch(r"(?i)\(?\s*([A-Z])\s*\)?", s)
    if m:
        label = m.group(1).upper()
        if label in set(task_info.get("choice_labels") or task_info.get("choice_map", {}).keys()):
            return label

    choice_map = task_info.get("choice_map") or {}
    norm = _normalize_choice_text(s)
    if norm is None:
        return None
    for label, text in choice_map.items():
        if _normalize_choice_text(text) == norm:
            return label
    return None


def normalize_numeric_exact_answer(value: str | None) -> str | None:
    s = _normalize_exact_math(value)
    if s is None:
        return None
    s = s.replace("−", "-")
    if re.fullmatch(r"-?\d+", s):
        return normalize_numeric_string(s)
    tuple_match = re.fullmatch(r"([(\[])(.+)([)\]])", s)
    if tuple_match:
        inner = tuple_match.group(2)
        parts = [part for part in inner.split(",")]
        normalized_parts: list[str] = []
        for part in parts:
            token = part.strip()
            if re.fullmatch(r"-?\d+", token):
                norm = normalize_numeric_string(token)
                normalized_parts.append(norm if norm is not None else token)
            else:
                normalized_parts.append(token)
        return "(" + ",".join(normalized_parts) + ")"
    return s


def normalize_freeform_exact_answer(value: str | None) -> str | None:
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value))
    s = s.replace("\\n", "\n")
    s = re.sub(r"\\(?:text|mathrm|mathbf)\s*{([^{}]+)}", r"\1", s)
    s = re.sub(r"\s+(?=\\[a-zA-Z])", "", s)
    s = normalize_freeform_string(s)
    if s is None:
        return None
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _strip_markdown_bold_markers(text: str) -> str:
    """Remove inline markdown bold wrappers so cue regexes can see the underlying keyword."""
    return "\n".join(re.sub(r"\*\*([^*]+)\*\*", r"\1", line) for line in str(text).splitlines())


def _iter_candidate_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        s = value.strip()
        if s:
            yield s
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_candidate_strings(item)
        return
    if isinstance(value, dict):
        preferred_keys = (
            "accepted_answers",
            "acceptable_answers",
            "answer_aliases",
            "alternate_answers",
            "alternatives",
            "normalized_answer",
            "canonical_answer",
            "answer",
            "text",
            "value",
        )
        seen_any = False
        for key in preferred_keys:
            if key in value:
                seen_any = True
                yield from _iter_candidate_strings(value.get(key))
        if not seen_any:
            for nested in value.values():
                yield from _iter_candidate_strings(nested)
        return
    s = str(value).strip()
    if s:
        yield s


def _verified_freeform_answer_set(task_info: dict[str, Any]) -> list[str]:
    raw_task = prepare_task(task_info)
    candidate_sources: list[Any] = [raw_task.get("answer")]
    for key in (
        "accepted_answers",
        "acceptable_answers",
        "answer_aliases",
        "alternate_answers",
        "alternatives",
        "normalized_answer",
        "canonical_answer",
    ):
        if key in raw_task:
            candidate_sources.append(raw_task.get(key))
    verify_meta = raw_task.get("verify_meta_info")
    if isinstance(verify_meta, dict):
        candidate_sources.append(verify_meta)
        answer_verify = verify_meta.get("answer_verify")
        if isinstance(answer_verify, dict):
            candidate_sources.append(answer_verify)

    normalized: list[str] = []
    seen: set[str] = set()
    for source in candidate_sources:
        for candidate in _iter_candidate_strings(source):
            norm = normalize_freeform_exact_answer(candidate)
            if norm is None or norm in seen:
                continue
            seen.add(norm)
            normalized.append(norm)
    return normalized


def _extract_confidence(text: str) -> dict[str, Any]:
    stripped = text
    patterns = [
        re.compile(r"(?i)\bconfidence\b\s*[:=]\s*([01](?:\.\d+)?)\b"),
        re.compile(r"(?i)\bconfidence\b\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%"),
        re.compile(r"(?i)\bi am\s+(\d{1,3}(?:\.\d+)?)\s*%\s+confident\b"),
    ]
    for pattern in patterns:
        m = pattern.search(stripped)
        if not m:
            continue
        raw = m.group(0).strip()
        try:
            value = float(m.group(1))
        except Exception:
            return {"raw_text": raw, "value": None, "parse_failed": True}
        if "%" in raw:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return {"raw_text": raw, "value": value, "parse_failed": False}
        return {"raw_text": raw, "value": None, "parse_failed": True}
    if re.search(r"(?i)\bconfidence\b", stripped):
        line = next((ln.strip() for ln in stripped.splitlines() if "confidence" in ln.lower()), "")
        return {"raw_text": line or "confidence_mentioned", "value": None, "parse_failed": True}
    return {"raw_text": None, "value": None, "parse_failed": False}


def _extract_multiple_choice_candidate(text: str, task_info: dict[str, Any], *, strict: bool, recover: bool) -> str | None:
    stripped = text
    boxed = parse_math(stripped)
    if boxed is not None:
        normalized = normalize_multiple_choice_answer(boxed, task_info)
        if normalized is not None:
            return normalized
    stripped = _strip_markdown_bold_markers(stripped)

    labels = sorted(set(task_info.get("choice_labels") or task_info.get("choice_map", {}).keys()))
    if not labels:
        labels = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    label_class = "".join(re.escape(label) for label in labels)

    cue_pat = re.compile(
        r"(?im)^\s*(?:winning\s+answer|final\s+answer|answer|final\s+choice|choice|option|qed)\b(?:\s*(?:=+>|:|=|-)\s*|\s+)(.+?)\s*$"
    )
    last_label = None
    for m in cue_pat.finditer(stripped):
        candidate = m.group(1).strip()
        label = normalize_multiple_choice_answer(candidate, task_info)
        if label is not None:
            last_label = label
    if last_label is not None:
        return last_label

    inline_label_pat = re.compile(
        rf"(?i)\b(?:winning\s+answer|final\s+answer|answer|final\s+choice|choice|option)\b[^{label_class}]{{0,40}}\(?\s*([{label_class}])\s*\)?\b"
    )
    for m in inline_label_pat.finditer(stripped):
        label = normalize_multiple_choice_answer(m.group(1), task_info)
        if label is not None:
            return label

    if strict:
        return None

    tail = stripped[-2400:] if len(stripped) > 2400 else stripped
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if lines:
        label = normalize_multiple_choice_answer(lines[-1], task_info)
        if label is not None:
            return label

    if recover:
        for line in reversed(lines[-12:]):
            label = normalize_multiple_choice_answer(line, task_info)
            if label is not None:
                return label
    return None


def _extract_exact_candidate(text: str, *, strict: bool, recover: bool) -> str | None:
    stripped = text
    boxed = parse_math(stripped)
    if boxed is not None:
        return boxed.strip() or None
    stripped = _strip_markdown_bold_markers(stripped)

    cue_pat = re.compile(
        r"(?im)^\s*(?:winning\s+answer|final\s+answer|answer|final|qed)\b(?:\s*(?:=+>|:|=|-)\s*|\s+is\s+|\s+)(.+?)\s*$"
    )
    last_candidate = None
    for m in cue_pat.finditer(stripped):
        candidate = m.group(1).strip()
        candidate = candidate.splitlines()[0].strip()
        if candidate:
            last_candidate = candidate
    if last_candidate:
        return last_candidate

    if strict:
        return None

    tail = stripped[-4000:] if len(stripped) > 4000 else stripped
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    if recover:
        for line in reversed(lines[-12:]):
            if len(line) <= 256 and not re.match(r"(?i)^\s*confidence\s*[:=]", line):
                return line
    last = lines[-1]
    if re.match(r"(?i)^\s*confidence\s*[:=]", last):
        return None
    return last if len(last) <= 256 else None


def extract_response(
    text: str,
    task_info: dict[str, Any],
    *,
    parse_mode: Literal["default", "strict", "recover"] = "default",
) -> dict[str, Any]:
    raw_task = prepare_task(task_info)
    format_type = raw_task["answer_format_type"]
    strict = parse_mode == "strict"
    recover = parse_mode == "recover"

    def _extract_candidate(parse_with_recovery: bool) -> tuple[str | None, str | None]:
        if format_type == "multiple_choice":
            raw_candidate = _extract_multiple_choice_candidate(
                text,
                raw_task,
                strict=strict and not parse_with_recovery,
                recover=parse_with_recovery,
            )
            normalized_candidate = normalize_multiple_choice_answer(raw_candidate, raw_task) if raw_candidate is not None else None
            return raw_candidate, normalized_candidate
        if format_type == "numeric_exact":
            raw_candidate = _extract_exact_candidate(
                text,
                strict=strict and not parse_with_recovery,
                recover=parse_with_recovery,
            )
            normalized_candidate = normalize_numeric_exact_answer(raw_candidate) if raw_candidate is not None else None
            return raw_candidate, normalized_candidate
        raw_candidate = _extract_exact_candidate(
            text,
            strict=strict and not parse_with_recovery,
            recover=parse_with_recovery,
        )
        normalized_candidate = normalize_freeform_exact_answer(raw_candidate) if raw_candidate is not None else None
        return raw_candidate, normalized_candidate

    candidate, normalized = _extract_candidate(parse_with_recovery=recover)
    recovered_default = False
    if parse_mode == "default" and normalized is None:
        candidate, normalized = _extract_candidate(parse_with_recovery=True)
        recovered_default = normalized is not None

    confidence = _extract_confidence(text)
    diagnostics = {
        "parse_mode": parse_mode,
        "default_recovery_used": recovered_default,
        "answer_format_type": format_type,
        "choice_labels": list(raw_task.get("choice_labels") or []),
        "candidate_answer": candidate,
    }
    return {
        "extractor_provenance": HLE_EXTRACTOR_PROVENANCE,
        "raw_output": str(text),
        "answer_format_type": format_type,
        "choice_labels": list(raw_task.get("choice_labels") or []),
        "candidate_answer": candidate,
        "normalized_answer": normalized,
        "normalized_confidence": confidence["value"],
        "confidence_raw_text": confidence["raw_text"],
        "confidence_parse_failed": bool(confidence["parse_failed"]),
        "parse_success": normalized is not None,
        "parse_diagnostics": diagnostics,
    }


def parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    return extract_response(text, task_info, parse_mode="default").get("normalized_answer")


def strict_parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    return extract_response(text, task_info, parse_mode="strict").get("normalized_answer")


def recover_parse_answer(text: str, task_info: dict[str, Any]) -> str | None:
    return extract_response(text, task_info, parse_mode="recover").get("normalized_answer")


def score_answer(answer: str | None, task_info: dict[str, Any]) -> dict[str, Any]:
    raw_task = prepare_task(task_info)
    format_type = raw_task["answer_format_type"]
    expected_raw = str(raw_task.get("answer") or "")
    if format_type == "multiple_choice":
        expected = normalize_multiple_choice_answer(expected_raw, raw_task)
        actual = normalize_multiple_choice_answer(answer, raw_task) if answer is not None else None
        match_type = "multiple_choice_label"
    elif format_type == "numeric_exact":
        expected = normalize_numeric_exact_answer(expected_raw)
        actual = normalize_numeric_exact_answer(answer) if answer is not None else None
        match_type = "numeric_exact"
    else:
        accepted_answers = _verified_freeform_answer_set(raw_task)
        expected = accepted_answers[0] if accepted_answers else normalize_freeform_exact_answer(expected_raw)
        actual = normalize_freeform_exact_answer(answer) if answer is not None else None
        match_type = "freeform_verified_rule_set"
    if format_type == "freeform_exact":
        correct = int(actual is not None and ((expected is not None and actual == expected) or actual in accepted_answers))
    else:
        correct = int(actual is not None and expected is not None and actual == expected)
    result = {
        "correct": correct,
        "expected_answer": expected,
        "predicted_answer": actual,
        "answer_format_type": format_type,
        "match_type": match_type,
        "scorer_provenance": HLE_SCORER_PROVENANCE,
    }
    if format_type == "freeform_exact":
        result["accepted_answers"] = accepted_answers
    return result


def check_answer_correctness(answer: Any, gt: Any, task_info: dict[str, Any] | None = None) -> int:
    if task_info is None:
        if answer is None or gt is None:
            return 0
        return int(str(answer) == str(gt))
    return int(score_answer(None if answer is None else str(answer), task_info)["correct"])


def construct_debate_message(other_agent_answers: list[str], *, phase: str = "generic") -> dict[str, str]:
    if phase in {"critique", "defense"}:
        return build_structured_debate_message(
            phase=phase,
            updates=other_agent_answers,
            final_answer_instruction="using the same strict format required in the original question.",
        )
    body_lines = [AGENT_PROMPT["debate"][0]]
    for idx, answer in enumerate(other_agent_answers, start=1):
        body_lines.append(f"Agent {idx}: {answer}")
    body_lines.append(AGENT_PROMPT["debate"][1])
    return {
        "role": "user",
        "content": "\n".join(body_lines),
    }
