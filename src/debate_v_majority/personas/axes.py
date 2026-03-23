from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..engines import ensure_inference_results, inference_result_metadata
from .prompt_templates import AXIS_PROMPT_VERSION, build_task_axis_messages, parse_json_payload
from .schema import Axis, AxisSelection
from .validators import validate_text_for_leakage

AXIS_GENERATION_RETRIES = 4

_TASK_AXIS_REJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(log|metadata|label|labels|clock|timeline|configuration|layout|visible|visual|current state|provenance|formatting)\b",
            re.IGNORECASE,
        ),
        "uses source-artifact language instead of a portable debate behavior",
    ),
    (
        re.compile(
            r"\b("
            r"theorem|lemma|corollary|incidence matrix|null space|flow theory|nowhere-zero flow|nowhere-zero flows|"
            r"graph snark|graph snarks|snark|snarks|functor|functors|equivalence relation|equivalence relations|"
            r"generating function|generating functions|burnside|cohen forcing|forcing notion|forcing notions|"
            r"contact ion pair|contact ion pairs|ion pairing|hydration shell|hydration shells|solubility|"
            r"activation energy|transition state|transition states|air-water interface|functional graph|"
            r"functional graphs|f e n|fen|sicilian defense|coend|isomorphism class|isomorphism classes"
            r")\b",
            re.IGNORECASE,
        ),
        "uses theorem, object-family, or mechanism language instead of a portable debate behavior",
    ),
    (
        re.compile(r"\boption\s+[A-E]\b|\banswer[s]?\s+like\b", re.IGNORECASE),
        "references answer options instead of a benchmark-agnostic reasoning distinction",
    ),
]


class AxisGenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


FIXED_AXIS_BANK: list[Axis] = [
    Axis(
        axis_id="revision_resistance_vs_revision_readiness",
        name="Revision Resistance vs Revision Readiness",
        kind="fixed",
        low_desc="Keep the current answer unless a peer identifies a specific error, contradiction, or missing constraint that materially weakens it.",
        high_desc="Re-open the current answer quickly when a peer presents a concrete objection or a better-supported alternative, even if the current answer still seems plausible.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="defend_vs_rebuild_after_critique",
        name="Defend vs Rebuild After Critique",
        kind="fixed",
        low_desc="After criticism, start by defending and repairing the current line of reasoning before considering a full rethink.",
        high_desc="After criticism, temporarily rebuild the answer from the ground up before deciding whether the original line still survives.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="self_repair_vs_peer_pressure",
        name="Self-Repair vs Peer Pressure",
        kind="fixed",
        low_desc="Use later rounds mainly to tighten your own reasoning, patch weak steps, and check whether your answer still holds.",
        high_desc="Use later rounds mainly to pressure-test competing answers, expose their weak links, and raise the bar they must clear.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="contradiction_hunting_vs_support_auditing",
        name="Contradiction Hunting vs Support Auditing",
        kind="fixed",
        low_desc="Critique by looking for a concrete break, conflicting implication, or direct inconsistency that can overturn a line of reasoning.",
        high_desc="Critique by asking whether the line is sufficiently supported at all, even if there is no single decisive contradiction yet.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="explicit_claim_auditing_vs_assumption_auditing",
        name="Explicit Claim Auditing vs Assumption Auditing",
        kind="fixed",
        low_desc="Check whether stated claims are actually justified by the reasoning that has been made explicit.",
        high_desc="Probe for hidden assumptions, silent dependencies, and places where the argument relies on more than it openly establishes.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="independent_anchor_vs_convergence_awareness",
        name="Independent Anchor vs Convergence Awareness",
        kind="fixed",
        low_desc="Treat other agents' agreement as mostly irrelevant and update only for the substance of their arguments.",
        high_desc="Treat broad convergence as a cue to re-audit your view, while still requiring argument-level reasons before changing your answer.",
        source={"bank": "fixed_meta", "version": 1},
    ),
]


def infer_benchmark_family(dataset: str, raw_task: dict[str, Any]) -> str:
    if dataset == "gpqa":
        return "science_multiple_choice"
    if dataset == "aime25":
        return "competition_math"
    if dataset == "hle":
        category = str(raw_task.get("category") or "").strip().lower()
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
    return str(raw_task.get("family") or f"{dataset}_reasoning")


def summarize_question(question: str, *, max_words: int = 24) -> str:
    words = re.findall(r"\S+", question)
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def get_fixed_axes(count: int) -> list[Axis]:
    if count <= 0:
        return []
    return FIXED_AXIS_BANK[: min(count, len(FIXED_AXIS_BANK))]

def _axis_text(axis: Axis) -> str:
    parts = [axis.name, axis.low_desc, axis.high_desc]
    if axis.notes:
        parts.append(axis.notes)
    return " ".join(str(part) for part in parts if part)


def _task_axis_rejection_reason(axis: Axis) -> str | None:
    text = " ".join(
        [
            axis.axis_id,
            axis.name,
            axis.low_desc,
            axis.high_desc,
            "" if axis.notes is None else str(axis.notes),
        ]
    )
    for pattern, reason in _TASK_AXIS_REJECTION_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _normalize_llm_axes(
    *,
    axes_payload: list[dict[str, Any]],
    count: int,
    benchmark_family: str,
    generator_model: str | None,
    backend: str,
    call_metadata: dict[str, Any],
) -> tuple[list[Axis], list[dict[str, str]]]:
    normalized: list[Axis] = []
    seen_ids: set[str] = set()
    rejected: list[dict[str, str]] = []
    for raw_axis in axes_payload:
        axis_id = str(raw_axis.get("axis_id") or "").strip()
        name = str(raw_axis.get("name") or "").strip()
        low_desc = str(raw_axis.get("low_desc") or "").strip()
        high_desc = str(raw_axis.get("high_desc") or "").strip()
        if not axis_id or not name or not low_desc or not high_desc:
            continue
        if axis_id in seen_ids:
            continue
        axis = Axis(
            axis_id=axis_id,
            name=name,
            kind="task",
            low_desc=low_desc,
            high_desc=high_desc,
            notes=str(raw_axis.get("notes")).strip() if raw_axis.get("notes") is not None else None,
            source={
                "generator": generator_model or "llm",
                "family": benchmark_family,
                "backend": backend,
                "call_metadata": call_metadata,
            },
        )
        validation = validate_text_for_leakage(_axis_text(axis))
        if validation.status != "accept":
            rejected.append(
                {
                    "axis_id": axis.axis_id,
                    "reason": ", ".join(validation.reasons) or "failed axis text validation",
                }
            )
            continue
        bad_reason = _task_axis_rejection_reason(axis)
        if bad_reason is not None:
            rejected.append({"axis_id": axis.axis_id, "reason": bad_reason})
            continue
        normalized.append(axis)
        seen_ids.add(axis.axis_id)
        if len(normalized) >= count:
            break
    return normalized, rejected


def _build_task_axis_retry_feedback(
    *,
    parse_error: str | None,
    accepted_axis_count: int,
    requested_axis_count: int,
    rejected_axes: list[dict[str, str]],
) -> str:
    parts = ["Your previous axis proposal was not accepted. Issues found:"]
    if parse_error:
        parts.append(f"- JSON parse failure: {parse_error[:300]}")
    if accepted_axis_count < requested_axis_count:
        parts.append(f"- Only {accepted_axis_count} of {requested_axis_count} proposed axes were accepted.")
    for row in rejected_axes[:8]:
        parts.append(f"- {row.get('axis_id') or 'unknown_axis'}: {row.get('reason') or 'rejected'}")
    parts.append("")
    parts.append("Regenerate the full axis set.")
    parts.append("Requirements:")
    parts.append("- Propose only portable debate behaviors, not theorem families, object classes, or benchmark mechanisms")
    parts.append("- Frame low/high ends as evidence standard, comparison rule, search policy, pressure test, or revision trigger")
    parts.append("- If an axis reveals what kind of object is being analyzed, rewrite it to a general debate behavior")
    parts.append("- Do not mention answer options, logs, metadata, graph families, forcing notions, chemistry mechanisms, or other hidden ontology")
    parts.append("- Return ONLY valid JSON matching the schema")
    return "\n".join(parts)


def generate_task_axes(
    *,
    dataset: str,
    question: str,
    raw_task: dict[str, Any] | None = None,
    benchmark_family: str,
    count: int,
    generator_model: str | None = None,
    engine: Any | None = None,
    backend: str = "llm",
) -> list[Axis]:
    if count <= 0:
        return []
    if engine is None:
        raise ValueError("persona generation requires a generator engine for task-axis generation")
    question_media = None
    if dataset == "hle" and raw_task is not None:
        from ..datasets import hle as hle_dataset

        question_media = hle_dataset._image_part_specs_with_fallback(raw_task)
    base_messages = build_task_axis_messages(
        dataset=dataset,
        benchmark_family=benchmark_family,
        question=question,
        count=count,
        question_media=question_media,
    )
    attempt_audits: list[dict[str, Any]] = []
    retry_context: list[dict[str, Any]] | None = None
    for _attempt in range(AXIS_GENERATION_RETRIES + 1):
        messages = base_messages if retry_context is None else base_messages + retry_context
        result = ensure_inference_results(
            engine,
            [messages],
            batch_size=1,
            sampling_kwargs={"max_tokens": 4096},
            model_role="generator",
        )[0]
        raw_result_text = str(result.text)
        call_metadata = inference_result_metadata(result)
        try:
            payload = parse_json_payload(raw_result_text)
        except ValueError as exc:
            retry_context = [
                {
                    "role": "user",
                    "content": _build_task_axis_retry_feedback(
                        parse_error=str(exc),
                        accepted_axis_count=0,
                        requested_axis_count=count,
                        rejected_axes=[],
                    ),
                }
            ]
            attempt_audits.append(
                {
                    "request_messages": messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": str(exc),
                    "call_metadata": call_metadata,
                }
            )
            continue
        axes_payload = payload.get("axes") or []
        if isinstance(axes_payload, list):
            llm_axes, rejected_axes = _normalize_llm_axes(
                axes_payload=[dict(axis) for axis in axes_payload if isinstance(axis, dict)],
                count=count,
                benchmark_family=benchmark_family,
                generator_model=generator_model,
                backend="llm",
                call_metadata=call_metadata,
            )
            if len(llm_axes) >= count:
                return llm_axes[:count]
            retry_context = [
                {
                    "role": "user",
                    "content": _build_task_axis_retry_feedback(
                        parse_error=None,
                        accepted_axis_count=len(llm_axes),
                        requested_axis_count=count,
                        rejected_axes=rejected_axes,
                    ),
                }
            ]
            attempt_audits.append(
                {
                    "request_messages": messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": None,
                    "call_metadata": call_metadata,
                    "accepted_axis_count": len(llm_axes),
                    "requested_axis_count": count,
                    "rejected_axes": rejected_axes,
                }
            )
            continue
    raise AxisGenerationExhaustedError(
        f"Task-axis generation exhausted retries for benchmark_family={benchmark_family}",
        metadata={
            "dataset": dataset,
            "benchmark_family": benchmark_family,
            "requested_axis_count": count,
            "generator_model": generator_model,
            "attempt_audits": attempt_audits,
        },
    )


def _load_axes_file(path: Path) -> list[Axis]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("axes", [])
    if not isinstance(data, list):
        raise ValueError(f"Axis file must contain a list or {{'axes': [...]}} payload: {path}")
    return [Axis.from_dict(dict(row)) for row in data]


def build_axis_selection(
    *,
    mode: str,
    question: str,
    dataset: str,
    raw_task: dict[str, Any],
    fixed_count: int,
    task_count: int,
    generator_model: str | None,
    engine: Any | None = None,
    backend: str = "llm",
    axes_file: Path | None = None,
) -> AxisSelection:
    benchmark_family = infer_benchmark_family(dataset, raw_task)
    if mode == "fixed":
        axes = get_fixed_axes(fixed_count)
    elif mode == "task":
        axes = generate_task_axes(
            dataset=dataset,
            question=question,
            raw_task=raw_task,
            benchmark_family=benchmark_family,
            count=task_count,
            generator_model=generator_model,
            engine=engine,
            backend=backend,
        )
    elif mode == "file":
        if axes_file is None:
            raise ValueError("axis mode 'file' requires an axes_file path")
        axes = _load_axes_file(axes_file)
    else:
        axes = get_fixed_axes(fixed_count) + generate_task_axes(
            dataset=dataset,
            question=question,
            raw_task=raw_task,
            benchmark_family=benchmark_family,
            count=task_count,
            generator_model=generator_model,
            engine=engine,
            backend=backend,
        )
    return AxisSelection(
        mode=mode if mode in {"fixed", "task", "hybrid", "file", "replay"} else "hybrid",
        axes=axes,
        benchmark_family=benchmark_family,
        question_summary=summarize_question(question),
        generator_prompt_version=AXIS_PROMPT_VERSION,
        generator_model=generator_model,
    )
