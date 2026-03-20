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


class AxisGenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


FIXED_AXIS_BANK: list[Axis] = [
    Axis(
        axis_id="symbolic_vs_intuitive",
        name="Symbolic vs Intuitive",
        kind="fixed",
        low_desc="Lean on equations, symbolic manipulation, and explicit formal structure.",
        high_desc="Lean on intuitive structure, quick plausibility checks, and conceptual compression.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="exhaustive_vs_pruning",
        name="Exhaustive vs Pruning",
        kind="fixed",
        low_desc="Enumerate cases carefully before committing.",
        high_desc="Prune aggressively and focus only on the most promising path.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="propose_first_vs_verify_first",
        name="Propose First vs Verify First",
        kind="fixed",
        low_desc="Generate a candidate approach early, then test it.",
        high_desc="Delay commitment until invariants and constraints are checked.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="local_vs_global_focus",
        name="Local vs Global Focus",
        kind="fixed",
        low_desc="Work from local components and build upward.",
        high_desc="Start with global structure and back out the local details.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="mechanistic_vs_elimination_reasoning",
        name="Mechanistic vs Elimination Reasoning",
        kind="fixed",
        low_desc="Construct the answer from mechanism or derivation.",
        high_desc="Eliminate impossible options and exploit contradictions.",
        source={"bank": "fixed_meta", "version": 1},
    ),
    Axis(
        axis_id="low_vs_high_skepticism_of_intermediates",
        name="Low vs High Skepticism of Intermediates",
        kind="fixed",
        low_desc="Treat intermediate steps as tentatively valid unless contradicted.",
        high_desc="Repeatedly audit intermediate steps for hidden assumptions and arithmetic slips.",
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


def _normalize_llm_axes(
    *,
    axes_payload: list[dict[str, Any]],
    count: int,
    benchmark_family: str,
    generator_model: str | None,
    backend: str,
    call_metadata: dict[str, Any],
) -> list[Axis]:
    normalized: list[Axis] = []
    seen_ids: set[str] = set()
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
            continue
        normalized.append(axis)
        seen_ids.add(axis.axis_id)
        if len(normalized) >= count:
            break
    return normalized


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
    messages = build_task_axis_messages(
        dataset=dataset,
        benchmark_family=benchmark_family,
        question=question,
        count=count,
        question_media=question_media,
    )
    attempt_audits: list[dict[str, Any]] = []
    for _attempt in range(AXIS_GENERATION_RETRIES + 1):
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
            llm_axes = _normalize_llm_axes(
                axes_payload=[dict(axis) for axis in axes_payload if isinstance(axis, dict)],
                count=count,
                benchmark_family=benchmark_family,
                generator_model=generator_model,
                backend="llm",
                call_metadata=call_metadata,
            )
            if len(llm_axes) >= count:
                return llm_axes[:count]
            attempt_audits.append(
                {
                    "request_messages": messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": None,
                    "call_metadata": call_metadata,
                    "accepted_axis_count": len(llm_axes),
                    "requested_axis_count": count,
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
