from __future__ import annotations

from typing import Any, cast

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from .engine_runtime import _engine_backend_name, _provider_is_gemini

STRUCTURED_INDEPENDENT_SOLVE_INSTRUCTION = (
    "Round 1 - Independent solve:\n"
    "Solve the task independently from scratch using only the task materials. "
    "Give your current best answer and the reasoning that supports it. "
    "Before the final answer, name one short unresolved weakness, caveat, or uncertainty in your own reasoning."
)


def _append_user_instruction(message: dict[str, Any], instruction: str) -> dict[str, Any]:
    updated = dict(message)
    content = updated.get("content")
    if isinstance(content, str):
        updated["content"] = f"{content}\n\n{instruction}"
        return updated
    if isinstance(content, list):
        parts = list(content)
        if parts and isinstance(parts[0], dict) and parts[0].get("type") == "text":
            first = dict(parts[0])
            first["text"] = f"{first.get('text', '')}\n\n{instruction}"
            parts[0] = first
            updated["content"] = parts
            return updated
    return updated


def _get_dataset_module(dataset: DatasetName):
    """Lazily import dataset-specific module."""
    return resolve_dataset_adapter(dataset).module


def _get_dataset_adapter(dataset: DatasetName):
    return resolve_dataset_adapter(dataset)


def _parse_question_answer(dataset: DatasetName, sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Parse a dataset sample."""
    adapter = _get_dataset_adapter(dataset)
    return cast(tuple[str, str, dict[str, Any]], adapter.parse_question_answer(sample))


def _parse_answer(dataset: DatasetName, text: str, raw_task: dict[str, Any]) -> str | None:
    """Parse an answer from model response."""
    adapter = _get_dataset_adapter(dataset)
    parsed = adapter.parse_answer(text, raw_task)
    if parsed is not None:
        return parsed
    recovered = adapter.recover_parse_answer(text, raw_task)
    if recovered is not None:
        return recovered
    return None


def _check_answer_correctness(
    dataset: DatasetName,
    answer: Any,
    gt: Any,
    raw_task: dict[str, Any] | None = None,
) -> int:
    """Check if answer is correct."""
    adapter = _get_dataset_adapter(dataset)
    return adapter.check_answer_correctness(answer, gt, raw_task)


def _construct_debate_message(
    dataset: DatasetName,
    other_agent_answers: list[str],
    *,
    phase: str = "generic",
) -> dict[str, str]:
    """Construct debate prompt."""
    adapter = _get_dataset_adapter(dataset)
    return adapter.construct_debate_message(other_agent_answers, phase=phase)


def _build_initial_user_message(
    *,
    dataset: DatasetName,
    question: str,
    raw_task: dict[str, Any],
    engine: Any,
    debate_protocol: str = "legacy",
) -> dict[str, Any]:
    if dataset == "hle" and _provider_is_gemini(_engine_backend_name(engine)):
        mod = _get_dataset_module(dataset)
        builder = getattr(mod, "build_initial_message", None)
        if callable(builder):
            message = cast(dict[str, Any], builder(raw_task, attach_images=True))
            if debate_protocol == "structured":
                return _append_user_instruction(message, STRUCTURED_INDEPENDENT_SOLVE_INSTRUCTION)
            return message
    message = {"role": "user", "content": question}
    if debate_protocol == "structured":
        return _append_user_instruction(message, STRUCTURED_INDEPENDENT_SOLVE_INSTRUCTION)
    return message


def _get_judge_prompt(dataset: DatasetName) -> dict[str, str]:
    """Get judge prompt configuration."""
    adapter = _get_dataset_adapter(dataset)
    return adapter.judge_prompt


__all__ = [
    "_build_initial_user_message",
    "_check_answer_correctness",
    "_construct_debate_message",
    "_get_dataset_adapter",
    "_get_dataset_module",
    "_get_judge_prompt",
    "_parse_answer",
    "_parse_question_answer",
]
