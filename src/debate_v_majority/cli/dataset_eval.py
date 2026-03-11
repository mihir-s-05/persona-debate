from __future__ import annotations

from typing import Any, cast

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from .engine_runtime import _engine_backend_name, _provider_is_gemini


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


def _construct_debate_message(dataset: DatasetName, other_agent_answers: list[str]) -> dict[str, str]:
    """Construct debate prompt."""
    adapter = _get_dataset_adapter(dataset)
    return adapter.construct_debate_message(other_agent_answers)


def _build_initial_user_message(
    *,
    dataset: DatasetName,
    question: str,
    raw_task: dict[str, Any],
    engine: Any,
) -> dict[str, Any]:
    if dataset == "hle" and _provider_is_gemini(_engine_backend_name(engine)):
        mod = _get_dataset_module(dataset)
        builder = getattr(mod, "build_initial_message", None)
        if callable(builder):
            return cast(dict[str, Any], builder(raw_task, attach_images=True))
    return {"role": "user", "content": question}


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
