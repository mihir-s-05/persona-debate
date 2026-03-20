from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from ..engines import (
    InferenceEngine,
    InferenceResult,
    create_inference_engine,
    inference_result_metadata,
)
from ..personas import PersonaArtifact
from ..shared import PromptTokenCounter
from .judge_common import _count_prompt_tokens


def _engine_backend_name(engine: Any) -> str:
    for attr in ("backend_name", "provider_name", "backend"):
        value = getattr(engine, attr, None)
        if value:
            return str(value)
    return type(engine).__name__


def _provider_is_gemini(provider_name: str | None) -> bool:
    return str(provider_name or "").strip().lower() == "gemini"


def _create_role_engine(
    *,
    model_name: str,
    provider: str | None,
    model_role: str,
    context_len: int | None,
) -> InferenceEngine:
    return create_inference_engine(
        model_name=model_name,
        provider=provider,
        model_role=model_role,
        max_model_len=context_len,
    )


def _reuse_or_create_role_engine(
    *,
    existing_engine: InferenceEngine | None,
    existing_model_name: str | None,
    existing_provider_name: str | None,
    target_model_name: str,
    target_provider_name: str | None,
    model_role: str,
    context_len: int | None,
    quiet: bool,
    status_label: str,
) -> InferenceEngine:
    if (
        existing_engine is not None
        and target_model_name == existing_model_name
        and target_provider_name == existing_provider_name
    ):
        return existing_engine
    if not quiet:
        print(f"[engine] Creating {status_label} engine for {target_model_name}...", file=sys.stderr)
    return _create_role_engine(
        model_name=target_model_name,
        provider=target_provider_name,
        model_role=model_role,
        context_len=context_len,
    )


def _token_counts_from_result(result: InferenceResult) -> dict[str, Any]:
    usage = dict(result.usage or {})
    prompt_tokens = usage.get("prompt_token_count")
    visible_output_tokens = usage.get("candidates_token_count")
    thought_output_tokens = usage.get("thoughts_token_count")
    billable_output_tokens = None
    if visible_output_tokens is not None:
        billable_output_tokens = int(visible_output_tokens) + int(thought_output_tokens or 0)

    total_tokens = usage.get("total_token_count")
    cached_input_tokens = usage.get("cached_content_token_count")
    uncached_input_tokens = None
    if prompt_tokens is not None:
        uncached_input_tokens = int(prompt_tokens)
        if cached_input_tokens is not None:
            uncached_input_tokens = max(0, int(prompt_tokens) - int(cached_input_tokens))
    if total_tokens is None and prompt_tokens is not None and billable_output_tokens is not None:
        total_tokens = int(prompt_tokens) + int(billable_output_tokens)

    return {
        "input_tokens": None if prompt_tokens is None else int(prompt_tokens),
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": billable_output_tokens,
        "visible_output_tokens": None if visible_output_tokens is None else int(visible_output_tokens),
        "thought_output_tokens": None if thought_output_tokens is None else int(thought_output_tokens),
        "billable_output_tokens": billable_output_tokens,
        "total_tokens": None if total_tokens is None else int(total_tokens),
        "cached_input_tokens": None if cached_input_tokens is None else int(cached_input_tokens),
    }


def _merge_token_counts(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
    valid = [entry for entry in entries if entry is not None]
    input_tokens = sum(int(entry["input_tokens"]) for entry in valid if entry.get("input_tokens") is not None)
    uncached_input_tokens = sum(int(entry["uncached_input_tokens"]) for entry in valid if entry.get("uncached_input_tokens") is not None)
    output_tokens = sum(int(entry["output_tokens"]) for entry in valid if entry.get("output_tokens") is not None)
    visible_output_tokens = sum(int(entry["visible_output_tokens"]) for entry in valid if entry.get("visible_output_tokens") is not None)
    thought_output_tokens = sum(int(entry["thought_output_tokens"]) for entry in valid if entry.get("thought_output_tokens") is not None)
    billable_output_tokens = sum(int(entry["billable_output_tokens"]) for entry in valid if entry.get("billable_output_tokens") is not None)
    total_tokens = sum(int(entry["total_tokens"]) for entry in valid if entry.get("total_tokens") is not None)
    cached_input_tokens = sum(int(entry["cached_input_tokens"]) for entry in valid if entry.get("cached_input_tokens") is not None)
    n_calls = sum(int(entry.get("n_calls", 1) or 0) for entry in valid)
    return {
        "n_calls": n_calls,
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "visible_output_tokens": visible_output_tokens,
        "thought_output_tokens": thought_output_tokens,
        "billable_output_tokens": billable_output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "has_estimated_inputs": any(
            bool(entry.get("input_tokens_estimated")) or bool(entry.get("has_estimated_inputs"))
            for entry in valid
        ),
        "has_estimated_outputs": any(
            bool(entry.get("output_tokens_estimated")) or bool(entry.get("has_estimated_outputs"))
            for entry in valid
        ),
    }


def _inference_result_meta(
    result: InferenceResult | None,
    *,
    request_messages: list[dict[str, Any]] | None = None,
    engine: Any | None = None,
    prompt_token_counter: PromptTokenCounter | None = None,
) -> dict[str, Any] | None:
    meta = inference_result_metadata(result)
    if meta is None:
        return None
    meta["token_counts"] = _token_counts_from_result(cast(InferenceResult, result))
    request_prompt_tokens = None
    if request_messages is not None and engine is not None:
        request_prompt_tokens = _count_prompt_tokens(
            engine=engine,
            counter=prompt_token_counter,
            messages=request_messages,
        )
    meta["request_message_token_counts"] = (
        None
        if request_prompt_tokens is None
        else {"prompt_tokens": int(request_prompt_tokens)}
    )
    return meta


def _normalize_sampling_kwargs_for_engine(
    engine: Any,
    sampling_kwargs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    del engine
    if sampling_kwargs is None:
        return None
    return dict(sampling_kwargs)


def _default_judge_max_tokens(engine: Any) -> int:
    if _provider_is_gemini(_engine_backend_name(engine)):
        return 32768
    return 4096


def _default_token_ledger_path() -> Path:
    return Path("out") / "token_ledger.jsonl"


def _persona_summaries(artifact: PersonaArtifact | None, *, n_agents: int | None = None) -> list[dict[str, Any] | None] | None:
    if artifact is None:
        return None
    if artifact.slot_layout is None:
        return [
            {
                "persona_id": card.persona_id,
                "title": card.title,
                "core_reasoning_strategy": card.core_reasoning_strategy,
                "short_rule": next(
                    (descriptor.short_rule for descriptor in artifact.descriptors if descriptor.persona_id == card.persona_id),
                    None,
                ),
            }
            for card in artifact.cards
        ]
    total = n_agents if n_agents is not None else artifact.n_total_agents
    summaries: list[dict[str, Any] | None] = []
    for agent_idx in range(total):
        card = artifact.card_for_agent(agent_idx)
        if card is None:
            summaries.append(None)
        else:
            summaries.append({
                "persona_id": card.persona_id,
                "title": card.title,
                "core_reasoning_strategy": card.core_reasoning_strategy,
                "short_rule": next(
                    (descriptor.short_rule for descriptor in artifact.descriptors if descriptor.persona_id == card.persona_id),
                    None,
                ),
            })
    return summaries


def _judge_summary_from_card(card: Any | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {
        "judge_id": card.judge_id,
        "judge_family": card.judge_family,
        "domain_scope": card.domain_scope,
        "independent_resolve_policy": card.independent_resolve_policy,
        "answer_format_policy": card.answer_format_policy,
        "card_version": card.card_version,
    }


def _judge_summary(artifact: PersonaArtifact | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return _judge_summary_from_card(artifact.judge_card)


__all__ = [
    "_create_role_engine",
    "_default_judge_max_tokens",
    "_default_token_ledger_path",
    "_engine_backend_name",
    "_inference_result_meta",
    "_judge_summary",
    "_judge_summary_from_card",
    "_merge_token_counts",
    "_normalize_sampling_kwargs_for_engine",
    "_persona_summaries",
    "_provider_is_gemini",
    "_reuse_or_create_role_engine",
    "_token_counts_from_result",
]
