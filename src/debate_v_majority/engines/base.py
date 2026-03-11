from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class InferenceResult:
    text: str
    thought_summary: str | None = None
    thought_summary_available: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    error: dict[str, Any] | None = None
    model_role: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    token_budget: dict[str, Any] = field(default_factory=dict)


class BaseInferenceEngine:
    provider_name = "unknown"

    def __init__(self, *, model_name: str, model_role: str | None = None) -> None:
        self.model_name = model_name
        self.model_role = model_role

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[str]:
        return results_to_texts(
            self.generate_batch_results(
                contexts,
                batch_size=batch_size,
                sampling_kwargs=sampling_kwargs,
                progress_callback=progress_callback,
                model_role=model_role,
            )
        )

    def generate_batch_results(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[InferenceResult]:
        raise NotImplementedError

    def shutdown(self) -> None:
        return None

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        raise NotImplementedError


@runtime_checkable
class SupportsInferenceResults(Protocol):
    model_name: str

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[str]: ...

    def generate_batch_results(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[InferenceResult]: ...

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int: ...


def make_text_result(
    text: str,
    *,
    model_name: str | None,
    provider_name: str | None,
    model_role: str | None = None,
    thought_summary: str | None = None,
    thought_summary_available: bool = False,
    usage: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    provider_meta: dict[str, Any] | None = None,
    retries: int = 0,
    error: dict[str, Any] | None = None,
    token_budget: dict[str, Any] | None = None,
) -> InferenceResult:
    return InferenceResult(
        text=str(text),
        thought_summary=None if thought_summary is None else str(thought_summary),
        thought_summary_available=bool(thought_summary_available),
        usage=dict(usage or {}),
        latency_ms=latency_ms,
        provider_meta=dict(provider_meta or {}),
        retries=int(retries),
        error=None if error is None else dict(error),
        model_role=model_role,
        model_name=model_name,
        provider_name=provider_name,
        token_budget=dict(token_budget or {}),
    )


def results_to_texts(results: list[InferenceResult]) -> list[str]:
    return [str(result.text) for result in results]


def _call_legacy_generate_batch(
    engine: Any,
    contexts: list[list[dict[str, str]]],
    batch_size: int | None = None,
    *,
    sampling_kwargs: dict[str, Any] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    model_role: str | None = None,
) -> list[str]:
    generate_batch = engine.generate_batch
    try:
        parameters = inspect.signature(generate_batch).parameters
    except (TypeError, ValueError):
        return list(generate_batch(contexts))

    args: list[Any] = [contexts]
    kwargs: dict[str, Any] = {}
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    for name, value in (
        ("batch_size", batch_size),
        ("sampling_kwargs", sampling_kwargs),
        ("progress_callback", progress_callback),
        ("model_role", model_role),
    ):
        parameter = parameters.get(name)
        if parameter is None:
            if accepts_var_kwargs:
                kwargs[name] = value
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
            continue
        if parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            kwargs[name] = value
    return list(generate_batch(*args, **kwargs))


def ensure_inference_results(
    engine: Any,
    contexts: list[list[dict[str, str]]],
    batch_size: int | None = None,
    *,
    sampling_kwargs: dict[str, Any] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    model_role: str | None = None,
) -> list[InferenceResult]:
    if hasattr(engine, "generate_batch_results"):
        return list(
            engine.generate_batch_results(
                contexts,
                batch_size=batch_size,
                sampling_kwargs=sampling_kwargs,
                progress_callback=progress_callback,
                model_role=model_role,
            )
        )
    texts = _call_legacy_generate_batch(
        engine,
        contexts,
        batch_size=batch_size,
        sampling_kwargs=sampling_kwargs,
        progress_callback=progress_callback,
        model_role=model_role,
    )
    model_name = getattr(engine, "model_name", None)
    provider_name = getattr(engine, "provider_name", None) or "legacy_text_only"
    return [
        make_text_result(
            str(text),
            model_name=str(model_name) if model_name is not None else None,
            provider_name=str(provider_name),
            model_role=model_role,
        )
        for text in texts
    ]


def inference_token_counts(
    result: InferenceResult | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    usage = dict(result.usage or {})
    prompt_tokens = usage.get("prompt_token_count")
    visible_output_tokens = usage.get("candidates_token_count")
    thought_output_tokens = usage.get("thoughts_token_count")
    billable_output_tokens = None
    if visible_output_tokens is not None:
        billable_output_tokens = int(visible_output_tokens) + int(thought_output_tokens or 0)
    total_tokens = usage.get("total_token_count")
    cached_input_tokens = usage.get("cached_content_token_count")
    if total_tokens is None and prompt_tokens is not None and billable_output_tokens is not None:
        total_tokens = int(prompt_tokens) + int(billable_output_tokens)
    uncached_input_tokens = None
    if prompt_tokens is not None:
        uncached_input_tokens = int(prompt_tokens)
        if cached_input_tokens is not None:
            uncached_input_tokens = max(0, int(prompt_tokens) - int(cached_input_tokens))
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


def inference_result_metadata(result: InferenceResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "provider_name": result.provider_name,
        "model_name": result.model_name,
        "model_role": result.model_role,
        "thought_summary": result.thought_summary,
        "thought_summary_available": bool(result.thought_summary_available),
        "usage": dict(result.usage),
        "latency_ms": result.latency_ms,
        "provider_meta": dict(result.provider_meta),
        "retries": result.retries,
        "error": None if result.error is None else dict(result.error),
        "token_budget": dict(result.token_budget),
        "token_counts": inference_token_counts(result),
    }


def serialize_inference_result(result: InferenceResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return asdict(result)
