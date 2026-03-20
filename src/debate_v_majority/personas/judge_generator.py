from __future__ import annotations

from typing import Any

from ..engines import ensure_inference_results, inference_result_metadata
from .axes import infer_benchmark_family
from .prompt_templates import JUDGE_GUIDANCE, JUDGE_PROMPT_VERSION, build_judge_messages, parse_json_payload
from .schema import JudgeCard, JudgePersonaMode

JUDGE_CARD_MAX_TOKENS = 8192
JUDGE_GENERATION_RETRIES = 2


class JudgeGenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata

def build_judge_card(
    *,
    dataset: str,
    raw_task: dict[str, object],
    question: str,
    mode: JudgePersonaMode = "task_family_generated",
    benchmark_family_override: str | None = None,
    engine: Any | None = None,
    generator_model: str | None = None,
    backend: str = "llm",
) -> JudgeCard | None:
    # benchmark_family_bank mode is resolved at the runner level; this path provides per-question fallback.
    family = str(benchmark_family_override or infer_benchmark_family(dataset, raw_task))
    if engine is None:
        raise ValueError("persona generation requires a judge generator engine for judge-card generation")
    question_media = None
    if dataset == "hle":
        from ..datasets import hle as hle_dataset

        question_media = hle_dataset._image_part_specs_with_fallback(raw_task)
    messages = build_judge_messages(
        dataset=dataset,
        benchmark_family=family,
        question=question,
        mode=mode,
        question_media=question_media,
    )
    attempt_audits: list[dict[str, Any]] = []
    for attempt in range(JUDGE_GENERATION_RETRIES + 1):
        result = ensure_inference_results(
            engine,
            [messages],
            batch_size=1,
            sampling_kwargs={"max_tokens": JUDGE_CARD_MAX_TOKENS},
            model_role="judge_generator",
        )[0]
        raw_result_text = str(result.text)
        call_metadata = inference_result_metadata(result)
        try:
            payload = parse_json_payload(raw_result_text)
        except ValueError as exc:
            attempt_audits.append(
                {
                    "attempt": attempt,
                    "request_messages": messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": str(exc),
                    "call_metadata": call_metadata,
                }
            )
            continue
        return JudgeCard(
            judge_id=str(payload.get("judge_id") or f"judge_{family}"),
            judge_family=str(payload.get("judge_family") or family),
            domain_scope=str(payload.get("domain_scope") or family),
            evaluation_priorities=[str(x) for x in payload.get("evaluation_priorities", [])],
            tie_break_policy=str(payload.get("tie_break_policy") or "prefer the clearest supported answer"),
            independent_resolve_policy=str(payload.get("independent_resolve_policy") or "limited_check_only"),
            answer_format_policy=str(payload.get("answer_format_policy") or "return one final answer in strict format"),
            confidence_policy=str(payload.get("confidence_policy")) if payload.get("confidence_policy") is not None else None,
            system_prompt=str(payload.get("system_prompt") or JUDGE_GUIDANCE),
            card_version=JUDGE_PROMPT_VERSION,
            source={
                "generator": generator_model or "llm",
                "family": family,
                "backend": "llm",
                "call_metadata": call_metadata,
            },
        )
    raise JudgeGenerationExhaustedError(
        f"Judge card generation exhausted retries for family={family}",
        metadata={
            "judge_family": family,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "attempt_audits": attempt_audits,
        },
    )
