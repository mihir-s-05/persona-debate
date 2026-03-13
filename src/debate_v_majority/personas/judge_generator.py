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



def _heuristic_judge_card(*, family: str, question: str, mode: JudgePersonaMode) -> JudgeCard:
    if mode == "neutral_baseline":
        return JudgeCard(
            judge_id="judge_neutral_baseline",
            judge_family="neutral_baseline",
            domain_scope="general_reasoning",
            evaluation_priorities=[
                "prefer the answer best supported by the transcript",
                "avoid solving the item from scratch when the transcript already settles it",
            ],
            tie_break_policy="prefer the answer backed by the clearest explicit argument and consistency checks",
            independent_resolve_policy="limited_check_only",
            answer_format_policy="return one final answer in strict dataset format",
            confidence_policy=None,
            system_prompt=JUDGE_GUIDANCE,
            card_version=JUDGE_PROMPT_VERSION,
            source={"backend": "heuristic", "family": "neutral_baseline"},
        )
    if mode == "question_conditioned_generated":
        judge_family = f"{family}:question_conditioned"
        scope = question[:120].strip()
    else:
        judge_family = family
        scope = family
    return JudgeCard(
        judge_id=f"judge_{judge_family}",
        judge_family=judge_family,
        domain_scope=scope,
        evaluation_priorities=[
            "score the arguments already present in the debate transcript",
            "prefer internally consistent arguments that engage concrete constraints",
            "separate strong argument quality from mere fluency",
        ],
        tie_break_policy="if two answers are close, prefer the one with clearer constraint tracking and fewer unsupported jumps",
        independent_resolve_policy="limited_check_only",
        answer_format_policy="return one final answer in the strict format expected by the dataset",
        confidence_policy="optional",
        system_prompt=(
            f"{JUDGE_GUIDANCE}\n"
            f"Judge family: {judge_family}\n"
            "Primary job: evaluate the transcript and select the strongest supported final answer."
        ),
        card_version=JUDGE_PROMPT_VERSION,
        source={"backend": "heuristic", "family": judge_family},
    )


def build_judge_card(
    *,
    dataset: str,
    raw_task: dict[str, object],
    question: str,
    mode: JudgePersonaMode = "task_family_generated",
    benchmark_family_override: str | None = None,
    engine: Any | None = None,
    generator_model: str | None = None,
    backend: str = "heuristic",
) -> JudgeCard | None:
    # benchmark_family_bank mode is resolved at the runner level; this path provides per-question fallback.
    family = str(benchmark_family_override or infer_benchmark_family(dataset, raw_task))
    if backend != "heuristic" and engine is not None:
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
                    "backend": backend,
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
    return _heuristic_judge_card(family=family, question=question, mode=mode)
