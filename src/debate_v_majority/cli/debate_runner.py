from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

from .. import DatasetName
from ..engines import InferenceEngine, ensure_inference_results, infer_native_context_len
from ..personas import JudgeCard, PersonaArtifact
from ..shared import PrevJudgeInfo, PromptTokenCounter
from .dataset_eval import (
    _build_initial_user_message,
    _check_answer_correctness,
    _construct_debate_message,
    _parse_question_answer,
)
from .engine_runtime import (
    _build_gemini_judge_cache_context,
    _build_gemini_debate_cache_context,
    _default_judge_max_tokens,
    _engine_backend_name,
    _inference_result_meta,
    _judge_summary_from_card,
    _merge_token_counts,
    _normalize_sampling_kwargs_for_engine,
    _persona_summaries,
    _provider_is_gemini,
)
from .judge import JUDGE_RETRY_NUDGE, _build_judge_context, _build_judge_round_context, _parse_judge_output
from .judge_common import _count_prompt_tokens
from .persona_runtime import _resolve_persona_artifact, _resolve_runtime_judge_card
from .result_rows import (
    _base_row_fields,
    _compute_answer_changes,
    _compute_persona_fidelity_metrics,
    _compute_round_convergence,
    _compute_round_token_usage,
    _persona_runtime_meta,
    _vote_result_payload,
)
from .response_parsing import (
    _build_round_output,
    _extract_output_details,
    _format_debate_share_entry,
    _judge_trace_mode_enabled,
    _render_agent_round_outputs_for_judge,
)
from .stage_state import (
    append_stage_entry,
    load_latest_stage_entry_of_type,
    make_stage_entry,
    path_setting,
    subset_item_resume_signature,
)
from .subset import SubsetItem


DEBATE_DEFAULT_MAX_OUTPUT_TOKENS = 32768


class _DebateStopped(Exception):
    """Raised internally when debate_stop_after triggers an early exit."""

    def __init__(self, results_by_round: dict[int, list[dict[str, Any]]]):
        self.results_by_round = results_by_round


def _parse_debate_stage_name(stage_name: str) -> tuple[int, bool] | None:
    if not stage_name.startswith("round_"):
        return None
    suffix = stage_name[len("round_") :]
    is_judged = suffix.endswith("_judge")
    if is_judged:
        suffix = suffix[: -len("_judge")]
    try:
        return int(suffix), is_judged
    except ValueError:
        return None


def _serialise_prev_judge(prev: PrevJudgeInfo | None) -> dict[str, Any] | None:
    if prev is None:
        return None
    return {
        "start_round": int(prev.start_round),
        "end_round": int(prev.end_round),
        "parsed_answer": str(prev.parsed_answer),
        "raw_output": str(prev.raw_output),
    }


def _deserialise_prev_judge(data: dict[str, Any] | None) -> PrevJudgeInfo | None:
    if not data:
        return None
    return PrevJudgeInfo(
        start_round=int(data["start_round"]),
        end_round=int(data["end_round"]),
        parsed_answer=str(data["parsed_answer"]),
        raw_output=str(data["raw_output"]),
    )


def _build_initial_agent_contexts(
    *,
    dataset: DatasetName,
    parsed_items: list[tuple[SubsetItem, str, Any, dict[str, Any]]],
    engine: InferenceEngine,
    persona_artifacts: list[PersonaArtifact | None],
    n_agents: int,
) -> list[list[list[dict[str, Any]]]]:
    return [
        [
            [
                *(
                    [{"role": "system", "content": persona_artifacts[q_idx].cards[agent_idx].system_prompt}]
                    if persona_artifacts[q_idx] is not None
                    else []
                ),
                _build_initial_user_message(
                    dataset=dataset,
                    question=question,
                    raw_task=raw_task,
                    engine=engine,
                ),
            ]
            for agent_idx in range(n_agents)
        ]
        for q_idx, (_item, question, _gt_answer, raw_task) in enumerate(parsed_items)
    ]


def _build_agent_debate_message(
    *,
    dataset: DatasetName,
    previous_round_outputs: list[dict[str, Any]],
    agent_idx: int,
    persona_artifact: PersonaArtifact | None,
) -> dict[str, Any]:
    other_answers = [
        _format_debate_share_entry(previous_round_outputs[idx])
        for idx in range(len(previous_round_outputs))
        if idx != agent_idx
    ]
    debate_msg = _construct_debate_message(dataset, other_answers)
    if persona_artifact is None or agent_idx >= len(persona_artifact.cards):
        return debate_msg
    card = persona_artifact.cards[agent_idx]
    return {
        **debate_msg,
        "content": (
            f"[Reminder: You are reasoning as {card.title}. "
            f"Strategy: {card.core_reasoning_strategy}]\n\n"
            + debate_msg["content"]
        ),
    }


def _debate_persona_settings(
    *,
    use_personas: bool,
    runtime_judge_persona_enabled: bool,
    persona_seed: int,
    persona_axis_mode: str,
    persona_fixed_axis_count: int,
    persona_task_axis_count: int,
    persona_sampling_method: str,
    persona_judge_mode: str,
    persona_backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    persona_axes_file: Path | None,
    judge_bank_dir: Path | None,
    judge_bank_refresh: bool,
    gpqa_family_cache_path: Path | None,
) -> dict[str, Any]:
    return {
        "use_personas": bool(use_personas),
        "runtime_judge_persona_enabled": bool(runtime_judge_persona_enabled),
        "persona_seed": int(persona_seed),
        "persona_axis_mode": str(persona_axis_mode),
        "persona_fixed_axis_count": int(persona_fixed_axis_count),
        "persona_task_axis_count": int(persona_task_axis_count),
        "persona_sampling_method": str(persona_sampling_method),
        "persona_judge_mode": str(persona_judge_mode),
        "persona_backend": str(persona_backend),
        "generator_model": generator_model,
        "judge_generator_model": judge_generator_model,
        "persona_axes_file": path_setting(persona_axes_file),
        "judge_bank_dir": path_setting(judge_bank_dir),
        "judge_bank_refresh": bool(judge_bank_refresh),
        "gpqa_family_cache_path": path_setting(gpqa_family_cache_path),
    }


def _debate_runtime_settings(
    *,
    engine: InferenceEngine,
    judge_engine: InferenceEngine,
    judge_block_size: int | None,
    judge_sampling_kwargs: dict[str, Any] | None,
    judge_strict_final_only: bool,
    judge_recovery_parse_enabled: bool,
    judge_trace_mode: str,
    public_rationale_max_tokens: int,
) -> dict[str, Any]:
    return {
        "debater_model": getattr(engine, "model_name", None),
        "debater_backend": _engine_backend_name(engine),
        "judge_model": getattr(judge_engine, "model_name", None),
        "judge_backend": _engine_backend_name(judge_engine),
        "judge_block_size": None if judge_block_size is None else int(judge_block_size),
        "judge_sampling_kwargs": dict(judge_sampling_kwargs or {}),
        "judge_strict_final_only": bool(judge_strict_final_only),
        "judge_recovery_parse_enabled": bool(judge_recovery_parse_enabled),
        "judge_trace_mode": str(judge_trace_mode),
        "public_rationale_max_tokens": int(public_rationale_max_tokens),
    }


def _validate_resume_runtime_settings(
    *,
    resume_state: dict[str, Any],
    current_settings: dict[str, Any],
) -> None:
    saved_settings = resume_state.get("runtime_settings")
    if not isinstance(saved_settings, dict) or dict(saved_settings) != current_settings:
        raise ValueError("Debate state runtime settings mismatch")


def _normalise_judge_rounds(judge_rounds: list[int]) -> list[int]:
    return sorted({int(round_num) for round_num in judge_rounds})


def _validate_resume_persona_settings(
    *,
    resume_state: dict[str, Any],
    current_settings: dict[str, Any],
) -> None:
    saved_settings = resume_state.get("persona_settings")
    if not isinstance(saved_settings, dict) or dict(saved_settings) != current_settings:
        raise ValueError("Debate state persona settings mismatch")


def _require_resume_list_length(
    values: list[Any],
    *,
    expected: int,
    field_name: str,
) -> list[Any]:
    if len(values) != expected:
        raise ValueError(f"Debate state {field_name} length mismatch")
    return values


def run_debate(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    engine: InferenceEngine,
    n_agents: int,
    n_rounds: int,
    judge_rounds: list[int],
    batch_size: int | None,
    judge_block_size: int | None = None,
    judge_sampling_kwargs: dict[str, Any] | None = None,
    judge_strict_final_only: bool = True,
    judge_recovery_parse_enabled: bool = True,
    judge_engine: InferenceEngine | None = None,
    use_personas: bool = False,
    artifacts_dir: Path | None = None,
    persona_seed: int = 0,
    persona_axis_mode: str = "hybrid",
    persona_fixed_axis_count: int = 2,
    persona_task_axis_count: int = 1,
    persona_sampling_method: str = "maximin",
    persona_judge_mode: str = "task_family_generated",
    persona_backend: str = "auto",
    generator_model: str | None = None,
    judge_generator_model: str | None = None,
    persona_generator_engine: InferenceEngine | None = None,
    persona_judge_engine: InferenceEngine | None = None,
    persona_axes_file: Path | None = None,
    persona_save_artifacts: bool = False,
    persona_replay: bool = False,
    judge_trace_mode: str = "visible_plus_thought_summary",
    judge_bank_dir: Path | None = None,
    judge_bank_refresh: bool = False,
    gpqa_family_cache_path: Path | None = None,
    public_rationale_max_tokens: int = 96,
    enable_runtime_judge_persona: bool | None = None,
    persona_artifacts_by_item: dict[str, PersonaArtifact] | None = None,
    progress_file: TextIO = sys.stdout,
    debate_stop_after: str | None = None,
    stage_state_file: Path | None = None,
) -> dict[int, list[dict[str, Any]]]:
    judge_rounds = _normalise_judge_rounds(judge_rounds)
    debate_update_rounds = max(0, int(n_rounds))
    total_answer_rounds = debate_update_rounds + 1
    runtime_judge_persona_enabled = bool(use_personas) if enable_runtime_judge_persona is None else bool(enable_runtime_judge_persona)
    effective_judge_engine = judge_engine or engine
    current_persona_settings = _debate_persona_settings(
        use_personas=bool(use_personas),
        runtime_judge_persona_enabled=runtime_judge_persona_enabled,
        persona_seed=persona_seed,
        persona_axis_mode=persona_axis_mode,
        persona_fixed_axis_count=persona_fixed_axis_count,
        persona_task_axis_count=persona_task_axis_count,
        persona_sampling_method=persona_sampling_method,
        persona_judge_mode=persona_judge_mode,
        persona_backend=persona_backend,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        persona_axes_file=persona_axes_file,
        judge_bank_dir=judge_bank_dir,
        judge_bank_refresh=judge_bank_refresh,
        gpqa_family_cache_path=gpqa_family_cache_path,
    )
    current_runtime_settings = _debate_runtime_settings(
        engine=engine,
        judge_engine=effective_judge_engine,
        judge_block_size=judge_block_size,
        judge_sampling_kwargs=judge_sampling_kwargs,
        judge_strict_final_only=judge_strict_final_only,
        judge_recovery_parse_enabled=judge_recovery_parse_enabled,
        judge_trace_mode=judge_trace_mode,
        public_rationale_max_tokens=public_rationale_max_tokens,
    )
    parsed_items: list[tuple[SubsetItem, str, Any, dict[str, Any]]] = []
    for item in items:
        question, gt_answer, raw_task = _parse_question_answer(dataset, item.raw_task)
        parsed_items.append((item, question, gt_answer, raw_task))
    expected_persona_artifacts = None
    if persona_artifacts_by_item is not None:
        expected_persona_artifacts = [
            None if not bool(use_personas) else (
                persona_artifacts_by_item[item.item_uid].to_dict()
                if item.item_uid in persona_artifacts_by_item
                else None
            )
            for item, _question, _gt_answer, _raw_task in parsed_items
        ]

    resume_entry = None if stage_state_file is None or not stage_state_file.exists() else load_latest_stage_entry_of_type(stage_state_file, "debate")
    if resume_entry is not None and expected_persona_artifacts is not None:
        saved_persona_artifacts = (resume_entry.debate_data or {}).get("persona_artifacts")
        if list(saved_persona_artifacts or []) != expected_persona_artifacts:
            resume_entry = None
    resume_state: dict[str, Any] = {}
    if resume_entry is not None:
        saved_items = [subset_item_resume_signature(row) for row in resume_entry.items]
        current_items = [
            subset_item_resume_signature(item)
            for item, _question, _gt_answer, _raw_task in parsed_items
        ]
        if resume_entry.dataset != str(dataset):
            raise ValueError(
                f"Debate state dataset mismatch: {resume_entry.dataset} != {dataset}"
            )
        if saved_items != current_items:
            raise ValueError("Debate state items do not match the requested subset")
        debate_meta = dict(resume_entry.debate_data or {})
        resume_n_agents = debate_meta.get("n_agents")
        resume_n_rounds = debate_meta.get("n_rounds")
        if resume_n_agents is not None and int(resume_n_agents) != int(n_agents):
            raise ValueError(f"Debate state n_agents mismatch: {resume_n_agents} != {n_agents}")
        if resume_n_rounds is not None and int(resume_n_rounds) != int(n_rounds):
            raise ValueError(f"Debate state n_rounds mismatch: {resume_n_rounds} != {n_rounds}")
        resume_judge_rounds = debate_meta.get("judge_rounds")
        if not isinstance(resume_judge_rounds, list):
            raise ValueError("Debate state missing judge_rounds")
        if [int(round_num) for round_num in resume_judge_rounds] != judge_rounds:
            raise ValueError(
                f"Debate state judge_rounds mismatch: {resume_judge_rounds} != {judge_rounds}"
            )
        resume_state = debate_meta
        _validate_resume_persona_settings(
            resume_state=resume_state,
            current_settings=current_persona_settings,
        )
        _validate_resume_runtime_settings(
            resume_state=resume_state,
            current_settings=current_runtime_settings,
        )

    if not resume_state:
        persona_artifacts = []
        persona_artifact_paths = []
        runtime_judge_cards = []
        runtime_judge_bank_meta = []
        for item, question, _gt_answer, raw_task in parsed_items:
            runtime_judge_card = None
            runtime_judge_meta = None
            if use_personas:
                if artifacts_dir is None:
                    raise ValueError("artifacts_dir is required when use_personas=True")
                artifact = None if persona_artifacts_by_item is None else persona_artifacts_by_item.get(item.item_uid)
                artifact_path = None
                if artifact is None:
                    artifact, artifact_path = _resolve_persona_artifact(
                        dataset=dataset,
                        item=item,
                        question=question,
                        raw_task=raw_task,
                        artifacts_dir=artifacts_dir,
                        n_personas=n_agents,
                        persona_seed=persona_seed,
                        axis_mode=persona_axis_mode,
                        fixed_axis_count=persona_fixed_axis_count,
                        task_axis_count=persona_task_axis_count,
                        sampling_method=persona_sampling_method,
                        judge_persona_mode=persona_judge_mode,
                        backend=persona_backend,
                        generator_model=generator_model,
                        judge_generator_model=judge_generator_model,
                        generator_engine=persona_generator_engine,
                        judge_engine=persona_judge_engine or judge_engine or engine,
                        axes_file=persona_axes_file,
                        judge_bank_dir=judge_bank_dir,
                        judge_bank_refresh=judge_bank_refresh,
                        gpqa_family_cache_path=gpqa_family_cache_path,
                        save_artifact=persona_save_artifacts,
                        replay=persona_replay,
                    )
                elif len(artifact.cards) != int(n_agents):
                    raise ValueError(
                        f"Precomputed persona artifact for {item.item_uid!r} has {len(artifact.cards)} cards; expected {n_agents}"
                    )
                persona_artifacts.append(artifact)
                persona_artifact_paths.append(artifact_path)
                runtime_judge_card = artifact.judge_card
                validator_meta = artifact.validator_metadata.get("judge_bank")
                if isinstance(validator_meta, dict):
                    runtime_judge_meta = validator_meta
            else:
                persona_artifacts.append(None)
                persona_artifact_paths.append(None)
            if runtime_judge_persona_enabled and (runtime_judge_card is None or (persona_judge_mode == "benchmark_family_bank" and runtime_judge_meta is None)):
                runtime_judge_card, runtime_judge_meta = _resolve_runtime_judge_card(
                    dataset=dataset,
                    item=item,
                    question=question,
                    raw_task=raw_task,
                    persona_artifacts_dir=artifacts_dir or Path.cwd() / "persona_artifacts",
                    judge_bank_dir=judge_bank_dir,
                    judge_bank_refresh=judge_bank_refresh,
                    gpqa_family_cache_path=gpqa_family_cache_path,
                    judge_persona_mode=persona_judge_mode,
                    persona_backend=persona_backend,
                    judge_generator_model=judge_generator_model,
                    judge_engine=persona_judge_engine or judge_engine or engine,
                )
            runtime_judge_cards.append(runtime_judge_card)
            runtime_judge_bank_meta.append(runtime_judge_meta)
    else:
        persona_artifacts = _require_resume_list_length(
            [
                PersonaArtifact.from_dict(data) if data is not None else None
                for data in resume_state.get("persona_artifacts") or []
            ],
            expected=len(parsed_items),
            field_name="persona_artifacts",
        )
        persona_artifact_paths = _require_resume_list_length(
            [
                None if path is None else Path(str(path))
                for path in resume_state.get("persona_artifact_paths") or []
            ],
            expected=len(parsed_items),
            field_name="persona_artifact_paths",
        )
        runtime_judge_cards = _require_resume_list_length(
            [
                JudgeCard.from_dict(data)
                if data is not None else None
                for data in resume_state.get("runtime_judge_cards") or []
            ],
            expected=len(parsed_items),
            field_name="runtime_judge_cards",
        )
        runtime_judge_bank_meta = _require_resume_list_length(
            [
                dict(meta) if isinstance(meta, dict) else None
                for meta in resume_state.get("runtime_judge_bank_meta") or []
            ],
            expected=len(parsed_items),
            field_name="runtime_judge_bank_meta",
        )

    all_contexts = _build_initial_agent_contexts(
        dataset=dataset,
        parsed_items=parsed_items,
        engine=engine,
        persona_artifacts=persona_artifacts,
        n_agents=n_agents,
    )
    agent_round_outputs_by_q: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(n_agents)]
        for _ in parsed_items
    ]

    judge_engine = effective_judge_engine
    debater_token_counter = None
    debater_count_fn = getattr(engine, "count_prompt_tokens", None)
    base_count_fn = getattr(InferenceEngine, "count_prompt_tokens", None)
    if not callable(debater_count_fn) or getattr(debater_count_fn, "__func__", None) is base_count_fn:
        debater_token_counter = PromptTokenCounter(getattr(engine, "model_name", ""))
    judge_token_counter = None
    judge_count_fn = getattr(judge_engine, "count_prompt_tokens", None)
    if not callable(judge_count_fn) or getattr(judge_count_fn, "__func__", None) is base_count_fn:
        judge_token_counter = PromptTokenCounter(getattr(judge_engine, "model_name", engine.model_name))
    prev_judge_by_q = (
        _require_resume_list_length(
            [
                _deserialise_prev_judge(data)
                for data in resume_state.get("prev_judge_by_q") or []
            ],
            expected=len(parsed_items),
            field_name="prev_judge_by_q",
        )
        if resume_state
        else [None for _ in parsed_items]
    )
    results_by_round: dict[int, list[dict[str, Any]]] = {
        r: list(rows)
        for r, rows in (
            {
                int(k): list(v)
                for k, v in dict(resume_state.get("results_by_round") or {}).items()
            }.items()
        )
        if r in judge_rounds
    }
    for round_num in judge_rounds:
        results_by_round.setdefault(round_num, [])
    round1_majority_by_q = (
        _require_resume_list_length(
            list(resume_state.get("round1_majority_by_q") or []),
            expected=len(parsed_items),
            field_name="round1_majority_by_q",
        )
        if resume_state
        else [None for _ in parsed_items]
    )

    if resume_state:
        restored_contexts = resume_state.get("all_contexts")
        restored_outputs = resume_state.get("agent_round_outputs_by_q")
        if not isinstance(restored_outputs, list) or len(restored_outputs) != len(parsed_items):
            raise ValueError("Debate state agent_round_outputs_by_q mismatch")
        agent_round_outputs_by_q = [
            [
                [dict(round_output) for round_output in agent_outputs]
                for agent_outputs in item_outputs
            ]
            for item_outputs in restored_outputs
        ]
        if not isinstance(restored_contexts, list) or len(restored_contexts) != len(parsed_items):
            raise ValueError("Debate state all_contexts mismatch")
        all_contexts = [
            [
                [dict(message) for message in agent_context]
                for agent_context in item_contexts
            ]
            for item_contexts in restored_contexts
        ]

    start_round_idx = 0
    pending_judge_round: int | None = None
    if resume_entry is not None:
        parsed_stage = _parse_debate_stage_name(resume_entry.completed_stage)
        if parsed_stage is not None:
            completed_round, completed_judge = parsed_stage
            if completed_judge:
                start_round_idx = completed_round + 1
            elif completed_round in judge_rounds:
                start_round_idx = completed_round
                pending_judge_round = completed_round
            else:
                start_round_idx = completed_round + 1

    def _append_state(completed_stage: str) -> None:
        if stage_state_file is None:
            return
        _debate_append_state(
            stage_state_file=stage_state_file,
            completed_stage=completed_stage,
            dataset=str(dataset),
            items=items,
            all_contexts=all_contexts,
            agent_round_outputs_by_q=agent_round_outputs_by_q,
            results_by_round=results_by_round,
            prev_judge_by_q=prev_judge_by_q,
            round1_majority_by_q=round1_majority_by_q,
            persona_artifacts=persona_artifacts,
            persona_artifact_paths=persona_artifact_paths,
            runtime_judge_cards=runtime_judge_cards,
            runtime_judge_bank_meta=runtime_judge_bank_meta,
            persona_settings=current_persona_settings,
            runtime_settings=current_runtime_settings,
            n_agents=n_agents,
            n_rounds=n_rounds,
            judge_rounds=judge_rounds,
        )

    for round_idx in tqdm(
        range(start_round_idx, total_answer_rounds),
        desc="answer rounds",
        total=max(0, total_answer_rounds - start_round_idx),
        file=progress_file,
    ):
        skip_debater = pending_judge_round == round_idx
        if round_idx > 0 and not skip_debater:
            for q_idx in range(len(parsed_items)):
                agent_contexts = all_contexts[q_idx]
                previous_round_outputs = [agent_round_outputs_by_q[q_idx][agent_idx][-1] for agent_idx in range(n_agents)]
                for agent_idx, agent_ctx in enumerate(agent_contexts):
                    agent_ctx.append(
                        _build_agent_debate_message(
                            dataset=dataset,
                            previous_round_outputs=previous_round_outputs,
                            agent_idx=agent_idx,
                            persona_artifact=persona_artifacts[q_idx],
                        )
                    )

        current_round_num = round_idx + 1
        current_debate_round = max(0, current_round_num - 1)
        if not skip_debater:
            flat_contexts = [
                all_contexts[q_idx][agent_idx]
                for q_idx in range(len(parsed_items))
                for agent_idx in range(n_agents)
            ]
            flat_request_contexts = flat_contexts
            if round_idx > 0 and _provider_is_gemini(_engine_backend_name(engine)):
                flat_request_contexts = []
                for q_idx in range(len(parsed_items)):
                    artifact = persona_artifacts[q_idx]
                    for agent_idx in range(n_agents):
                        persona_id = None
                        if artifact is not None and agent_idx < len(artifact.cards):
                            persona_id = artifact.cards[agent_idx].persona_id
                        flat_request_contexts.append(
                            _build_gemini_debate_cache_context(
                                messages=all_contexts[q_idx][agent_idx],
                                dataset=dataset,
                                item_uid=parsed_items[q_idx][0].item_uid,
                                agent_idx=agent_idx,
                                round_idx=round_idx,
                                persona_id=persona_id,
                            )
                        )

            round_pbar = tqdm(
                total=len(flat_request_contexts),
                desc=f"round {round_idx + 1}",
                unit="call",
                leave=False,
                file=progress_file,
            )
            debater_sampling_kwargs: dict[str, Any] | None = None
            if _judge_trace_mode_enabled(judge_trace_mode) == "visible_plus_thought_summary" and _provider_is_gemini(_engine_backend_name(engine)):
                debater_sampling_kwargs = {"include_thought_summaries": True}
            if debater_sampling_kwargs is None:
                debater_sampling_kwargs = {}
            debater_sampling_kwargs.setdefault("max_tokens", DEBATE_DEFAULT_MAX_OUTPUT_TOKENS)
            try:
                flat_results = ensure_inference_results(
                    engine,
                    flat_request_contexts,
                    batch_size=batch_size,
                    sampling_kwargs=debater_sampling_kwargs,
                    progress_callback=round_pbar.update,
                    model_role="debater",
                )
            finally:
                round_pbar.close()
            flat_completions = [result.text for result in flat_results]

            for q_idx in range(len(parsed_items)):
                for agent_idx in range(n_agents):
                    flat_idx = q_idx * n_agents + agent_idx
                    raw_completion = str(flat_completions[flat_idx])
                    all_contexts[q_idx][agent_idx].append({"role": "assistant", "content": raw_completion})
                    round_output = _build_round_output(
                        dataset=dataset,
                        raw_response=raw_completion,
                        raw_task=parsed_items[q_idx][3],
                        gt_answer=parsed_items[q_idx][2],
                        public_rationale_max_tokens=public_rationale_max_tokens,
                        inference_result=flat_results[flat_idx],
                        request_messages=flat_request_contexts[flat_idx],
                        request_engine=engine,
                        prompt_token_counter=debater_token_counter,
                    )
                    agent_round_outputs_by_q[q_idx][agent_idx].append(round_output)

            debater_step_name = f"round_{current_debate_round}"
            _append_state(debater_step_name)
            if debate_stop_after == debater_step_name:
                is_final_unjudged_stage = (
                    round_idx == total_answer_rounds - 1
                    and current_debate_round not in judge_rounds
                )
                if not is_final_unjudged_stage:
                    raise _DebateStopped(results_by_round)

        if round_idx == 0:
            for q_idx, (_item, _question, gt_answer, raw_task) in enumerate(parsed_items):
                r1_answers = [
                    agent_round_outputs_by_q[q_idx][agent_idx][0].get("final_answer")
                    for agent_idx in range(n_agents)
                ]
                round1_majority_by_q[q_idx] = _vote_result_payload(
                    answers=r1_answers,
                    dataset=dataset,
                    gt_answer=gt_answer,
                    raw_task=raw_task,
                    result_kind="debate_round1_majority",
                    result_origin="shared_debate_round1",
                )

        if current_debate_round not in judge_rounds:
            continue

        block_end = current_round_num
        used_start_rounds: list[int] = [1 for _ in parsed_items]
        used_prev_texts: list[str | None] = [None for _ in parsed_items]

        ctx_len = int(getattr(judge_engine, "context_len_tokens", 0) or 0) or (
            infer_native_context_len(getattr(judge_engine, "model_name", engine.model_name)) or 32768
        )

        if judge_sampling_kwargs and judge_sampling_kwargs.get("max_tokens") is not None:
            judge_max_new_tokens = int(judge_sampling_kwargs["max_tokens"])
        else:
            from ..engines import get_sampling_config

            sampling_cfg = get_sampling_config()
            judge_max_new_tokens = sampling_cfg.max_tokens or _default_judge_max_tokens(judge_engine)

        judge_request_sampling_kwargs = _normalize_sampling_kwargs_for_engine(
            judge_engine,
            judge_sampling_kwargs,
        ) or {}
        judge_request_sampling_kwargs.setdefault("max_tokens", int(judge_max_new_tokens))

        def _build_contexts_for_round() -> list[list[dict[str, Any]]]:
            judge_contexts: list[list[dict[str, Any]]] = []
            for q_idx, (_item, question, _gt_answer, raw_task) in enumerate(parsed_items):
                judge_system_prompt = (
                    runtime_judge_cards[q_idx].system_prompt
                    if runtime_judge_cards[q_idx] is not None
                    else None
                )
                block_start = 1
                prev_text = None
                ctx_msgs = _build_judge_context(
                    dataset=dataset,
                    question=question,
                    raw_task=raw_task,
                    responses=_render_agent_round_outputs_for_judge(
                        agent_round_outputs=agent_round_outputs_by_q[q_idx],
                        start_round=1,
                        end_round=block_end,
                        judge_trace_mode=judge_trace_mode,
                    ),
                    previous_judge=None,
                    judge_system_prompt=judge_system_prompt,
                )
                budget = max(1, int(ctx_len) - max(0, int(judge_max_new_tokens)) - 256)
                prompt_tokens = _count_prompt_tokens(
                    engine=judge_engine,
                    counter=judge_token_counter,
                    messages=ctx_msgs,
                )
                if prompt_tokens is not None and prompt_tokens > budget:
                    block_start, prev_text, ctx_msgs = _build_judge_round_context(
                        dataset=dataset,
                        question=question,
                        raw_task=raw_task,
                        agent_round_outputs=agent_round_outputs_by_q[q_idx],
                        block_end=block_end,
                        judge_block_size=judge_block_size,
                        prev_judge=prev_judge_by_q[q_idx],
                        judge_system_prompt=judge_system_prompt,
                        judge_trace_mode=judge_trace_mode,
                        judge_engine=judge_engine,
                        judge_token_counter=judge_token_counter,
                        context_len_tokens=ctx_len,
                        judge_max_new_tokens=judge_max_new_tokens,
                    )
                used_start_rounds[q_idx] = block_start
                used_prev_texts[q_idx] = prev_text
                judge_contexts.append(ctx_msgs)
            return judge_contexts

        judge_contexts = _build_contexts_for_round()
        judge_request_contexts = judge_contexts
        if _provider_is_gemini(_engine_backend_name(judge_engine)):
            judge_request_contexts = [
                _build_gemini_judge_cache_context(
                    messages=judge_contexts[q_idx],
                    dataset=dataset,
                    item_uid=parsed_items[q_idx][0].item_uid,
                    round_idx=round_idx,
                    phase="raw",
                )
                for q_idx in range(len(parsed_items))
            ]
        judge_raw_results = ensure_inference_results(
            judge_engine,
            judge_request_contexts,
            batch_size=batch_size,
            sampling_kwargs=judge_request_sampling_kwargs,
            model_role="judge",
        )
        judge_raw_outputs = [result.text for result in judge_raw_results]

        judged_answers: list[str | None] = [None for _ in parsed_items]
        judge_retry_raw_outputs: list[str | None] = [None for _ in parsed_items]
        judge_retry_results: list[Any | None] = [None for _ in parsed_items]
        judge_retry_contexts: list[list[dict[str, Any]] | None] = [None for _ in parsed_items]
        judge_parse_failed: list[bool] = [False for _ in parsed_items]
        judge_used_fallback: list[bool] = [False for _ in parsed_items]
        judge_parse_mode: list[str] = ["none" for _ in parsed_items]
        judge_parse_source: list[str] = ["none" for _ in parsed_items]
        judge_retry_reason: list[str | None] = [None for _ in parsed_items]
        judge_finish_state: list[str] = ["raw_pending" for _ in parsed_items]
        judge_raw_had_strict_final: list[bool] = [False for _ in parsed_items]
        judge_retry_had_strict_final: list[bool] = [False for _ in parsed_items]
        allow_recovery_on_retry = bool(judge_recovery_parse_enabled)

        needs_retry_idxs: list[int] = []
        for q_idx, (raw_out, (_item, _question, _gt_answer, raw_task)) in enumerate(zip(judge_raw_outputs, parsed_items)):
            raw_strict_probe = _parse_judge_output(
                dataset=dataset,
                text=raw_out,
                raw_task=raw_task,
                source_prefix="raw",
                strict_enabled=True,
                recovery_enabled=False,
            )
            judge_raw_had_strict_final[q_idx] = bool(raw_strict_probe.strict_success)

            parsed = _parse_judge_output(
                dataset=dataset,
                text=raw_out,
                raw_task=raw_task,
                source_prefix="raw",
                strict_enabled=True,
                recovery_enabled=bool(judge_recovery_parse_enabled),
            )
            if parsed.answer is None:
                needs_retry_idxs.append(q_idx)
                judge_retry_reason[q_idx] = "parse_none"
                judge_finish_state[q_idx] = "retry_needed"
            else:
                judged_answers[q_idx] = parsed.answer
                judge_parse_mode[q_idx] = parsed.mode
                judge_parse_source[q_idx] = parsed.source
                judge_used_fallback[q_idx] = parsed.mode == "recover"
                judge_finish_state[q_idx] = "raw_parsed"

        if needs_retry_idxs:
            retry_sampling = dict(judge_request_sampling_kwargs)
            if not _provider_is_gemini(_engine_backend_name(judge_engine)):
                retry_sampling["temperature"] = 0.0
                retry_sampling["top_p"] = 1.0
            retry_sampling["max_tokens"] = int(judge_max_new_tokens)

            retry_contexts: list[list[dict[str, str]]] = []
            for q_idx in needs_retry_idxs:
                retry_ctx = list(judge_contexts[q_idx]) + [
                    {"role": "assistant", "content": str(judge_raw_outputs[q_idx])},
                    {"role": "user", "content": JUDGE_RETRY_NUDGE},
                ]
                retry_contexts.append(retry_ctx)
                judge_retry_contexts[q_idx] = retry_ctx
            retry_request_contexts = retry_contexts
            if _provider_is_gemini(_engine_backend_name(judge_engine)):
                retry_request_contexts = [
                    _build_gemini_judge_cache_context(
                        messages=retry_contexts[i],
                        dataset=dataset,
                        item_uid=parsed_items[q_idx][0].item_uid,
                        round_idx=round_idx,
                        phase="retry",
                    )
                    for i, q_idx in enumerate(needs_retry_idxs)
                ]

            retry_results_batch = ensure_inference_results(
                judge_engine,
                retry_request_contexts,
                batch_size=batch_size,
                sampling_kwargs=retry_sampling,
                model_role="judge",
            )
            retry_outputs = [result.text for result in retry_results_batch]

            for i, q_idx in enumerate(needs_retry_idxs):
                _item, _question, _gt_answer, raw_task = parsed_items[q_idx]
                retry_out = retry_outputs[i]
                judge_retry_raw_outputs[q_idx] = str(retry_out)
                if i < len(retry_results_batch):
                    judge_retry_results[q_idx] = retry_results_batch[i]

                retry_strict_probe = _parse_judge_output(
                    dataset=dataset,
                    text=retry_out,
                    raw_task=raw_task,
                    source_prefix="retry",
                    strict_enabled=True,
                    recovery_enabled=False,
                )
                judge_retry_had_strict_final[q_idx] = bool(retry_strict_probe.strict_success)
                parsed_retry = _parse_judge_output(
                    dataset=dataset,
                    text=retry_out,
                    raw_task=raw_task,
                    source_prefix="retry",
                    strict_enabled=True,
                    recovery_enabled=allow_recovery_on_retry,
                )
                judged = parsed_retry.answer
                if judged is not None:
                    judge_parse_mode[q_idx] = parsed_retry.mode
                    judge_parse_source[q_idx] = parsed_retry.source
                    judge_used_fallback[q_idx] = parsed_retry.mode == "recover"
                    judge_finish_state[q_idx] = "retry_parsed"
                else:
                    judge_finish_state[q_idx] = "retry_unparsed"

                judged_answers[q_idx] = judged

        for q_idx, (item, _question, _gt_answer, _raw_task) in enumerate(parsed_items):
            if judged_answers[q_idx] is None:
                judge_parse_failed[q_idx] = True
                judge_finish_state[q_idx] = "failed_unparsed"
                print(
                    f"[warn] Judge output unparsable after retry: "
                    f"subset_id={item.subset_id} orig_id={item.orig_id} round={current_round_num}",
                    file=sys.stderr,
                )

        for q_idx, (raw_out, judged) in enumerate(zip(judge_raw_outputs, judged_answers)):
            if judged is None or judge_parse_mode[q_idx] != "strict":
                continue
            cache_raw_output = raw_out
            if (
                str(judge_parse_source[q_idx]).startswith("retry_")
                and judge_retry_raw_outputs[q_idx] is not None
            ):
                cache_raw_output = judge_retry_raw_outputs[q_idx]
            prev_judge_by_q[q_idx] = PrevJudgeInfo(
                start_round=int(used_start_rounds[q_idx]),
                end_round=block_end,
                parsed_answer=str(judged),
                raw_output=str(cache_raw_output),
            )

        for q_idx, (item, question, gt_answer, raw_task) in enumerate(parsed_items):
            agent_contexts = all_contexts[q_idx]
            judged = judged_answers[q_idx]
            agent_round_outputs = [
                [dict(round_output) for round_output in agent_round_outputs_by_q[q_idx][agent_idx][:current_round_num]]
                for agent_idx in range(n_agents)
            ]
            agent_round_parsed_answers = [
                [round_output.get("final_answer") for round_output in agent_outputs]
                for agent_outputs in agent_round_outputs
            ]
            round1_majority_result = round1_majority_by_q[q_idx] or _vote_result_payload(
                answers=[
                    agent_outputs[0].get("final_answer") if agent_outputs else None
                    for agent_outputs in agent_round_outputs
                ],
                dataset=dataset,
                gt_answer=gt_answer,
                raw_task=raw_task,
                result_kind="debate_round1_majority",
                result_origin="shared_debate_round1",
            )
            round1_majority_answer = round1_majority_result["majority_answer"]
            round1_majority_correct = round1_majority_result["majority_correct"]
            final_round_answers = [answers[-1] if answers else None for answers in agent_round_parsed_answers]
            final_round_majority_result = _vote_result_payload(
                answers=final_round_answers,
                dataset=dataset,
                gt_answer=gt_answer,
                raw_task=raw_task,
                result_kind="debate_final_round_majority",
                result_origin="debate_final_round",
            )
            final_majority_answer = final_round_majority_result["majority_answer"]
            final_majority_correct = final_round_majority_result["majority_correct"]
            convergence_per_round = _compute_round_convergence(
                agent_round_outputs,
                n_rounds=current_round_num,
            )
            answer_changes_per_agent = _compute_answer_changes(agent_round_outputs)
            persona_fidelity_metrics = _compute_persona_fidelity_metrics(
                agent_round_outputs,
                answer_changes=answer_changes_per_agent,
                convergence=convergence_per_round,
            )

            final_judge_answer = judged
            judge_extraction = _extract_output_details(
                dataset=dataset,
                raw_response=str(judge_raw_outputs[q_idx]),
                raw_task=raw_task,
                gt_answer=gt_answer,
                parse_mode=judge_parse_mode[q_idx] if judge_parse_mode[q_idx] in {"default", "strict", "recover"} else "default",
            )
            judge_retry_extraction = (
                _extract_output_details(
                    dataset=dataset,
                    raw_response=str(judge_retry_raw_outputs[q_idx]),
                    raw_task=raw_task,
                    gt_answer=gt_answer,
                    parse_mode=judge_parse_mode[q_idx] if judge_parse_mode[q_idx] in {"default", "strict", "recover"} else "recover",
                )
                if judge_retry_raw_outputs[q_idx] is not None
                else None
            )
            accepted_judge_extraction = judge_extraction
            accepted_judge_scoring = judge_extraction["scoring_result"]
            accepted_judge_trace_source = "raw"
            if str(judge_parse_source[q_idx]).startswith("retry_") and judge_retry_extraction is not None:
                accepted_judge_extraction = judge_retry_extraction
                accepted_judge_scoring = judge_retry_extraction["scoring_result"]
                accepted_judge_trace_source = "retry"
            final_judge_correct = _check_answer_correctness(dataset, final_judge_answer, gt_answer, raw_task)
            judge_raw_call_meta = _inference_result_meta(
                judge_raw_results[q_idx],
                request_messages=judge_contexts[q_idx],
                engine=judge_engine,
                prompt_token_counter=judge_token_counter,
            )
            judge_retry_call_meta = _inference_result_meta(
                judge_retry_results[q_idx],
                request_messages=judge_retry_contexts[q_idx],
                engine=judge_engine,
                prompt_token_counter=judge_token_counter,
            )
            debater_round_token_usage = _compute_round_token_usage(agent_round_outputs)
            judge_token_usage = {
                    "round": current_debate_round,
                "raw_call": None if judge_raw_call_meta is None else judge_raw_call_meta["token_counts"],
                "retry_call": None if judge_retry_call_meta is None else judge_retry_call_meta["token_counts"],
                "aggregate": _merge_token_counts(
                    [
                        None if judge_raw_call_meta is None else judge_raw_call_meta["token_counts"],
                        None if judge_retry_call_meta is None else judge_retry_call_meta["token_counts"],
                    ]
                ),
            }
            runtime_judge_meta = runtime_judge_bank_meta[q_idx] or {}
            judge_family_assignment = (
                runtime_judge_meta.get("judge_family_assignment")
                if isinstance(runtime_judge_meta, dict)
                else None
            )
            judge_bank_meta = (
                runtime_judge_meta
                if isinstance(runtime_judge_meta, dict) and runtime_judge_meta.get("judge_bank_path")
                else None
            )
            judge_trace = {
                "judge_backend": _engine_backend_name(judge_engine),
                "judge_model": getattr(judge_engine, "model_name", engine.model_name),
                "judge_trace_mode": _judge_trace_mode_enabled(judge_trace_mode),
                "judge_context_start_round": used_start_rounds[q_idx],
                "judge_context_end_round": block_end,
                "judge_context_is_full_transcript": used_start_rounds[q_idx] == 1 and used_prev_texts[q_idx] is None,
                "judge_context": judge_contexts[q_idx],
                "judge_raw_response": judge_raw_outputs[q_idx],
                "judge_raw_call_metadata": judge_raw_call_meta,
                "judge_retry_raw_response": judge_retry_raw_outputs[q_idx],
                "judge_retry_call_metadata": judge_retry_call_meta,
                "judge_parsed_answer": judged,
                "judge_extractor_trace": accepted_judge_extraction["extractor_trace"],
                "judge_extractor_trace_source": accepted_judge_trace_source,
                "judge_raw_extractor_trace": judge_extraction["extractor_trace"],
                "judge_retry_extractor_trace": None if judge_retry_extraction is None else judge_retry_extraction["extractor_trace"],
                "judge_scoring_result": accepted_judge_scoring,
                "judge_raw_scoring_result": judge_extraction["scoring_result"],
                "judge_retry_scoring_result": None if judge_retry_extraction is None else judge_retry_extraction["scoring_result"],
                "judge_parse_failed": bool(judge_parse_failed[q_idx]),
                "judge_used_fallback": bool(judge_used_fallback[q_idx]),
                "judge_parse_mode": judge_parse_mode[q_idx],
                "judge_parse_source": judge_parse_source[q_idx],
                "judge_retry_reason": judge_retry_reason[q_idx],
                "judge_finish_state": judge_finish_state[q_idx],
                "judge_raw_had_strict_final": bool(judge_raw_had_strict_final[q_idx]),
                "judge_retry_had_strict_final": bool(judge_retry_had_strict_final[q_idx]),
                "judge_card": None if runtime_judge_cards[q_idx] is None else asdict(runtime_judge_cards[q_idx]),
                "judge_family_assignment": judge_family_assignment,
                "judge_bank": judge_bank_meta,
                "judge_previous_summary": used_prev_texts[q_idx],
                "judge_correct": final_judge_correct,
                "judge_token_usage": judge_token_usage,
            }
            judge_summary = _judge_summary_from_card(runtime_judge_cards[q_idx])
            persona_summary_rows = _persona_summaries(persona_artifacts[q_idx]) or []
            revision_profiles = persona_fidelity_metrics.get("revision_rate_by_persona") or []
            if isinstance(revision_profiles, list):
                for agent_idx, profile in enumerate(revision_profiles):
                    if not isinstance(profile, dict):
                        continue
                    summary = persona_summary_rows[agent_idx] if agent_idx < len(persona_summary_rows) else {}
                    if isinstance(summary, dict):
                        profile.setdefault("persona_id", summary.get("persona_id"))
                        profile.setdefault("title", summary.get("title"))
                        profile.setdefault("short_rule", summary.get("short_rule"))
            artifact_path = persona_artifact_paths[q_idx]
            persona_meta = _persona_runtime_meta(
                persona_artifacts[q_idx],
                artifact_path=artifact_path,
                allow_missing_artifact_path=bool(persona_replay or persona_save_artifacts),
                persona_sampling_method=persona_sampling_method,
                persona_backend=persona_backend,
                public_rationale_max_tokens=public_rationale_max_tokens,
            )
            judge_meta = {
                "judge_model": getattr(judge_engine, "model_name", engine.model_name),
                "judge_persona_mode": persona_judge_mode if runtime_judge_cards[q_idx] is not None else None,
                "judge_trace_mode": _judge_trace_mode_enabled(judge_trace_mode),
                "judge_summary": judge_summary,
                "judge_family_assignment": judge_family_assignment,
                "judge_bank": judge_bank_meta,
            }
            judge_result = {
                "result_kind": "debate_judge_selection",
                "result_origin": "debate_judge",
                "answer": final_judge_answer,
                "correct": final_judge_correct,
            }

            is_final_round = current_round_num == total_answer_rounds
            agent_responses = agent_contexts if is_final_round else [ctx[:] for ctx in agent_contexts]

            results_by_round[current_debate_round].append(
                {
                    "schema_version": "phase2.debate.v1",
                    "mode": "debate",
                    "row_origin": "debate_judge",
                    "debater_model": getattr(engine, "model_name", None),
                    "debater_backend": _engine_backend_name(engine),
                    "n_agents": n_agents,
                    "n_rounds": current_debate_round,
                    "total_answer_rounds": current_round_num,
                    "persona_meta": persona_meta,
                    "judge_meta": judge_meta,
                    "persona_ids": [card.persona_id for card in persona_artifacts[q_idx].cards] if persona_artifacts[q_idx] is not None else None,
                    "persona_summaries": persona_summary_rows,
                    "judge_summary": judge_summary,
                    "agent_responses": agent_responses,
                    "agent_round_outputs": agent_round_outputs,
                    "agent_round_parsed_answers": agent_round_parsed_answers,
                    "debater_round_token_usage": debater_round_token_usage,
                    "judge_round_token_usage": judge_token_usage,
                    "token_usage_summary": {
                        "debater": _merge_token_counts(debater_round_token_usage),
                        "judge": judge_token_usage["aggregate"],
                        "all": _merge_token_counts(
                            debater_round_token_usage + [judge_token_usage["aggregate"]]
                        ),
                    },
                    "round1_majority_result": round1_majority_result,
                    "round1_majority_origin": round1_majority_result["result_origin"],
                    "round1_vote_counts": round1_majority_result["vote_counts"],
                    "round1_strict_majority_answer": round1_majority_result["strict_majority_answer"],
                    "round1_plurality_answer": round1_majority_result["plurality_answer"],
                    "round1_majority_answer": round1_majority_answer,
                    "round1_majority_correct": round1_majority_correct,
                    "final_round_majority_result": final_round_majority_result,
                    "final_round_majority_answer": final_majority_answer,
                    "final_round_majority_correct": final_majority_correct,
                    "judge_result": judge_result,
                    "judge_final_answer": final_judge_answer,
                    "judge_final_correct": final_judge_correct,
                    "convergence_per_round": convergence_per_round,
                    "answer_changes_per_agent": answer_changes_per_agent,
                    "persona_fidelity_metrics": persona_fidelity_metrics,
                    "final_majority_answer": final_majority_answer,
                    "final_majority_correct": final_majority_correct,
                    "judge_trace": judge_trace,
                    "final_judge_answer": final_judge_answer,
                    "final_judge_correct": final_judge_correct,
                    "final_answer": final_judge_answer,
                    "final_correct": final_judge_correct,
                    "final_answer_source": "judge",
                }
            )
            results_by_round[current_debate_round][-1].update(
                _base_row_fields(
                    dataset=dataset,
                    item=item,
                    question=question,
                    gt_answer=gt_answer,
                    raw_task=raw_task,
                )
            )

        judge_step_name = f"round_{current_debate_round}_judge"
        _append_state(judge_step_name)
        if debate_stop_after == judge_step_name:
            is_final_judge_stage = round_idx == total_answer_rounds - 1
            if not is_final_judge_stage:
                raise _DebateStopped(results_by_round)

    return results_by_round


def _debate_append_state(
    *,
    stage_state_file: Path,
    completed_stage: str,
    dataset: str,
    items: list[SubsetItem],
    all_contexts: list[list[list[dict[str, Any]]]],
    agent_round_outputs_by_q: list[list[list[dict[str, Any]]]],
    results_by_round: dict[int, list[dict[str, Any]]],
    prev_judge_by_q: list[PrevJudgeInfo | None],
    round1_majority_by_q: list[dict[str, Any] | None],
    persona_artifacts: list[PersonaArtifact | None],
    persona_artifact_paths: list[Path | None],
    runtime_judge_cards: list[Any | None],
    runtime_judge_bank_meta: list[dict[str, Any] | None],
    persona_settings: dict[str, Any],
    runtime_settings: dict[str, Any],
    n_agents: int,
    n_rounds: int,
    judge_rounds: list[int],
) -> None:
    serialised_outputs = [
        [list(agent_outputs) for agent_outputs in item_outputs]
        for item_outputs in agent_round_outputs_by_q
    ]
    entry = make_stage_entry(
        stage_type="debate",
        completed_stage=completed_stage,
        dataset=dataset,
        items=[asdict(it) for it in items],
        debate_data={
            "all_contexts": all_contexts,
            "agent_round_outputs_by_q": serialised_outputs,
            "results_by_round": {str(k): v for k, v in results_by_round.items()},
            "prev_judge_by_q": [_serialise_prev_judge(prev) for prev in prev_judge_by_q],
            "round1_majority_by_q": round1_majority_by_q,
            "persona_artifacts": [
                None if artifact is None else artifact.to_dict()
                for artifact in persona_artifacts
            ],
            "persona_artifact_paths": [
                None if artifact_path is None else str(artifact_path)
                for artifact_path in persona_artifact_paths
            ],
            "runtime_judge_cards": [
                None if judge_card is None else asdict(judge_card)
                for judge_card in runtime_judge_cards
            ],
            "runtime_judge_bank_meta": runtime_judge_bank_meta,
            "persona_settings": dict(persona_settings),
            "runtime_settings": dict(runtime_settings),
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "judge_rounds": list(judge_rounds),
        },
    )
    append_stage_entry(stage_state_file, entry)


__all__ = ["run_debate", "_DebateStopped"]
