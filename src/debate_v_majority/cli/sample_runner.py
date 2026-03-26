from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from tqdm import tqdm

from .. import DatasetName, Mode
from ..engines import InferenceEngine, ensure_inference_results
from ..personas import PersonaArtifact
from ..shared import PromptTokenCounter
from .dataset_eval import (
    _build_initial_user_message,
    _check_answer_correctness,
    _parse_question_answer,
)
from .engine_runtime import _engine_backend_name, _inference_result_meta, _persona_summaries
from .persona_runtime import _resolve_persona_artifact
from .result_rows import _base_row_fields, _persona_runtime_meta, _sample_row_common, _vote_result_payload
from .response_parsing import _extract_output_details
from .subset import SubsetItem


SINGLE_DEFAULT_MAX_OUTPUT_TOKENS = 32768


def run_sampled(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    engine: InferenceEngine,
    n_samples: int,
    batch_size: int | None,
    mode_label: Mode,
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
    judge_bank_dir: Path | None = None,
    judge_bank_refresh: bool = False,
    gpqa_family_cache_path: Path | None = None,
    progress_file: TextIO = sys.stdout,
) -> list[dict[str, Any]]:
    if use_personas and artifacts_dir is None:
        raise ValueError("artifacts_dir is required when use_personas=True")

    effective_persona_generator_engine = persona_generator_engine or (engine if use_personas else None)
    effective_persona_judge_engine = persona_judge_engine or effective_persona_generator_engine

    model_name = getattr(engine, "model_name", None)
    model_backend = _engine_backend_name(engine)
    parsed_inputs: list[tuple[SubsetItem, str, Any, dict[str, Any], PersonaArtifact | None, Path | None]] = []
    contexts_flat: list[list[dict[str, Any]]] = []
    debater_token_counter = None
    debater_count_fn = getattr(engine, "count_prompt_tokens", None)
    base_count_fn = getattr(InferenceEngine, "count_prompt_tokens", None)
    if not callable(debater_count_fn) or getattr(debater_count_fn, "__func__", None) is base_count_fn:
        debater_token_counter = PromptTokenCounter(getattr(engine, "model_name", ""))

    for item in items:
        question, gt_answer, raw_task = _parse_question_answer(dataset, item.raw_task)
        artifact: PersonaArtifact | None = None
        artifact_path: Path | None = None
        if use_personas:
            artifact, artifact_path = _resolve_persona_artifact(
                dataset=dataset,
                item=item,
                question=question,
                raw_task=raw_task,
                artifacts_dir=cast(Path, artifacts_dir),
                n_personas=n_samples,
                persona_seed=persona_seed,
                axis_mode=persona_axis_mode,
                fixed_axis_count=persona_fixed_axis_count,
                task_axis_count=persona_task_axis_count,
                sampling_method=persona_sampling_method,
                judge_persona_mode=persona_judge_mode,
                backend=persona_backend,
                generator_model=generator_model,
                judge_generator_model=judge_generator_model,
                generator_engine=effective_persona_generator_engine,
                judge_engine=effective_persona_judge_engine,
                axes_file=persona_axes_file,
                judge_bank_dir=judge_bank_dir,
                judge_bank_refresh=judge_bank_refresh,
                gpqa_family_cache_path=gpqa_family_cache_path,
                save_artifact=persona_save_artifacts,
                replay=persona_replay,
            )
        parsed_inputs.append((item, question, gt_answer, raw_task, artifact, artifact_path))
        for sample_idx in range(n_samples):
            ctx: list[dict[str, Any]] = []
            if artifact is not None:
                ctx.append({"role": "system", "content": artifact.cards[sample_idx].initial_system_prompt})
            ctx.append(
                _build_initial_user_message(
                    dataset=dataset,
                    question=question,
                    raw_task=raw_task,
                    engine=engine,
                )
            )
            contexts_flat.append(ctx)

    pbar = tqdm(total=len(contexts_flat), desc=mode_label, unit="call", file=progress_file)
    sampling_kwargs: dict[str, Any] | None = None
    if mode_label == "single":
        sampling_kwargs = {"max_tokens": SINGLE_DEFAULT_MAX_OUTPUT_TOKENS}
    inference_results_flat = ensure_inference_results(
        engine,
        contexts_flat,
        batch_size=batch_size,
        sampling_kwargs=sampling_kwargs,
        progress_callback=pbar.update,
        model_role="debater",
    )
    pbar.close()
    completions_flat = [result.text for result in inference_results_flat]

    records: list[dict[str, Any]] = []
    for item_idx, (item, question, gt_answer, raw_task, artifact, artifact_path) in enumerate(parsed_inputs):
        start = item_idx * n_samples
        end = start + n_samples
        sample_completions = completions_flat[start:end]
        sample_inference_results = inference_results_flat[start:end]
        sample_contexts = contexts_flat[start:end]
        sample_call_metadata = [
            _inference_result_meta(
                result,
                request_messages=sample_contexts[result_idx],
                engine=engine,
                prompt_token_counter=debater_token_counter,
            )
            for result_idx, result in enumerate(sample_inference_results)
        ]
        sample_extractions = [
            _extract_output_details(
                dataset=dataset,
                raw_response=c,
                raw_task=raw_task,
                gt_answer=gt_answer,
                parse_mode="default",
            )
            for c in sample_completions
        ]
        sample_parsed = [extraction["final_answer"] for extraction in sample_extractions]
        vote = _vote_result_payload(
            answers=sample_parsed,
            dataset=dataset,
            gt_answer=gt_answer,
            raw_task=raw_task,
            result_kind="standalone_majority" if n_samples > 1 else "single_independent",
            result_origin="standalone_persona" if artifact is not None else "standalone_sampling",
        )
        final_answer = sample_parsed[0] if n_samples == 1 else vote["majority_answer"]
        final_correct = (
            _check_answer_correctness(dataset, final_answer, gt_answer, raw_task)
            if n_samples == 1
            else vote["majority_correct"]
        )
        record = _sample_row_common(
            mode_label=mode_label,
            model_name=model_name,
            model_backend=model_backend,
            n_samples=n_samples,
            sample_completions=sample_completions,
            sample_call_metadata=sample_call_metadata,
            sample_parsed=sample_parsed,
            sample_extractions=sample_extractions,
            final_answer=final_answer,
            final_correct=final_correct,
        )
        if mode_label == "majority":
            row = record
            row.update(
                {
                    "schema_version": "phase2.majority.v1" if use_personas else "phase2.majority_compat.v1",
                    "row_origin": "standalone_persona_majority" if use_personas else "standalone_sampling_majority",
                    "majority_origin": "standalone_persona" if use_personas else "standalone_sampling",
                    "majority_result": vote,
                    "vote_counts": vote["vote_counts"],
                    "strict_majority_answer": vote["strict_majority_answer"],
                    "plurality_answer": vote["plurality_answer"],
                    "final_majority_answer": vote["majority_answer"],
                    "final_majority_correct": vote["majority_correct"],
                }
            )
            row.update(
                _base_row_fields(
                    dataset=dataset,
                    item=item,
                    question=question,
                    gt_answer=gt_answer,
                    raw_task=raw_task,
                )
            )
            row["persona_meta"] = _persona_runtime_meta(
                artifact,
                artifact_path=artifact_path,
                allow_missing_artifact_path=bool(persona_replay or persona_save_artifacts),
                persona_sampling_method=persona_sampling_method,
                persona_backend=persona_backend,
            )
            if artifact is not None:
                row["sample_persona_ids"] = [card.persona_id for card in artifact.cards]
                row["sample_persona_summaries"] = _persona_summaries(artifact)
                row["persona_ids"] = row["sample_persona_ids"]
                row["persona_summaries"] = row["sample_persona_summaries"]
            records.append(row)
            continue

        row = {
            "schema_version": "phase2.single.v1",
            "row_origin": "single_independent",
        }
        row.update(record)
        row.update(
            _base_row_fields(
                dataset=dataset,
                item=item,
                question=question,
                gt_answer=gt_answer,
                raw_task=raw_task,
            )
        )
        records.append(row)
    return records


__all__ = ["run_sampled"]
