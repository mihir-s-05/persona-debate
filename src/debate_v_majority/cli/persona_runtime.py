from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO, cast

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from ..personas import (
    PersonaArtifact,
    PersonaGenerationConfig,
    artifact_path as persona_artifact_path,
    build_judge_card,
    build_persona_artifact,
    default_gpqa_family_cache_path,
    default_judge_bank_dir,
    ensure_judge_bank_card,
    legacy_artifact_path as legacy_persona_artifact_path,
    load_artifact as load_persona_artifact,
    resolve_judge_family_assignment,
    save_artifact as save_persona_artifact,
)

from .subset import SubsetItem


def _parse_question_answer(dataset: DatasetName, sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    adapter = resolve_dataset_adapter(dataset)
    return cast(tuple[str, str, dict[str, Any]], adapter.parse_question_answer(sample))


def _persona_generation_settings(
    *,
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    judge_persona_mode: str,
    backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    axes_file: Path | None,
) -> dict[str, Any]:
    return {
        "n_personas": int(n_personas),
        "persona_seed": int(persona_seed),
        "axis_mode": str(axis_mode),
        "fixed_axis_count": int(fixed_axis_count),
        "task_axis_count": int(task_axis_count),
        "sampling_method": str(sampling_method),
        "judge_persona_mode": str(judge_persona_mode),
        "backend": str(backend),
        "generator_model": generator_model,
        "judge_generator_model": judge_generator_model,
        "axes_file": None if axes_file is None else str(axes_file),
    }


def _replayed_persona_artifact_mismatch(
    *,
    artifact: PersonaArtifact,
    artifact_path: Path,
    dataset: DatasetName,
    item: SubsetItem,
    persona_seed: int,
    axis_mode: str,
    n_personas: int,
    generator_model: str | None,
    judge_generator_model: str | None,
    generation_settings: dict[str, Any],
) -> str | None:
    if artifact.dataset != dataset:
        return f"Artifact {artifact_path} dataset mismatch: {artifact.dataset} != {dataset}"
    if artifact.item_uid != item.item_uid:
        return f"Artifact {artifact_path} item_uid mismatch: {artifact.item_uid} != {item.item_uid}"
    if artifact.dataset_revision != item.dataset_revision:
        return f"Artifact {artifact_path} dataset_revision mismatch: {artifact.dataset_revision} != {item.dataset_revision}"
    if artifact.persona_seed != persona_seed:
        return f"Artifact {artifact_path} persona_seed mismatch: {artifact.persona_seed} != {persona_seed}"
    if axis_mode != "replay" and artifact.axes.mode != axis_mode:
        return f"Artifact {artifact_path} axis_mode mismatch: {artifact.axes.mode} != {axis_mode}"
    if len(artifact.cards) != n_personas:
        return f"Artifact {artifact_path} n_personas mismatch: {len(artifact.cards)} != {n_personas}"
    if (artifact.generator_model or None) != (generator_model or None):
        return f"Artifact {artifact_path} generator_model mismatch: {artifact.generator_model} != {generator_model}"
    if (artifact.judge_generator_model or None) != (judge_generator_model or None):
        return f"Artifact {artifact_path} judge_generator_model mismatch: {artifact.judge_generator_model} != {judge_generator_model}"
    if artifact.generation_settings:
        for key, expected in generation_settings.items():
            if axis_mode == "replay" and key == "axis_mode":
                continue
            actual = artifact.generation_settings.get(key)
            if actual != expected:
                return (
                    f"Artifact {artifact_path} generation setting mismatch for {key}: {actual!r} != {expected!r}"
                )
    return None


def _persona_artifact_row(artifact: PersonaArtifact, *, artifact_path_str: str | None) -> dict[str, Any]:
    data = artifact.to_dict()
    return {
        "schema_version": "phase0.v1",
        "mode": "personas",
        "dataset": artifact.dataset,
        "item_uid": artifact.item_uid,
        "dataset_revision": artifact.dataset_revision,
        "item_display_id": artifact.item_display_id,
        "artifact_version": artifact.artifact_version,
        "persona_seed": artifact.persona_seed,
        "generator_model": artifact.generator_model,
        "judge_generator_model": artifact.judge_generator_model,
        "axes": data["axes"],
        "sampled_points": data["sampled_points"],
        "descriptors": data["descriptors"],
        "cards": data["cards"],
        "judge_card": data["judge_card"],
        "prompt_versions": artifact.prompt_versions,
        "created_at": artifact.created_at,
        "validator_metadata": data["validator_metadata"],
        "artifact_path": artifact_path_str,
    }


def _print_persona_artifact_summary(
    artifact: PersonaArtifact,
    *,
    output_file: TextIO,
    dump_cards: bool,
    artifact_path_str: str | None,
) -> None:
    print(
        f"[persona] item_uid={artifact.item_uid} axes={len(artifact.axes.axes)} personas={len(artifact.cards)}",
        file=output_file,
    )
    if artifact_path_str:
        print(f"[persona] artifact={artifact_path_str}", file=output_file)
    for card in artifact.cards:
        print(f"[persona] {card.persona_id}: {card.title}", file=output_file)
        if dump_cards:
            print(card.system_prompt, file=output_file)
    if artifact.judge_card is not None:
        print(f"[persona] judge={artifact.judge_card.judge_family}", file=output_file)


def _resolve_persona_artifact(
    *,
    dataset: DatasetName,
    item: SubsetItem,
    question: str,
    raw_task: dict[str, Any],
    artifacts_dir: Path,
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    judge_persona_mode: str,
    backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    generator_engine: Any | None,
    judge_engine: Any | None,
    axes_file: Path | None,
    judge_bank_dir: Path | None = None,
    judge_bank_refresh: bool = False,
    gpqa_family_cache_path: Path | None = None,
    save_artifact: bool,
    replay: bool,
) -> tuple[PersonaArtifact, Path]:
    generation_settings = _persona_generation_settings(
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        judge_persona_mode=judge_persona_mode,
        backend=backend,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        axes_file=axes_file,
    )
    artifact_file = persona_artifact_path(
        artifacts_dir=artifacts_dir,
        dataset=dataset,
        item_uid=item.item_uid,
        dataset_revision=item.dataset_revision,
        generation_settings=generation_settings,
    )
    legacy_artifact_file = legacy_persona_artifact_path(
        artifacts_dir=artifacts_dir,
        dataset=dataset,
        item_uid=item.item_uid,
    )
    if replay:
        safe_name = item.item_uid.replace(":", "__")
        candidate_paths: list[Path] = []
        for candidate in [artifact_file, legacy_artifact_file, *sorted((artifacts_dir / dataset).glob(f"{safe_name}*.json"))]:
            if candidate.exists() and candidate not in candidate_paths:
                candidate_paths.append(candidate)
        if not candidate_paths:
            raise FileNotFoundError(f"Replay requested but no artifact exists at {artifact_file}")
        mismatches: list[str] = []
        for candidate_path in candidate_paths:
            artifact = load_persona_artifact(path=candidate_path)
            mismatch = _replayed_persona_artifact_mismatch(
                artifact=artifact,
                artifact_path=candidate_path,
                dataset=dataset,
                item=item,
                persona_seed=persona_seed,
                axis_mode=axis_mode,
                n_personas=n_personas,
                generator_model=generator_model,
                judge_generator_model=judge_generator_model,
                generation_settings=generation_settings,
            )
            if mismatch is not None:
                mismatches.append(mismatch)
                continue
            return artifact, candidate_path
        raise ValueError("\n".join(mismatches) if mismatches else f"No compatible replay artifact found for {item.item_uid}")

    config = PersonaGenerationConfig(
        dataset=dataset,
        question=question,
        raw_task=raw_task,
        item_uid=item.item_uid,
        item_display_id=item.item_display_id,
        dataset_revision=item.dataset_revision,
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=cast(Any, axis_mode),
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=cast(Any, sampling_method),
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        judge_persona_mode=cast(Any, judge_persona_mode),
        backend=cast(Any, backend),
        axes_file=axes_file,
    )
    judge_card, judge_bank_meta = _resolve_runtime_judge_card(
        dataset=dataset,
        item=item,
        question=question,
        raw_task=raw_task,
        persona_artifacts_dir=artifacts_dir,
        judge_bank_dir=judge_bank_dir,
        judge_bank_refresh=judge_bank_refresh,
        gpqa_family_cache_path=gpqa_family_cache_path,
        judge_persona_mode=judge_persona_mode,
        persona_backend=backend,
        judge_generator_model=judge_generator_model,
        judge_engine=judge_engine,
    )
    artifact = build_persona_artifact(
        config=config,
        judge_card=judge_card,
        generator_engine=generator_engine,
    )
    if judge_bank_meta is not None:
        artifact = replace(
            artifact,
            validator_metadata={**artifact.validator_metadata, "judge_bank": judge_bank_meta},
        )
    if save_artifact:
        artifact_file = save_persona_artifact(artifacts_dir=artifacts_dir, artifact=artifact)
    return artifact, artifact_file


def run_persona_generation(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    artifacts_dir: Path,
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    judge_persona_mode: str,
    backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    generator_engine: Any | None,
    judge_engine: Any | None,
    axes_file: Path | None,
    judge_bank_dir: Path | None = None,
    judge_bank_refresh: bool = False,
    gpqa_family_cache_path: Path | None = None,
    save_artifacts: bool,
    replay: bool,
    dump_cards: bool,
    summary_file: TextIO,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        question, _, raw_task = _parse_question_answer(dataset, item.raw_task)
        artifact, artifact_file = _resolve_persona_artifact(
            dataset=dataset,
            item=item,
            question=question,
            raw_task=raw_task,
            artifacts_dir=artifacts_dir,
            n_personas=n_personas,
            persona_seed=persona_seed,
            axis_mode=axis_mode,
            fixed_axis_count=fixed_axis_count,
            task_axis_count=task_axis_count,
            sampling_method=sampling_method,
            judge_persona_mode=judge_persona_mode,
            backend=backend,
            generator_model=generator_model,
            judge_generator_model=judge_generator_model,
            generator_engine=generator_engine,
            judge_engine=judge_engine,
            axes_file=axes_file,
            judge_bank_dir=judge_bank_dir,
            judge_bank_refresh=judge_bank_refresh,
            gpqa_family_cache_path=gpqa_family_cache_path,
            save_artifact=save_artifacts,
            replay=replay,
        )
        artifact_path_str = str(artifact_file) if artifact_file.exists() or save_artifacts else None
        _print_persona_artifact_summary(
            artifact,
            output_file=summary_file,
            dump_cards=dump_cards,
            artifact_path_str=artifact_path_str,
        )
        rows.append(_persona_artifact_row(artifact, artifact_path_str=artifact_path_str))
    return rows


def _resolve_generated_judge_family_assignment(
    *,
    dataset: DatasetName,
    item: SubsetItem,
    question: str,
    raw_task: dict[str, Any],
    persona_artifacts_dir: Path,
    judge_bank_dir: Path | None,
    gpqa_family_cache_path: Path | None,
    judge_generator_model: str | None,
    judge_engine: Any | None,
) -> dict[str, Any] | None:
    if dataset not in {"aime25", "gpqa", "hle"}:
        return None
    resolved_judge_bank_dir = judge_bank_dir or default_judge_bank_dir(artifacts_dir=persona_artifacts_dir)
    gpqa_cache_path = gpqa_family_cache_path or default_gpqa_family_cache_path(judge_bank_dir=resolved_judge_bank_dir)
    assignment = resolve_judge_family_assignment(
        dataset=dataset,
        item_uid=item.item_uid,
        question=question,
        raw_task=raw_task,
        gpqa_family_cache_path=gpqa_cache_path if dataset == "gpqa" else None,
        gpqa_classifier_engine=judge_engine if dataset == "gpqa" else None,
        gpqa_classifier_model=judge_generator_model if dataset == "gpqa" else None,
    )
    return assignment.to_dict()


def _resolve_runtime_judge_card(
    *,
    dataset: DatasetName,
    item: SubsetItem,
    question: str,
    raw_task: dict[str, Any],
    persona_artifacts_dir: Path,
    judge_bank_dir: Path | None,
    judge_bank_refresh: bool,
    gpqa_family_cache_path: Path | None,
    judge_persona_mode: str,
    persona_backend: str,
    judge_generator_model: str | None,
    judge_engine: Any | None,
) -> tuple[Any | None, dict[str, Any] | None]:
    if judge_persona_mode != "benchmark_family_bank":
        family_assignment = None
        benchmark_family_override = None
        if judge_persona_mode != "neutral_baseline":
            family_assignment = _resolve_generated_judge_family_assignment(
                dataset=dataset,
                item=item,
                question=question,
                raw_task=raw_task,
                persona_artifacts_dir=persona_artifacts_dir,
                judge_bank_dir=judge_bank_dir,
                gpqa_family_cache_path=gpqa_family_cache_path,
                judge_generator_model=judge_generator_model,
                judge_engine=judge_engine,
            )
            if isinstance(family_assignment, dict):
                benchmark_family_override = family_assignment.get("judge_family")
        judge_card = build_judge_card(
            dataset=dataset,
            raw_task=raw_task,
            question=question,
            mode=cast(Any, judge_persona_mode),
            benchmark_family_override=cast(str | None, benchmark_family_override),
            engine=judge_engine,
            generator_model=judge_generator_model,
            backend=persona_backend,
        )
        meta = None
        if family_assignment is not None:
            meta = {"judge_family_assignment": family_assignment}
        return judge_card, meta
    resolved_judge_bank_dir = judge_bank_dir or default_judge_bank_dir(artifacts_dir=persona_artifacts_dir)
    gpqa_cache_path = gpqa_family_cache_path or default_gpqa_family_cache_path(judge_bank_dir=resolved_judge_bank_dir)
    family_assignment = resolve_judge_family_assignment(
        dataset=dataset,
        item_uid=item.item_uid,
        question=question,
        raw_task=raw_task,
        gpqa_family_cache_path=gpqa_cache_path if dataset == "gpqa" else None,
        gpqa_classifier_engine=judge_engine if dataset == "gpqa" else None,
        gpqa_classifier_model=judge_generator_model if dataset == "gpqa" else None,
    )
    judge_card, judge_bank_artifact, bank_path = ensure_judge_bank_card(
        judge_bank_dir=resolved_judge_bank_dir,
        dataset=dataset,
        judge_family=family_assignment.judge_family,
        engine=judge_engine,
        generator_model=judge_generator_model,
        backend=persona_backend,
        refresh=judge_bank_refresh,
    )
    return judge_card, {
        "judge_family_assignment": family_assignment.to_dict(),
        "judge_bank_dir": str(resolved_judge_bank_dir),
        "judge_bank_path": str(bank_path),
        "judge_bank_artifact_version": judge_bank_artifact.artifact_version,
    }


__all__ = [
    "_resolve_persona_artifact",
    "_resolve_runtime_judge_card",
    "run_persona_generation",
]
