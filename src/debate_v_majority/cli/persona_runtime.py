from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from ..personas import (
    AxisSelection,
    JudgeCard,
    PersonaArtifact,
    PersonaCard,
    PersonaDescriptor,
    PersonaGenerationConfig,
    artifact_path as persona_artifact_path,
    build_judge_card,
    build_persona_artifact,
    build_persona_validator_metadata,
    default_gpqa_family_cache_path,
    default_judge_bank_dir,
    ensure_judge_bank_card,
    expand_cards,
    generate_descriptors_from_state,
    legacy_artifact_path as legacy_persona_artifact_path,
    load_artifact as load_persona_artifact,
    prepare_descriptor_generation,
    resolve_judge_family_assignment,
    save_artifact as save_persona_artifact,
)
from ..personas.axes import AXIS_BANK_VERSION
from ..personas.generator import (
    COVERAGE_AUDIT_VERSION,
    GENERATION_SETTINGS_VERSION,
    SEMANTIC_REDUNDANCY_VERSION,
    SLOT_SAMPLING_VERSION,
    _effective_persona_seed,
)
from ..personas.schema import AxisSelectionMode, JudgePersonaMode
from ..personas.prompt_templates import (
    ARTIFACT_VERSION,
    CARD_PROMPT_VERSION,
    DESCRIPTOR_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
)

from .stage_state import (
    StageEntry,
    append_stage_entry,
    load_latest_stage_entry_of_type,
    make_stage_entry,
    path_setting,
    subset_item_resume_signature,
)
from .subset import SubsetItem


def _parse_question_answer(dataset: DatasetName, sample: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    adapter = resolve_dataset_adapter(dataset)
    return cast(tuple[str, str, dict[str, Any]], adapter.parse_question_answer(sample))


def _persona_generation_settings(
    *,
    n_personas: int,
    persona_seed: int,
    item_uid: str | None = None,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    judge_persona_mode: str,
    backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    axes_file: Path | None,
    n_plain_agents: int = 0,
) -> dict[str, Any]:
    settings = {
        "generation_settings_version": GENERATION_SETTINGS_VERSION,
        "n_personas": int(n_personas),
        "persona_seed": int(persona_seed),
        "axis_mode": str(axis_mode),
        "fixed_axis_count": int(fixed_axis_count),
        "task_axis_count": int(task_axis_count),
        "sampling_method": str(sampling_method),
        "slot_sampling_version": SLOT_SAMPLING_VERSION,
        "slot_role_scheme": "generic_coverage_v1",
        "population_design_version": "generic_persona_coverage.v1",
        "judge_persona_mode": str(judge_persona_mode),
        "backend": str(backend),
        "generator_model": generator_model,
        "judge_generator_model": judge_generator_model,
        "axis_bank_version": AXIS_BANK_VERSION,
        "generic_axis_bank_version": AXIS_BANK_VERSION,
        "semantic_redundancy_version": SEMANTIC_REDUNDANCY_VERSION,
        "coverage_audit_version": COVERAGE_AUDIT_VERSION,
        "axes_file": None if axes_file is None else str(axes_file),
        "n_plain_agents": int(n_plain_agents),
    }
    if item_uid is not None:
        settings["effective_persona_seed"] = _effective_persona_seed(
            persona_seed=persona_seed,
            item_uid=item_uid,
        )
    return settings


def _staged_persona_resume_settings(
    *,
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    judge_persona_mode: str,
    requested_backend: str,
    effective_backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    axes_file: Path | None,
    n_plain_agents: int = 0,
) -> dict[str, Any]:
    settings = _persona_generation_settings(
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        judge_persona_mode=judge_persona_mode,
        backend=requested_backend,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        axes_file=axes_file,
        n_plain_agents=n_plain_agents,
    )
    settings.update(
        {
            "effective_backend": str(effective_backend),
        }
    )
    return settings


def _validate_staged_persona_resume(
    *,
    prev_entry: StageEntry,
    dataset: DatasetName,
    items: list[SubsetItem],
    current_settings: dict[str, Any],
) -> None:
    if prev_entry.dataset != str(dataset):
        raise ValueError(f"Persona state dataset mismatch: {prev_entry.dataset} != {dataset}")

    saved_items = [subset_item_resume_signature(row) for row in prev_entry.items]
    current_items = [subset_item_resume_signature(item) for item in items]
    if saved_items != current_items:
        raise ValueError("Persona state items do not match the requested subset")

    saved_settings = prev_entry.meta.get("resume_settings")
    if not isinstance(saved_settings, dict):
        raise ValueError("Persona state generation settings mismatch")
    if dict(saved_settings) != current_settings:
        raise ValueError("Persona state generation settings mismatch")


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
    expected_n_plain = int(generation_settings.get("n_plain_agents", 0))
    actual_n_plain = artifact.n_plain_agents
    if actual_n_plain != expected_n_plain:
        return f"Artifact {artifact_path} n_plain_agents mismatch: {actual_n_plain} != {expected_n_plain}"
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
    row: dict[str, Any] = {
        "schema_version": "phase0.v2",
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
        "generation_settings": data.get("generation_settings", {}),
    }
    if artifact.slot_layout is not None:
        row["slot_layout"] = list(artifact.slot_layout)
    return row


def persona_rows_from_stage_entry(entry: StageEntry) -> list[dict[str, Any]]:
    if str(entry.stage_type) != "persona":
        raise ValueError(f"Expected persona stage entry, got {entry.stage_type!r}")

    rows: list[dict[str, Any]] = []
    per_item = dict(entry.persona_data or {})
    for item in list(entry.items or []):
        item_uid = str(dict(item).get("item_uid") or "")
        item_data = per_item.get(item_uid)
        if not isinstance(item_data, dict):
            raise ValueError(f"Persona stage entry is missing state for item {item_uid!r}")
        artifact_dict = item_data.get("artifact")
        if not isinstance(artifact_dict, dict):
            raise ValueError("Persona stage entry does not contain a final artifact payload")
        artifact = PersonaArtifact.from_dict(artifact_dict)
        artifact_path = item_data.get("artifact_path")
        rows.append(
            _persona_artifact_row(
                artifact,
                artifact_path_str=None if artifact_path is None else str(artifact_path),
            )
        )
    return rows


def persona_artifacts_from_rows(rows: list[dict[str, Any]]) -> dict[str, PersonaArtifact]:
    artifacts_by_item: dict[str, PersonaArtifact] = {}
    for row in rows:
        from_dict_payload: dict[str, Any] = {
            "artifact_version": row["artifact_version"],
            "dataset": row["dataset"],
            "item_uid": row["item_uid"],
            "dataset_revision": row.get("dataset_revision"),
            "item_display_id": row.get("item_display_id"),
            "persona_seed": row.get("persona_seed"),
            "generator_model": row.get("generator_model"),
            "judge_generator_model": row.get("judge_generator_model"),
            "axes": row["axes"],
            "sampled_points": row.get("sampled_points", []),
            "descriptors": row.get("descriptors", []),
            "cards": row.get("cards", []),
            "judge_card": row.get("judge_card"),
            "prompt_versions": row.get("prompt_versions", {}),
            "created_at": row["created_at"],
            "generation_settings": row.get("generation_settings", {}),
            "validator_metadata": row.get("validator_metadata", {}),
        }
        if "slot_layout" in row:
            from_dict_payload["slot_layout"] = row["slot_layout"]
        artifact = PersonaArtifact.from_dict(from_dict_payload)
        artifacts_by_item[artifact.item_uid] = artifact
    return artifacts_by_item


def _print_persona_artifact_summary(
    artifact: PersonaArtifact,
    *,
    output_file: TextIO,
    dump_cards: bool,
    artifact_path_str: str | None,
) -> None:
    plain_count = artifact.n_plain_agents
    slot_label = f" plain_agents={plain_count}" if plain_count > 0 else ""
    print(
        f"[persona] item_uid={artifact.item_uid} axes={len(artifact.axes.axes)} personas={len(artifact.cards)}{slot_label}",
        file=output_file,
    )
    if artifact_path_str:
        print(f"[persona] artifact={artifact_path_str}", file=output_file)
    for card in artifact.cards:
        print(f"[persona] {card.persona_id}: {card.title}", file=output_file)
        if dump_cards:
            print(card.initial_system_prompt, file=output_file)
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
    n_plain_agents: int = 0,
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
        n_plain_agents=n_plain_agents,
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

    config = _persona_config_for_item(
        dataset=dataset,
        item=item,
        question=question,
        raw_task=raw_task,
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
        n_plain_agents=n_plain_agents,
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
        judge_engine=judge_engine or generator_engine,
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
    n_plain_agents: int = 0,
) -> list[dict[str, Any]]:
    if replay:
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
                judge_engine=judge_engine or generator_engine,
                axes_file=axes_file,
                judge_bank_dir=judge_bank_dir,
                judge_bank_refresh=judge_bank_refresh,
                gpqa_family_cache_path=gpqa_family_cache_path,
                save_artifact=save_artifacts,
                replay=True,
                n_plain_agents=n_plain_agents,
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

    effective_backend = "llm"
    resume_settings = _staged_persona_resume_settings(
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        judge_persona_mode=judge_persona_mode,
        requested_backend=backend,
        effective_backend=effective_backend,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        axes_file=axes_file,
        n_plain_agents=n_plain_agents,
    )
    internal_stage_state_file = _persona_internal_stage_state_path(
        artifacts_dir=artifacts_dir,
        dataset=dataset,
        items=items,
        resume_settings=resume_settings,
    )
    final_entry: StageEntry | None = None
    try:
        for stage_name in _PERSONA_STAGE_ORDER:
            final_entry = run_persona_generation_staged(
                dataset=dataset,
                items=items,
                artifacts_dir=artifacts_dir,
                stage_state_file=internal_stage_state_file,
                persona_stage=stage_name,
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
                judge_engine=judge_engine or generator_engine,
                axes_file=axes_file,
                judge_bank_dir=judge_bank_dir,
                judge_bank_refresh=judge_bank_refresh,
                gpqa_family_cache_path=gpqa_family_cache_path,
                save_artifacts=save_artifacts,
                replay=False,
                dump_cards=dump_cards,
                summary_file=summary_file,
                n_plain_agents=n_plain_agents,
            )
    finally:
        if final_entry is not None and final_entry.completed_stage == "judge_card" and internal_stage_state_file.exists():
            try:
                internal_stage_state_file.unlink()
            except OSError:
                pass
    if final_entry is None:
        return []
    return persona_rows_from_stage_entry(final_entry)


_PERSONA_STAGE_ORDER = ["axes", "descriptors", "cards", "judge_card"]


def _stage_index(stage: str) -> int:
    return _PERSONA_STAGE_ORDER.index(stage)


def _persona_failure_audit_path(stage_state_file: Path) -> Path:
    return stage_state_file.parent / f"{stage_state_file.stem}.persona_failures.jsonl"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    return value


def _append_persona_failure_audit(
    *,
    stage_state_file: Path,
    dataset: DatasetName,
    failed_stage: str,
    last_completed_stage: str,
    items: list[SubsetItem],
    per_item: dict[str, dict[str, Any]],
    exc: Exception,
) -> Path:
    audit_path = _persona_failure_audit_path(stage_state_file)
    record = {
        "schema_version": "persona_stage_failure.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "failed_stage": failed_stage,
        "last_completed_stage": last_completed_stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "items": [subset_item_resume_signature(item) for item in items],
        "persona_data": deepcopy(per_item),
    }
    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, dict):
        record["failure_metadata"] = _json_ready(deepcopy(metadata))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return audit_path


def _persona_stage_item_complete(*, stage_name: str, item_data: dict[str, Any]) -> bool:
    if stage_name == "axes":
        return "axis_selection" in item_data and "sampled_points" in item_data
    if stage_name == "descriptors":
        return (
            "descriptors" in item_data
            and "descriptor_validator_metadata" in item_data
            and "descriptor_call_metadata" in item_data
        )
    if stage_name == "cards":
        return (
            "cards" in item_data
            and "card_metadata" in item_data
            and "card_call_metadata" in item_data
        )
    if stage_name == "judge_card":
        return (
            "judge_card" in item_data
            and "judge_bank_meta" in item_data
            and "artifact_created_at" in item_data
            and "artifact" in item_data
        )
    raise ValueError(f"Unknown persona stage: {stage_name}")


def _persona_stage_pending_items(
    *,
    stage_name: str,
    items: list[SubsetItem],
    per_item: dict[str, dict[str, Any]],
) -> list[SubsetItem]:
    return [
        item
        for item in items
        if not _persona_stage_item_complete(stage_name=stage_name, item_data=dict(per_item.get(item.item_uid) or {}))
    ]


def _persona_stage_meta(
    *,
    persona_stage: str,
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    backend: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    current_resume_settings: dict[str, Any],
    active_stage: str,
    active_stage_complete: bool,
) -> dict[str, Any]:
    return {
        "persona_stage": persona_stage,
        "n_personas": n_personas,
        "persona_seed": persona_seed,
        "axis_mode": axis_mode,
        "backend": backend,
        "generator_model": generator_model,
        "judge_generator_model": judge_generator_model,
        "resume_settings": dict(current_resume_settings),
        "active_stage": active_stage,
        "active_stage_complete": bool(active_stage_complete),
    }


def _staged_persona_start_index(*, prev_entry: StageEntry | None) -> int:
    if prev_entry is None:
        return 0
    active_stage = str(prev_entry.meta.get("active_stage") or "")
    active_stage_complete = bool(prev_entry.meta.get("active_stage_complete", True))
    if active_stage in _PERSONA_STAGE_ORDER and not active_stage_complete:
        return _stage_index(active_stage)
    completed = prev_entry.completed_stage
    if completed in _PERSONA_STAGE_ORDER:
        return _stage_index(completed) + 1
    return 0


def _persona_internal_stage_state_path(
    *,
    artifacts_dir: Path,
    dataset: DatasetName,
    items: list[SubsetItem],
    resume_settings: dict[str, Any],
) -> Path:
    payload = {
        "dataset": str(dataset),
        "items": [subset_item_resume_signature(item) for item in items],
        "resume_settings": dict(resume_settings),
    }
    digest = hashlib.sha256(
        json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return artifacts_dir / f".persona_stage_state.{digest}.jsonl"


def _build_staged_persona_artifact(
    *,
    config: PersonaGenerationConfig,
    item_data: dict[str, Any],
    backend: str,
) -> PersonaArtifact:
    axis_selection = AxisSelection.from_dict(item_data["axis_selection"])
    descriptors = [PersonaDescriptor.from_dict(x) for x in item_data.get("descriptors", [])]
    cards = [PersonaCard.from_dict(x) for x in item_data.get("cards", [])]
    judge_card = (
        JudgeCard.from_dict(item_data["judge_card"])
        if item_data.get("judge_card") is not None
        else None
    )
    descriptor_validator_metadata = dict(item_data.get("descriptor_validator_metadata") or {})
    if not descriptor_validator_metadata and item_data.get("descriptor_call_metadata") is not None:
        descriptor_validator_metadata = {
            "descriptor_call_metadata": item_data.get("descriptor_call_metadata"),
            "descriptor_prompt_version": DESCRIPTOR_PROMPT_VERSION,
            "descriptor_backend": backend,
        }
    card_metadata = dict(item_data.get("card_metadata") or {})
    if not card_metadata and item_data.get("card_call_metadata") is not None:
        card_metadata = {
            "card_call_metadata": item_data.get("card_call_metadata"),
            "card_prompt_version": CARD_PROMPT_VERSION,
            "card_backend": backend,
        }
    validator_metadata = build_persona_validator_metadata(
        axis_selection=axis_selection,
        descriptor_validator_metadata=descriptor_validator_metadata,
        card_metadata=card_metadata,
        backend=backend,
        judge_card=judge_card,
    )
    judge_bank_meta = item_data.get("judge_bank_meta")
    if isinstance(judge_bank_meta, dict):
        validator_metadata["judge_bank"] = judge_bank_meta
    return PersonaArtifact(
        artifact_version=ARTIFACT_VERSION,
        dataset=config.dataset,
        item_uid=config.item_uid,
        dataset_revision=config.dataset_revision,
        item_display_id=config.item_display_id,
        persona_seed=config.persona_seed,
        generator_model=config.generator_model,
        judge_generator_model=config.judge_generator_model,
        axes=axis_selection,
        sampled_points=[
            {str(k): float(v) for k, v in point.items()}
            for point in item_data.get("sampled_points", [])
        ],
        descriptors=descriptors,
        cards=cards,
        judge_card=judge_card,
        prompt_versions={
            "axis": axis_selection.generator_prompt_version,
            "descriptor": DESCRIPTOR_PROMPT_VERSION,
            "card": CARD_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
        },
        created_at=str(item_data.get("artifact_created_at") or datetime.now(timezone.utc).isoformat()),
        generation_settings=_persona_generation_settings(
            n_personas=config.n_personas,
            persona_seed=config.persona_seed,
            item_uid=config.item_uid,
            axis_mode=str(config.axis_mode),
            fixed_axis_count=config.fixed_axis_count,
            task_axis_count=config.task_axis_count,
            sampling_method=str(config.sampling_method),
            judge_persona_mode=str(config.judge_persona_mode),
            backend=backend,
            generator_model=config.generator_model,
            judge_generator_model=config.judge_generator_model,
            axes_file=config.axes_file,
        ),
        validator_metadata=validator_metadata,
    )


def _persona_config_for_item(
    *,
    dataset: DatasetName,
    item: SubsetItem,
    question: str,
    raw_task: dict[str, Any],
    n_personas: int,
    persona_seed: int,
    axis_mode: AxisSelectionMode,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: Literal["maximin", "halton"],
    generator_model: str | None,
    judge_generator_model: str | None,
    judge_persona_mode: JudgePersonaMode,
    backend: Literal["llm"],
    axes_file: Path | None,
    n_plain_agents: int = 0,
) -> PersonaGenerationConfig:
    return PersonaGenerationConfig(
        dataset=dataset,
        question=question,
        raw_task=raw_task,
        item_uid=item.item_uid,
        item_display_id=item.item_display_id,
        dataset_revision=item.dataset_revision,
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        judge_persona_mode=judge_persona_mode,
        backend=backend,
        axes_file=axes_file,
        n_plain_agents=n_plain_agents,
    )


def _descriptor_stage_metadata_from_artifact(artifact: PersonaArtifact) -> dict[str, Any]:
    return {
        key: deepcopy(artifact.validator_metadata.get(key))
        for key in (
            "descriptor_validations",
            "descriptor_duplicates",
            "descriptor_semantic_redundancy",
            "descriptor_coverage_audit",
            "descriptor_prompt_version",
            "descriptor_semantic_redundancy_version",
            "descriptor_coverage_audit_version",
            "descriptor_backend",
            "descriptor_call_metadata",
            "generation_settings_version",
            "semantic_redundancy_version",
            "coverage_audit_version",
            "slot_sampling_version",
            "slot_role_scheme",
            "population_design_version",
            "axis_bank_version",
            "generic_axis_bank_version",
        )
        if key in artifact.validator_metadata
    }


def _card_stage_metadata_from_artifact(artifact: PersonaArtifact) -> dict[str, Any]:
    return {
        key: deepcopy(artifact.validator_metadata.get(key))
        for key in (
            "card_validations",
            "card_duplicates",
            "card_semantic_redundancy",
            "card_coverage_audit",
            "card_prompt_version",
            "card_semantic_redundancy_version",
            "card_coverage_audit_version",
            "card_backend",
            "card_call_metadata",
            "generation_settings_version",
            "semantic_redundancy_version",
            "coverage_audit_version",
            "slot_sampling_version",
            "slot_role_scheme",
            "population_design_version",
            "axis_bank_version",
            "generic_axis_bank_version",
            "card_schema_version",
        )
        if key in artifact.validator_metadata
    }


PERSONA_STAGE_MAX_WORKERS = 8


def _persona_stage_worker_count(n_items: int) -> int:
    return max(1, min(PERSONA_STAGE_MAX_WORKERS, int(n_items)))


def _persona_stage_parallel_map(
    *,
    stage_name: str,
    stage_inputs: list[Any],
    worker: Any,
    on_result: Any | None = None,
) -> list[Any]:
    if len(stage_inputs) <= 1:
        results = [worker(stage_input) for stage_input in stage_inputs]
        if on_result is not None:
            for idx, result in enumerate(results):
                on_result(idx, result)
        return results
    results: list[Any] = [None] * len(stage_inputs)
    first_exc: Exception | None = None
    with ThreadPoolExecutor(
        max_workers=_persona_stage_worker_count(len(stage_inputs)),
        thread_name_prefix=f"persona_{stage_name}",
    ) as executor:
        future_to_idx = {
            executor.submit(worker, stage_input): idx
            for idx, stage_input in enumerate(stage_inputs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc
                    for pending in future_to_idx:
                        if pending is not future:
                            pending.cancel()
                continue
            results[idx] = result
            if on_result is not None:
                on_result(idx, result)
    if first_exc is not None:
        raise first_exc
    return results


def _persona_inputs_for_items(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    n_personas: int,
    persona_seed: int,
    axis_mode: str,
    fixed_axis_count: int,
    task_axis_count: int,
    sampling_method: str,
    generator_model: str | None,
    judge_generator_model: str | None,
    judge_persona_mode: str,
    backend: str,
    axes_file: Path | None,
    n_plain_agents: int = 0,
) -> list[dict[str, Any]]:
    parsed_inputs: list[dict[str, Any]] = []
    for item in items:
        question, _, raw_task = _parse_question_answer(dataset, item.raw_task)
        parsed_inputs.append(
            {
                "item": item,
                "question": question,
                "raw_task": raw_task,
                "config": _persona_config_for_item(
                    dataset=dataset,
                    item=item,
                    question=question,
                    raw_task=raw_task,
                    n_personas=n_personas,
                    persona_seed=persona_seed,
                    axis_mode=axis_mode,
                    fixed_axis_count=fixed_axis_count,
                    task_axis_count=task_axis_count,
                    sampling_method=sampling_method,
                    generator_model=generator_model,
                    judge_generator_model=judge_generator_model,
                    judge_persona_mode=judge_persona_mode,
                    backend=backend,
                    axes_file=axes_file,
                    n_plain_agents=n_plain_agents,
                ),
            }
        )
    return parsed_inputs


def _run_axes_stage_parallel(
    *,
    parsed_inputs: list[dict[str, Any]],
    generator_engine: Any | None,
    on_result: Any | None = None,
) -> list[dict[str, Any]]:
    def _worker(parsed_input: dict[str, Any]) -> dict[str, Any]:
        axis_selection, points, _stage_backend = prepare_descriptor_generation(
            config=parsed_input["config"],
            engine=generator_engine,
        )
        return {
            "axis_selection": asdict(axis_selection),
            "sampled_points": points,
        }

    return _persona_stage_parallel_map(
        stage_name="axes",
        stage_inputs=parsed_inputs,
        worker=_worker,
        on_result=on_result,
    )


def _run_descriptors_stage_parallel(
    *,
    parsed_inputs: list[dict[str, Any]],
    per_item_data: dict[str, dict[str, Any]],
    generator_engine: Any | None,
    on_result: Any | None = None,
) -> list[dict[str, Any]]:
    stage_inputs = [
        {
            "parsed_input": parsed_input,
            "axis_selection": AxisSelection.from_dict(per_item_data[parsed_input["item"].item_uid]["axis_selection"]),
            "sampled_points": [
                {str(k): float(v) for k, v in point.items()}
                for point in per_item_data[parsed_input["item"].item_uid]["sampled_points"]
            ],
        }
        for parsed_input in parsed_inputs
    ]

    def _worker(stage_input: dict[str, Any]) -> dict[str, Any]:
        descriptors, descriptor_meta = generate_descriptors_from_state(
            config=stage_input["parsed_input"]["config"],
            axis_selection=stage_input["axis_selection"],
            points=stage_input["sampled_points"],
            engine=generator_engine,
        )
        return {
            "descriptors": [asdict(desc) for desc in descriptors],
            "descriptor_validator_metadata": dict(descriptor_meta["validator_metadata"]),
            "descriptor_call_metadata": descriptor_meta["validator_metadata"].get("descriptor_call_metadata"),
        }

    return _persona_stage_parallel_map(
        stage_name="descriptors",
        stage_inputs=stage_inputs,
        worker=_worker,
        on_result=on_result,
    )


def _run_cards_stage_parallel(
    *,
    parsed_inputs: list[dict[str, Any]],
    per_item_data: dict[str, dict[str, Any]],
    generator_engine: Any | None,
    effective_backend: str,
    on_result: Any | None = None,
) -> list[dict[str, Any]]:
    stage_inputs = [
        {
            "parsed_input": parsed_input,
            "descriptors": [
                PersonaDescriptor.from_dict(desc_dict)
                for desc_dict in per_item_data[parsed_input["item"].item_uid]["descriptors"]
            ],
        }
        for parsed_input in parsed_inputs
    ]

    def _worker(stage_input: dict[str, Any]) -> dict[str, Any]:
        parsed_input = stage_input["parsed_input"]
        cards, card_meta = expand_cards(
            stage_input["descriptors"],
            dataset=parsed_input["config"].dataset,
            question=parsed_input["question"],
            raw_task=parsed_input["raw_task"],
            engine=generator_engine,
            backend=effective_backend,
        )
        return {
            "cards": [asdict(card) for card in cards],
            "card_metadata": dict(card_meta),
            "card_call_metadata": deepcopy(card_meta.get("card_call_metadata") or []),
        }

    return _persona_stage_parallel_map(
        stage_name="cards",
        stage_inputs=stage_inputs,
        worker=_worker,
        on_result=on_result,
    )


def run_persona_generation_staged(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    artifacts_dir: Path,
    stage_state_file: Path,
    persona_stage: str,
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
    save_artifacts: bool = False,
    replay: bool = False,
    dump_cards: bool = False,
    summary_file: TextIO,
    n_plain_agents: int = 0,
) -> StageEntry:
    """Run persona generation one stage at a time, appending state after each batch."""

    target_idx = _stage_index(persona_stage)
    items_ser = [asdict(it) for it in items]
    effective_backend = "llm"
    current_resume_settings = _staged_persona_resume_settings(
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        judge_persona_mode=judge_persona_mode,
        requested_backend=backend,
        effective_backend=effective_backend,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        axes_file=axes_file,
        n_plain_agents=n_plain_agents,
    )

    prev_entry = load_latest_stage_entry_of_type(stage_state_file, "persona") if stage_state_file.exists() else None
    if prev_entry is not None:
        _validate_staged_persona_resume(
            prev_entry=prev_entry,
            dataset=dataset,
            items=items,
            current_settings=current_resume_settings,
        )

    per_item: dict[str, dict[str, Any]] = {} if prev_entry is None else dict(prev_entry.persona_data)
    parsed_inputs = _persona_inputs_for_items(
        dataset=dataset,
        items=items,
        n_personas=n_personas,
        persona_seed=persona_seed,
        axis_mode=axis_mode,
        fixed_axis_count=fixed_axis_count,
        task_axis_count=task_axis_count,
        sampling_method=sampling_method,
        generator_model=generator_model,
        judge_generator_model=judge_generator_model,
        judge_persona_mode=judge_persona_mode,
        backend=backend,
        axes_file=axes_file,
        n_plain_agents=n_plain_agents,
    )
    parsed_input_by_uid = {
        parsed_input["item"].item_uid: parsed_input
        for parsed_input in parsed_inputs
    }

    def _item_data(item: SubsetItem) -> dict[str, Any]:
        return per_item.setdefault(item.item_uid, {})

    start_idx = _staged_persona_start_index(prev_entry=prev_entry)

    if prev_entry is not None and start_idx > target_idx:
        print(
            f"[staged-persona] requested stage '{persona_stage}' is already complete in {stage_state_file}",
            file=summary_file,
        )
        return prev_entry

    replay_artifacts: dict[str, tuple[PersonaArtifact, Path]] = {}
    if replay:
        for item in items:
            question, _, raw_task = _parse_question_answer(dataset, item.raw_task)
            replay_artifacts[item.item_uid] = _resolve_persona_artifact(
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
                        judge_engine=judge_engine or generator_engine,
                axes_file=axes_file,
                judge_bank_dir=judge_bank_dir,
                n_plain_agents=n_plain_agents,
                judge_bank_refresh=judge_bank_refresh,
                gpqa_family_cache_path=gpqa_family_cache_path,
                save_artifact=False,
                replay=True,
            )

    for stage_idx in range(start_idx, target_idx + 1):
        stage_name = _PERSONA_STAGE_ORDER[stage_idx]
        stage_items = _persona_stage_pending_items(stage_name=stage_name, items=items, per_item=per_item)
        print(f"[staged-persona] running stage: {stage_name}", file=summary_file)
        try:
            if stage_name == "axes":
                if replay:
                    for item in stage_items:
                        d = _item_data(item)
                        artifact, _artifact_path = replay_artifacts[item.item_uid]
                        d["axis_selection"] = asdict(artifact.axes)
                        d["sampled_points"] = deepcopy(artifact.sampled_points)
                else:
                    stage_parsed_inputs = [parsed_input_by_uid[item.item_uid] for item in stage_items]
                    axis_results = _run_axes_stage_parallel(
                        parsed_inputs=stage_parsed_inputs,
                        generator_engine=generator_engine,
                        on_result=lambda idx, stage_result: _item_data(stage_parsed_inputs[idx]["item"]).update(
                            {
                                "axis_selection": stage_result["axis_selection"],
                                "sampled_points": stage_result["sampled_points"],
                            }
                        ),
                    )
                    for parsed_input, stage_result in zip(stage_parsed_inputs, axis_results):
                        d = _item_data(parsed_input["item"])
                        d["axis_selection"] = stage_result["axis_selection"]
                        d["sampled_points"] = stage_result["sampled_points"]

            elif stage_name == "descriptors":
                if replay:
                    for item in stage_items:
                        d = _item_data(item)
                        artifact, _artifact_path = replay_artifacts[item.item_uid]
                        d["descriptors"] = [asdict(desc) for desc in artifact.descriptors]
                        d["descriptor_validator_metadata"] = _descriptor_stage_metadata_from_artifact(artifact)
                        d["descriptor_call_metadata"] = artifact.validator_metadata.get("descriptor_call_metadata")
                else:
                    stage_parsed_inputs = [parsed_input_by_uid[item.item_uid] for item in stage_items]
                    descriptor_results = _run_descriptors_stage_parallel(
                        parsed_inputs=stage_parsed_inputs,
                        per_item_data=per_item,
                        generator_engine=generator_engine,
                        on_result=lambda idx, stage_result: _item_data(stage_parsed_inputs[idx]["item"]).update(
                            {
                                "descriptors": stage_result["descriptors"],
                                "descriptor_validator_metadata": stage_result["descriptor_validator_metadata"],
                                "descriptor_call_metadata": stage_result["descriptor_call_metadata"],
                            }
                        ),
                    )
                    for parsed_input, stage_result in zip(stage_parsed_inputs, descriptor_results):
                        d = _item_data(parsed_input["item"])
                        d["descriptors"] = stage_result["descriptors"]
                        d["descriptor_validator_metadata"] = stage_result["descriptor_validator_metadata"]
                        d["descriptor_call_metadata"] = stage_result["descriptor_call_metadata"]

            elif stage_name == "cards":
                if replay:
                    for item in stage_items:
                        d = _item_data(item)
                        artifact, _artifact_path = replay_artifacts[item.item_uid]
                        d["cards"] = [asdict(card) for card in artifact.cards]
                        d["card_metadata"] = _card_stage_metadata_from_artifact(artifact)
                        d["card_call_metadata"] = deepcopy(artifact.validator_metadata.get("card_call_metadata") or [])
                else:
                    stage_parsed_inputs = [parsed_input_by_uid[item.item_uid] for item in stage_items]
                    card_results = _run_cards_stage_parallel(
                        parsed_inputs=stage_parsed_inputs,
                        per_item_data=per_item,
                        generator_engine=generator_engine,
                        effective_backend=effective_backend,
                        on_result=lambda idx, stage_result: _item_data(stage_parsed_inputs[idx]["item"]).update(
                            {
                                "cards": stage_result["cards"],
                                "card_metadata": stage_result["card_metadata"],
                                "card_call_metadata": stage_result["card_call_metadata"],
                            }
                        ),
                    )
                    for parsed_input, stage_result in zip(stage_parsed_inputs, card_results):
                        d = _item_data(parsed_input["item"])
                        d["cards"] = stage_result["cards"]
                        d["card_metadata"] = stage_result["card_metadata"]
                        d["card_call_metadata"] = stage_result["card_call_metadata"]

            elif stage_name == "judge_card":
                for item in stage_items:
                    d = _item_data(item)
                    artifact_path_str: str | None = None
                    if replay:
                        artifact, artifact_file = replay_artifacts[item.item_uid]
                        judge_bank_meta = artifact.validator_metadata.get("judge_bank")
                        d["judge_card"] = asdict(artifact.judge_card) if artifact.judge_card is not None else None
                        d["judge_bank_meta"] = judge_bank_meta if isinstance(judge_bank_meta, dict) else None
                        d["artifact_created_at"] = artifact.created_at
                        d["artifact"] = artifact.to_dict()
                        artifact_path_str = str(artifact_file) if artifact_file.exists() else None
                        d["artifact_path"] = artifact_path_str
                        _print_persona_artifact_summary(
                            artifact,
                            output_file=summary_file,
                            dump_cards=dump_cards,
                            artifact_path_str=artifact_path_str,
                        )
                        continue
                    parsed_input = parsed_input_by_uid[item.item_uid]
                    question = parsed_input["question"]
                    raw_task = parsed_input["raw_task"]
                    judge_card, judge_bank_meta = _resolve_runtime_judge_card(
                        dataset=dataset, item=item, question=question, raw_task=raw_task,
                        persona_artifacts_dir=artifacts_dir,
                        judge_bank_dir=judge_bank_dir, judge_bank_refresh=judge_bank_refresh,
                        gpqa_family_cache_path=gpqa_family_cache_path,
                        judge_persona_mode=judge_persona_mode, persona_backend=effective_backend,
                        judge_generator_model=judge_generator_model, judge_engine=judge_engine or generator_engine,
                    )
                    d["judge_card"] = asdict(judge_card) if judge_card is not None else None
                    d["judge_bank_meta"] = judge_bank_meta
                    artifact = _build_staged_persona_artifact(
                        config=parsed_input["config"],
                        item_data=d,
                        backend=effective_backend,
                    )
                    if save_artifacts:
                        artifact_file = save_persona_artifact(artifacts_dir=artifacts_dir, artifact=artifact)
                        artifact_path_str = str(artifact_file)
                    d["artifact_created_at"] = artifact.created_at
                    d["artifact"] = artifact.to_dict()
                    d["artifact_path"] = artifact_path_str
                    _print_persona_artifact_summary(
                        artifact,
                        output_file=summary_file,
                        dump_cards=dump_cards,
                        artifact_path_str=artifact_path_str,
                    )
        except Exception as exc:
            last_completed_stage = prev_entry.completed_stage if prev_entry is not None else ""
            if stage_idx > 0:
                last_completed_stage = _PERSONA_STAGE_ORDER[stage_idx - 1]
            partial_entry = make_stage_entry(
                stage_type="persona",
                completed_stage=last_completed_stage,
                dataset=str(dataset),
                items=items_ser,
                persona_data=per_item,
                meta=_persona_stage_meta(
                    persona_stage=persona_stage,
                    n_personas=n_personas,
                    persona_seed=persona_seed,
                    axis_mode=axis_mode,
                    backend=effective_backend,
                    generator_model=generator_model,
                    judge_generator_model=judge_generator_model,
                    current_resume_settings=current_resume_settings,
                    active_stage=stage_name,
                    active_stage_complete=False,
                ),
            )
            append_stage_entry(stage_state_file, partial_entry)
            audit_path = _append_persona_failure_audit(
                stage_state_file=stage_state_file,
                dataset=dataset,
                failed_stage=stage_name,
                last_completed_stage=last_completed_stage,
                items=items,
                per_item=per_item,
                exc=exc,
            )
            print(f"[staged-persona] failure audit appended to {audit_path}", file=summary_file)
            raise

        entry = make_stage_entry(
            stage_type="persona",
            completed_stage=stage_name,
            dataset=str(dataset),
            items=items_ser,
            persona_data=per_item,
            meta=_persona_stage_meta(
                persona_stage=persona_stage,
                n_personas=n_personas,
                persona_seed=persona_seed,
                axis_mode=axis_mode,
                backend=effective_backend,
                generator_model=generator_model,
                judge_generator_model=judge_generator_model,
                current_resume_settings=current_resume_settings,
                active_stage=stage_name,
                active_stage_complete=True,
            ),
        )
        append_stage_entry(stage_state_file, entry)
        print(f"[staged-persona] stage {stage_name} complete, state appended to {stage_state_file}", file=summary_file)

    return entry


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
    "persona_artifacts_from_rows",
    "persona_rows_from_stage_entry",
    "run_persona_generation",
    "run_persona_generation_staged",
]
