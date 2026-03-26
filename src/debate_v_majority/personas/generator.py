from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from ..engines import InferenceEngine, ensure_inference_results, inference_result_metadata
from .axes import build_axis_selection
from .prompt_templates import (
    ARTIFACT_VERSION,
    CARD_PROMPT_VERSION,
    DESCRIPTOR_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    build_stage1_messages,
    build_stage2_messages,
    parse_json_payload,
)
from .sampling import sample_axis_points, slot_role_for_index
from .schema import (
    Axis,
    AxisSelection,
    JudgeCard,
    PersonaArtifact,
    PersonaCard,
    PersonaDescriptor,
    PersonaGenerationConfig,
    ValidationResult,
    build_slot_layout,
)
from .validators import (
    COVERAGE_AUDIT_VERSION,
    card_coverage_audit,
    descriptor_coverage_audit,
    duplicate_diagnostics,
    semantic_redundancy_audit,
    validate_card,
    validate_descriptor,
    validate_descriptor_against_task,
)


MAX_GENERATION_RETRIES = 2
DESCRIPTOR_MAX_TOKENS = 12288
CARD_MAX_TOKENS = 12288
GENERATION_SETTINGS_VERSION = "phase1.persona_generation_settings.v1"
SEMANTIC_REDUNDANCY_VERSION = "phase0.semantic_redundancy.v1"
SLOT_SAMPLING_VERSION = "phase1.slot_sampling.v1"


class GenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, stage: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.metadata = metadata


def _question_media_for_task(*, dataset: str, raw_task: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if dataset != "hle" or raw_task is None:
        return None
    from ..datasets import hle as hle_dataset

    media = hle_dataset._image_part_specs_with_fallback(raw_task)
    return media or None


def _axis_bucket(value: float) -> str:
    if value <= 0.33:
        return "low"
    if value >= 0.67:
        return "high"
    return "balanced"


def _strip_sentence(text: str) -> str:
    return str(text).strip().rstrip(".!?")


def _lower_sentence_start(text: str) -> str:
    text = str(text).strip()
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _axis_interpretation(axis: Axis, value: float) -> str:
    bucket = _axis_bucket(value)
    if bucket == "low":
        return axis.low_desc
    if bucket == "high":
        return axis.high_desc
    if value < 0.5:
        primary = _strip_sentence(axis.low_desc)
        secondary = _lower_sentence_start(_strip_sentence(axis.high_desc))
    else:
        primary = _strip_sentence(axis.high_desc)
        secondary = _lower_sentence_start(_strip_sentence(axis.low_desc))
    return f"blend two moves: {primary}; also {secondary}."


def _persona_name(idx: int, point: dict[str, float]) -> str:
    if not point:
        return f"Persona {idx + 1}"
    ranked = sorted(sorted(point), key=lambda k: abs(float(point[k]) - 0.5), reverse=True)
    dominant_key = ranked[0]
    dominant_bucket = "High" if float(point[dominant_key]) >= 0.5 else "Low"
    base = dominant_key.replace("_vs_", "/").replace("_", " ")
    if len(ranked) >= 2:
        secondary_key = ranked[1]
        secondary_bucket = "H" if float(point[secondary_key]) >= 0.5 else "L"
        secondary_short = secondary_key.split("_vs_")[0].replace("_", " ").title()
        return f"{dominant_bucket} {base.title()} + {secondary_bucket}-{secondary_short} P{idx + 1}"
    return f"{dominant_bucket} {base.title()} P{idx + 1}"


def build_descriptor_messages(
    *,
    config: PersonaGenerationConfig,
    axis_selection: AxisSelection,
    points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    question_media = _question_media_for_task(dataset=config.dataset, raw_task=config.raw_task)
    return build_stage1_messages(
        dataset=config.dataset,
        benchmark_family=axis_selection.benchmark_family or config.dataset,
        question=config.question,
        axes=_descriptor_prompt_axes(axis_selection),
        sampled_points=points,
        question_media=question_media,
    )


def parse_descriptor_result(
    result_text: str,
    *,
    n_personas: int,
    points: list[dict[str, float]],
) -> list[PersonaDescriptor]:
    parse_error: str | None = None
    try:
        payload = parse_json_payload(result_text)
        rows = payload.get("descriptors") or []
    except ValueError as exc:
        parse_error = str(exc)
        rows = _recover_descriptor_rows(result_text)
    if len(rows) < n_personas:
        if parse_error is not None:
            raise ValueError(parse_error)
        raise ValueError(f"Stage-1 descriptor generation returned too few rows: {rows}")
    descriptors: list[PersonaDescriptor] = []
    for idx, row in enumerate(rows[:n_personas]):
        point = points[idx]
        descriptors.append(
            PersonaDescriptor(
                persona_id=str(row.get("persona_id") or f"persona_{idx + 1}"),
                name=str(row.get("name") or _persona_name(idx, point)),
                axis_values={k: float(v) for k, v in point.items()},
                axis_interpretation=_coerce_str_dict(row.get("axis_interpretation")),
                short_rule=str(
                    row.get("short_rule")
                    or row.get("question_approach_summary")
                    or "reason independently and verify the final answer"
                ),
                reasoning_summary=str(
                    row.get("reasoning_summary")
                    or row.get("disagreement_profile")
                    or "apply a distinct operational reasoning policy"
                ),
                question_approach_summary=str(
                    row.get("question_approach_summary")
                    or row.get("short_rule")
                    or ""
                ),
                disagreement_profile=str(
                    row.get("disagreement_profile")
                    or row.get("reasoning_summary")
                    or ""
                ),
                revision_profile=str(
                    row.get("revision_profile")
                    or row.get("revision_policy")
                    or ""
                ),
                solver_role=str(row.get("solver_role") or slot_role_for_index(idx)),
                round1_solver_profile=_coerce_str_dict(row.get("round1_solver_profile")),
                debate_temperament_profile=_coerce_str_dict(row.get("debate_temperament_profile")),
                likely_failure_mode=str(row.get("likely_failure_mode") or row.get("failure_mode_to_watch") or ""),
                revision_policy=str(row.get("revision_policy") or ""),
                confidence_policy=str(row.get("confidence_policy") or ""),
                failure_mode_to_watch=str(row.get("failure_mode_to_watch") or row.get("likely_failure_mode") or ""),
                stage_policy=row.get("stage_policy") or {},
            )
        )
    return descriptors


def _recover_descriptor_rows(result_text: str) -> list[dict[str, Any]]:
    text = str(result_text or "")
    anchor = text.find('"descriptors"')
    if anchor == -1:
        return []
    array_start = text.find("[", anchor)
    if array_start == -1:
        return []

    rows: list[dict[str, Any]] = []
    in_string = False
    escape = False
    depth = 0
    obj_start: int | None = None

    for idx in range(array_start + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and obj_start is not None:
                chunk = text[obj_start : idx + 1]
                try:
                    row = parse_json_payload(chunk)
                except ValueError:
                    obj_start = None
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                obj_start = None
            continue
        if ch == "]" and depth == 0:
            break

    return rows


def _effective_backend(*, config: PersonaGenerationConfig, engine: InferenceEngine | None) -> str:
    _ = config
    if engine is None:
        raise ValueError("persona generation requires a generator engine")
    return "llm"


def _effective_persona_seed(
    *,
    config: PersonaGenerationConfig | None = None,
    persona_seed: int | None = None,
    item_uid: str | None = None,
) -> int:
    if config is not None:
        base_seed = int(config.persona_seed)
        item_uid = str(config.item_uid)
    else:
        if persona_seed is None or item_uid is None:
            raise ValueError("effective persona seed requires either config or both persona_seed and item_uid")
        base_seed = int(persona_seed)
        item_uid = str(item_uid)
    payload = f"{base_seed}:{item_uid}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _descriptor_prompt_axis(axis: Axis) -> dict[str, Any]:
    axis_data = {
        "axis_id": axis.axis_id,
        "name": axis.name,
        "kind": axis.kind,
        "axis_role": axis.axis_role,
        "canonical_dimension": axis.canonical_dimension,
        "family_scope": axis.family_scope,
        "stage_affinity": axis.stage_affinity,
    }
    if axis.kind == "fixed":
        return {
            **axis_data,
            "low_desc": axis.low_desc,
            "high_desc": axis.high_desc,
            "notes": axis.notes,
        }

    return {
        **axis_data,
        "low_desc": f"Low end of the '{axis.name}' reasoning preference.",
        "high_desc": f"High end of the '{axis.name}' reasoning preference.",
        "notes": (
            "Interpret this axis abstractly from its name only. "
            "Do not reuse task-specific objects, formulas, constants, or answer conditions."
        ),
    }


def _descriptor_prompt_axes(axis_selection: AxisSelection) -> list[dict[str, Any]]:
    return [_descriptor_prompt_axis(axis) for axis in axis_selection.axes]


def _descriptor_context_texts(axis_selection: AxisSelection) -> list[str]:
    _ = axis_selection
    return []


def _coerce_str_dict(value: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (value or {}).items()}


def _descriptor_semantic_text(descriptor: PersonaDescriptor) -> str:
    parts = [
        descriptor.name,
        descriptor.solver_role,
        descriptor.question_approach_summary,
        descriptor.disagreement_profile,
        descriptor.revision_profile,
        descriptor.short_rule,
        descriptor.reasoning_summary,
        " ".join(str(v) for v in descriptor.round1_solver_profile.values()),
        " ".join(str(v) for v in descriptor.debate_temperament_profile.values()),
        descriptor.likely_failure_mode,
        " ".join(str(v) for v in descriptor.axis_interpretation.values()),
    ]
    return " ".join(part for part in parts if part)


def _card_semantic_text(card: PersonaCard) -> str:
    parts = [
        card.title,
        card.base_identity,
        card.core_reasoning_strategy,
        " ".join(card.priorities),
        " ".join(card.distrusts),
        card.decomposition_style,
        card.revision_policy,
        card.confidence_policy,
        card.failure_mode_to_avoid,
        " ".join(card.round1_solver_policy.values()),
        " ".join(card.round2_critique_policy.values()),
        " ".join(card.round3_revision_policy.values()),
    ]
    return " ".join(part for part in parts if part)


def _axis_bank_version(axis_selection: AxisSelection) -> int | None:
    for axis in axis_selection.axes:
        source = axis.source
        bank_version = source.get("bank_version")
        if isinstance(bank_version, int):
            return bank_version
    return None


def _axis_role_counts(axis_selection: AxisSelection) -> dict[str, int]:
    solver_axis_count = 0
    debate_axis_count = 0
    generated_solver_axis_count = 0
    for axis in axis_selection.axes:
        if axis.axis_role == "debate":
            debate_axis_count += 1
        else:
            solver_axis_count += 1
            if axis.kind == "task":
                generated_solver_axis_count += 1
    return {
        "solver_axis_count": solver_axis_count,
        "debate_axis_count": debate_axis_count,
        "generated_solver_axis_count": generated_solver_axis_count,
    }


def _validate_descriptor_for_generation(
    descriptor: PersonaDescriptor,
    *,
    config: PersonaGenerationConfig,
    axis_selection: AxisSelection,
    backend: str,
) -> ValidationResult:
    base = validate_descriptor(descriptor)
    if base.status != "accept":
        return base
    _ = backend
    return validate_descriptor_against_task(
        descriptor,
        question=config.question,
        raw_task=config.raw_task,
        context_texts=_descriptor_context_texts(axis_selection),
    )


def prepare_descriptor_generation(
    *,
    config: PersonaGenerationConfig,
    engine: InferenceEngine | None = None,
) -> tuple[AxisSelection, list[dict[str, float]], str]:
    backend = _effective_backend(config=config, engine=engine)
    effective_seed = _effective_persona_seed(config=config)
    axis_selection = build_axis_selection(
        mode=config.axis_mode,
        question=config.question,
        dataset=config.dataset,
        raw_task=config.raw_task,
        fixed_count=config.fixed_axis_count,
        task_count=config.task_axis_count,
        generator_model=config.generator_model,
        engine=engine,
        backend=backend,
        axes_file=config.axes_file,
    )
    points = sample_axis_points(
        axes=axis_selection.axes,
        num_personas=config.n_personas,
        seed=effective_seed,
        method=config.sampling_method,
        benchmark_family=axis_selection.benchmark_family,
    )
    return axis_selection, points, backend


def _sum_token_counts(rows: list[dict[str, Any] | None]) -> dict[str, int]:
    valid = [row for row in rows if row is not None]
    return {
        "n_calls": len(valid),
        "input_tokens": sum(int((row.get("token_counts") or {}).get("input_tokens") or 0) for row in valid),
        "output_tokens": sum(int((row.get("token_counts") or {}).get("output_tokens") or 0) for row in valid),
        "total_tokens": sum(int((row.get("token_counts") or {}).get("total_tokens") or 0) for row in valid),
    }


def generate_descriptors(
    *,
    config: PersonaGenerationConfig,
    engine: InferenceEngine | None = None,
) -> tuple[list[PersonaDescriptor], dict[str, Any]]:
    axis_selection, points, _backend = prepare_descriptor_generation(
        config=config,
        engine=engine,
    )
    return generate_descriptors_from_state(
        config=config,
        axis_selection=axis_selection,
        points=points,
        engine=engine,
    )


def _build_descriptor_retry_feedback(
    *,
    parse_error: str | None,
    validator_rows: list[dict[str, Any]],
    duplicates: list[dict[str, float | int]],
    semantic_redundancy: dict[str, Any] | None,
    pending_ids: list[str] | None = None,
) -> str:
    parts = ["Your previous response was not accepted. Issues found:"]
    if parse_error:
        parts.append(f"- JSON parse failure: {parse_error[:300]}")
    for row in validator_rows:
        if row.get("status") in ("retry", "reject_hard"):
            pid = row.get("persona_id") or "unknown"
            reasons = ", ".join(str(r) for r in row.get("reasons", []))
            parts.append(f"- {pid}: {reasons}")
    if duplicates:
        for dupe in duplicates:
            parts.append(
                f"- personas {dupe['left']} and {dupe['right']} are too similar "
                f"(Jaccard={dupe['similarity']})"
            )
    if semantic_redundancy and semantic_redundancy.get("pairs"):
        for pair in list(semantic_redundancy.get("pairs", []))[:6]:
            shared_terms = ", ".join(str(term) for term in pair.get("shared_terms", [])[:6])
            shared_phrases = ", ".join(str(phrase) for phrase in pair.get("shared_bigrams", [])[:3])
            parts.append(
                f"- personas {pair.get('left')} and {pair.get('right')} still overlap semantically "
                f"(similarity={pair.get('similarity')}, shared terms={shared_terms or 'n/a'}, "
                f"shared phrases={shared_phrases or 'n/a'})"
            )
    parts.append("")
    if pending_ids:
        parts.append(f"Regenerate only these pending descriptors: {', '.join(pending_ids)}")
    else:
        parts.append("Regenerate only the pending descriptors.")
    parts.append("Requirements:")
    parts.append("- Describe a question-tailored reasoning POLICY only (search order, verification timing, "
                 "pruning strategy, revision triggers, evidence standards)")
    parts.append("- Tailor the approach to the prompt, but do not include answer hints, option hints, hidden cruxes, or solve recipes")
    parts.append("- Each persona must be operationally distinct from every other")
    parts.append("- Return ONLY a valid JSON object—no prose, no markdown fencing, no explanation")
    return "\n".join(parts)


def _build_locked_descriptor_context(
    *,
    accepted_descriptors: list[PersonaDescriptor | None],
    pending_indices: list[int],
) -> list[dict[str, str]]:
    locked_descriptors = [descriptor for descriptor in accepted_descriptors if descriptor is not None]
    if not locked_descriptors:
        return []
    target_ids = [f"persona_{idx + 1}" for idx in pending_indices]
    locked_lines = []
    for descriptor in locked_descriptors:
        approach = descriptor.question_approach_summary or descriptor.short_rule
        failure = descriptor.likely_failure_mode or descriptor.failure_mode_to_watch
        locked_lines.append(
            f"- {descriptor.persona_id}: {descriptor.name} | role={descriptor.solver_role} | "
            f"approach={_strip_sentence(approach)} | "
            f"failure={_strip_sentence(failure) or 'n/a'}"
        )
    return [
        {
            "role": "user",
            "content": (
                "The following descriptors from this population were already accepted and are locked. "
                "Do not regenerate them or drift toward them. Generate only the remaining personas listed below, "
                "and make them complementary rather than redundant.\n\n"
                f"Locked descriptors:\n{chr(10).join(locked_lines)}\n\n"
                f"Generate descriptors only for these persona_ids, in this order:\n"
                f"{json.dumps(target_ids, ensure_ascii=False)}"
            ),
        }
    ]


def generate_descriptors_from_state(
    *,
    config: PersonaGenerationConfig,
    axis_selection: AxisSelection,
    points: list[dict[str, float]],
    engine: InferenceEngine | None = None,
) -> tuple[list[PersonaDescriptor], dict[str, Any]]:
    last_meta: dict[str, Any] | None = None
    backend = _effective_backend(config=config, engine=engine)
    attempt_audits: list[dict[str, Any]] = []
    retry_context: list[dict[str, Any]] | None = None
    active_points = list(points)
    accepted_descriptors: list[PersonaDescriptor | None] = [None] * config.n_personas
    pending_indices = list(range(config.n_personas))
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        descriptor_messages: list[dict[str, Any]] | None = None
        raw_result_text: str | None = None
        parse_error: str | None = None
        target_indices = list(pending_indices)
        target_points = [active_points[idx] for idx in target_indices]
        base_messages = build_descriptor_messages(
            config=config,
            axis_selection=axis_selection,
            points=target_points,
        )
        descriptor_messages = list(base_messages)
        descriptor_messages += _build_locked_descriptor_context(
            accepted_descriptors=accepted_descriptors,
            pending_indices=target_indices,
        )
        if retry_context is not None:
            descriptor_messages += retry_context
        result = ensure_inference_results(
            engine,
            [descriptor_messages],
            batch_size=1,
            sampling_kwargs={"max_tokens": DESCRIPTOR_MAX_TOKENS},
            model_role="generator",
        )[0]
        raw_result_text = str(result.text)
        descriptor_call_meta = inference_result_metadata(result)
        try:
            descriptors = parse_descriptor_result(
                raw_result_text,
                n_personas=len(target_points),
                points=target_points,
            )
        except ValueError as exc:
            parse_error = str(exc)
            descriptors = []
        else:
            remapped: list[PersonaDescriptor] = []
            for local_idx, descriptor in enumerate(descriptors):
                global_idx = target_indices[local_idx]
                remapped.append(replace(descriptor, persona_id=f"persona_{global_idx + 1}"))
            descriptors = remapped
        validator_rows: list[dict[str, Any]] = []
        has_retry = parse_error is not None
        retry_indices: set[int] = set(target_indices if parse_error is not None else [])
        if parse_error is not None:
            validator_rows.append(
                {
                    "persona_id": None,
                    "kind": "descriptor_parse",
                    "status": "retry",
                    "reasons": [parse_error],
                    "attempt": attempt,
                }
            )
        for descriptor in descriptors:
            global_idx = int(descriptor.persona_id.split("_")[-1]) - 1
            validation = _validate_descriptor_for_generation(
                descriptor,
                config=config,
                axis_selection=axis_selection,
                backend=backend,
            )
            validator_rows.append(
                {
                    "persona_id": descriptor.persona_id,
                    "kind": "descriptor",
                    "status": validation.status,
                    "reasons": list(validation.reasons),
                    "attempt": attempt,
                }
            )
            if validation.status == "reject_hard":
                has_retry = True
                retry_indices.add(global_idx)
            if validation.status == "retry":
                has_retry = True
                retry_indices.add(global_idx)
        candidate_descriptors = list(accepted_descriptors)
        for descriptor in descriptors:
            global_idx = int(descriptor.persona_id.split("_")[-1]) - 1
            if global_idx not in retry_indices:
                candidate_descriptors[global_idx] = descriptor
        complete_descriptors = [descriptor for descriptor in candidate_descriptors if descriptor is not None]
        descriptor_semantic_audit = semantic_redundancy_audit(
            [_descriptor_semantic_text(descriptor) for descriptor in complete_descriptors],
            label="descriptor",
        )
        coverage_audit = descriptor_coverage_audit(complete_descriptors)
        dupes = duplicate_diagnostics(d.reasoning_summary for d in complete_descriptors)
        index_map = [idx for idx, descriptor in enumerate(candidate_descriptors) if descriptor is not None]
        resample_indices: set[int] = set()
        if not has_retry and len(complete_descriptors) == config.n_personas:
            for row in dupes:
                idx = index_map[int(row["right"])]
                retry_indices.add(idx)
                resample_indices.add(idx)
            for pair in descriptor_semantic_audit.get("pairs", []):
                idx = index_map[int(pair.get("right") or 0)]
                retry_indices.add(idx)
                resample_indices.add(idx)
            for idx in coverage_audit.get("redundant_indices", []):
                mapped_idx = index_map[int(idx)]
                retry_indices.add(mapped_idx)
                resample_indices.add(mapped_idx)
        if descriptor_semantic_audit.get("pairs") or coverage_audit.get("status") == "retry":
            has_retry = True
        validator_meta = {
            "descriptor_validations": validator_rows,
            "descriptor_duplicates": dupes,
            "descriptor_semantic_redundancy": descriptor_semantic_audit,
            "descriptor_coverage_audit": coverage_audit,
            "descriptor_prompt_version": DESCRIPTOR_PROMPT_VERSION,
            "descriptor_semantic_redundancy_version": SEMANTIC_REDUNDANCY_VERSION,
            "descriptor_coverage_audit_version": COVERAGE_AUDIT_VERSION,
            "descriptor_backend": backend,
            "descriptor_call_metadata": descriptor_call_meta,
            "descriptor_parse_error": parse_error,
        }
        audit_validator_meta = dict(validator_meta)
        attempt_audits.append(
            {
                "attempt": attempt,
                "backend": backend,
                "request_messages": descriptor_messages,
                "raw_result_text": raw_result_text,
                "parse_error": parse_error,
                "descriptors": [asdict(descriptor) for descriptor in descriptors],
                "accepted_descriptor_ids": [
                    descriptor.persona_id for descriptor in candidate_descriptors if descriptor is not None
                ],
                "pending_descriptor_ids": [f"persona_{idx + 1}" for idx in sorted(retry_indices)],
                "validator_metadata": audit_validator_meta,
            }
        )
        validator_meta["attempt_audits"] = [dict(audit) for audit in attempt_audits]
        last_meta = {
            "axis_selection": axis_selection,
            "sampled_points": active_points,
            "validator_metadata": validator_meta,
        }
        if not has_retry and not dupes and len(complete_descriptors) == config.n_personas:
            return complete_descriptors, last_meta
        accepted_descriptors = list(candidate_descriptors)
        for idx in retry_indices:
            accepted_descriptors[idx] = None
        pending_indices = sorted(retry_indices)
        retry_context = [
            {
                "role": "user",
                "content": _build_descriptor_retry_feedback(
                    parse_error=parse_error,
                    validator_rows=validator_rows,
                    duplicates=dupes,
                    semantic_redundancy=descriptor_semantic_audit,
                    pending_ids=[f"persona_{idx + 1}" for idx in pending_indices],
                ),
            },
        ]
        redundant_indices = sorted(resample_indices)
        if redundant_indices and attempt < MAX_GENERATION_RETRIES:
            refreshed_points = sample_axis_points(
                axes=axis_selection.axes,
                num_personas=config.n_personas,
                seed=_effective_persona_seed(config=config) + attempt + 1,
                method=config.sampling_method,
                benchmark_family=axis_selection.benchmark_family,
            )
            for idx in redundant_indices:
                if 0 <= idx < len(active_points):
                    active_points[idx] = refreshed_points[idx]
    assert last_meta is not None
    failure_metadata = {
        **last_meta,
        "validator_metadata": {
            **last_meta["validator_metadata"],
            "attempt_audits": [dict(audit) for audit in attempt_audits],
        },
    }
    raise GenerationExhaustedError(
        f"Descriptor generation exhausted retries: {failure_metadata['validator_metadata']}",
        stage="descriptors",
        metadata=failure_metadata,
    )


def build_card_messages(
    *,
    descriptor: PersonaDescriptor,
    question: str,
    question_media: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return build_stage2_messages(
        question=question,
        descriptor=asdict(descriptor),
        question_media=question_media,
    )


def parse_card_result(
    result_text: str,
    *,
    descriptor: PersonaDescriptor,
) -> PersonaCard:
    payload = parse_json_payload(result_text)
    round2_critique_policy = _coerce_str_dict(payload.get("round2_critique_policy"))
    round3_revision_policy = _coerce_str_dict(payload.get("round3_revision_policy"))
    decomposition_style = str(
        payload.get("decomposition_style")
        or round2_critique_policy.get("primary_attack_rule")
        or "Attack the earliest unsupported step a peer answer depends on."
    )
    revision_policy = str(
        payload.get("revision_policy")
        or round3_revision_policy.get("switch_triggers")
        or round3_revision_policy.get("default_stance")
        or "Defend unless a critique exposes a contradiction, missing necessary case, or broken assumption; switch only if the conclusion no longer holds."
    )
    confidence_policy = str(
        payload.get("confidence_policy")
        or "Keep confidence proportional to support and unresolved objections."
    )
    failure_mode_to_avoid = str(
        payload.get("failure_mode_to_avoid")
        or "Do not confuse a plausible path with a justified one."
    )
    system_prompt = str(
        payload.get("system_prompt")
        or (
            f"Reason as {payload.get('title') or descriptor.name}. "
            f"Round 1 solve policy: {str(payload.get('core_reasoning_strategy') or descriptor.question_approach_summary or descriptor.reasoning_summary)} "
            f"Round 2 critique policy: {decomposition_style} "
            f"Round 3 revision policy: {revision_policy} "
            f"Confidence rule: {confidence_policy} "
            f"Failure mode to avoid: {failure_mode_to_avoid}"
        )
    )
    return PersonaCard(
        persona_id=descriptor.persona_id,
        title=str(payload.get("title") or descriptor.name),
        core_reasoning_strategy=str(
            payload.get("core_reasoning_strategy")
            or descriptor.question_approach_summary
            or descriptor.reasoning_summary
        ),
        priorities=[str(x) for x in payload.get("priorities", [])],
        distrusts=[str(x) for x in payload.get("distrusts", [])],
        decomposition_style=decomposition_style,
        revision_policy=revision_policy,
        confidence_policy=confidence_policy,
        failure_mode_to_avoid=failure_mode_to_avoid,
        system_prompt=system_prompt,
        card_version=CARD_PROMPT_VERSION,
        base_identity=str(payload.get("base_identity") or payload.get("title") or descriptor.name),
        round1_solver_policy=_coerce_str_dict(payload.get("round1_solver_policy")),
        round2_critique_policy=round2_critique_policy,
        round3_revision_policy=round3_revision_policy,
        runtime_prompts=_coerce_str_dict(payload.get("runtime_prompts")),
        stage_policy=payload.get("stage_policy") or {},
    )


def _build_card_retry_feedback(
    *,
    parse_error: str | None,
    validation: ValidationResult | dict[str, Any] | None,
    semantic_redundancy: dict[str, Any] | None = None,
) -> str:
    parts = ["Your previous card response was not accepted. Issues found:"]
    if parse_error:
        parts.append(f"- JSON parse failure: {parse_error[:300]}")
    if isinstance(validation, dict):
        status = validation.get("status")
        reasons = validation.get("reasons", [])
    elif validation is not None:
        status = validation.status
        reasons = validation.reasons
    else:
        status = None
        reasons = []
    if status in ("retry", "reject_hard"):
        parts.append(f"- Validation: {', '.join(str(reason) for reason in reasons)}")
        joined_reasons = " ".join(str(reason) for reason in reasons).lower()
        if "contains answer-oriented leakage indicators" in joined_reasons:
            parts.append(
                "- Remove any phrase that states, predicts, labels, or eliminates a final answer."
            )
            parts.append(
                "- Never write `the answer is`, `answer:`, `likely answer`, `correct answer`, `option A-E`, or `rule out option`."
            )
        if "critique_policy is too generic" in joined_reasons:
            parts.append(
                "- `decomposition_style` must be a concrete Round-2 attack rule aimed at the opponent's reasoning, not a general solve style."
            )
            parts.append(
                "- Name a specific weakness to probe or failure pattern to expose, e.g. earliest unsupported step, missing necessary case, dropped constraint, or unjustified leap."
            )
        if "revision_policy must name a concrete switch trigger" in joined_reasons:
            parts.append(
                "- `revision_policy` must include an explicit trigger for defend vs revise vs switch using if/when/unless language."
            )
            parts.append(
                "- Tie the trigger to evidence such as contradiction, missing necessary case, broken assumption, or direct counterexample."
            )
    if semantic_redundancy and semantic_redundancy.get("pairs"):
        for pair in list(semantic_redundancy.get("pairs", []))[:4]:
            shared_terms = ", ".join(str(term) for term in pair.get("shared_terms", [])[:6])
            parts.append(
                f"- This card still overlaps semantically with another persona "
                f"(shared terms: {shared_terms or 'n/a'}). Make the solve policy, critique policy, and "
                f"revision trigger materially different."
            )
    parts.append("")
    parts.append("Regenerate this card. Requirements:")
    parts.append("- Return ONLY a valid JSON object matching the requested schema")
    parts.append("- No markdown fencing, no prose, no explanation")
    parts.append("- Keep the card compact and operational")
    parts.append("- Focus on reasoning policy, not biography or style")
    parts.append("- `core_reasoning_strategy` is round-1 solve behavior")
    parts.append("- `decomposition_style` is round-2 critique behavior only")
    parts.append("- `revision_policy` is round-3 defend/revise/switch behavior only")
    return "\n".join(parts)


def _build_locked_card_context(
    *,
    accepted_cards: list[PersonaCard | None],
    pending_indices: list[int],
) -> list[dict[str, str]]:
    locked_cards = [card for card in accepted_cards if card is not None]
    if not locked_cards:
        return []
    locked_lines = []
    for card in locked_cards:
        opening = card.round1_solver_policy.get("opening_strategy") or card.core_reasoning_strategy
        critique = card.round2_critique_policy.get("primary_attack_rule") or card.decomposition_style
        revision = card.round3_revision_policy.get("switch_triggers") or card.revision_policy
        locked_lines.append(
            f"- {card.title} | "
            f"solve={_strip_sentence(opening)} | "
            f"critique={_strip_sentence(critique)} | "
            f"revise={_strip_sentence(revision)}"
        )
    return [
        {
            "role": "user",
            "content": (
                "The following cards from this population were already accepted and are locked. "
                "Do not regenerate them or drift toward them. Generate only the card for the descriptor above, "
                "and keep its solve policy, critique policy, and revision triggers materially distinct.\n\n"
                f"Locked cards:\n{chr(10).join(locked_lines)}"
            ),
        }
    ]


def _llm_card(
    *,
    descriptor: PersonaDescriptor,
    question: str,
    question_media: list[dict[str, Any]] | None,
    engine: InferenceEngine,
) -> tuple[PersonaCard, dict[str, Any]]:
    messages = build_card_messages(
        descriptor=descriptor, question=question, question_media=question_media,
    )
    result = ensure_inference_results(
        engine,
        [messages],
        batch_size=1,
        sampling_kwargs={"max_tokens": CARD_MAX_TOKENS},
        model_role="generator",
    )[0]
    card = parse_card_result(str(result.text), descriptor=descriptor)
    return card, inference_result_metadata(result)


def _regenerate_duplicate_cards(
    *,
    cards: list[PersonaCard],
    descriptors: list[PersonaDescriptor],
    duplicate_rows: list[dict[str, float | int]],
    dataset: str,
    question: str,
    raw_task: dict[str, Any],
    engine: InferenceEngine,
) -> list[PersonaCard]:
    """Re-run card generation for duplicates with aligned descriptor/card indices."""
    updated = list(cards)
    regenerated_indices: set[int] = set()
    question_media = _question_media_for_task(dataset=dataset, raw_task=raw_task)
    for row in duplicate_rows:
        regenerated_indices.add(int(row["right"]))
    for idx in regenerated_indices:
        descriptor = descriptors[idx]
        card, _ = _llm_card(
            descriptor=descriptor,
            question=question,
            question_media=question_media,
            engine=engine,
        )
        updated[idx] = card
    return updated


def expand_cards(
    descriptors: list[PersonaDescriptor],
    *,
    dataset: str = "unknown",
    question: str,
    raw_task: dict[str, Any] | None = None,
    engine: InferenceEngine | None = None,
    backend: str = "llm",
) -> tuple[list[PersonaCard], dict[str, Any]]:
    last_meta: dict[str, Any] | None = None
    question_media = _question_media_for_task(dataset=dataset, raw_task=raw_task)
    if engine is None:
        raise ValueError("persona generation requires a generator engine for card generation")
    attempt_audits: list[dict[str, Any]] = []
    accepted_cards: list[PersonaCard | None] = [None] * len(descriptors)
    pending_indices = list(range(len(descriptors)))
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        validator_rows: list[dict[str, Any]] = []
        call_metadata: list[dict[str, Any] | None] = []
        has_retry = False
        current_attempt_rows: list[dict[str, Any]] = []
        target_indices = list(pending_indices)
        for descriptor_idx in target_indices:
            descriptor = descriptors[descriptor_idx]
            card_call_meta: dict[str, Any] | None = None
            raw_result_text: str | None = None
            parse_error: str | None = None
            request_messages: list[dict[str, Any]] | None = None
            validation: ValidationResult | None = None
            base_messages = build_card_messages(
                descriptor=descriptor,
                question=question,
                question_media=question_media,
            )
            extra_messages = _build_locked_card_context(
                accepted_cards=accepted_cards,
                pending_indices=target_indices,
            )
            if attempt > 0:
                prev = next(
                    (
                        audit for audit in reversed(attempt_audits)
                        if audit.get("persona_id") == descriptor.persona_id
                    ),
                    None,
                )
                extra_messages = extra_messages + [
                    {
                        "role": "user",
                        "content": _build_card_retry_feedback(
                            parse_error=None if prev is None else prev.get("parse_error"),
                            validation=None if prev is None else prev.get("validation"),
                            semantic_redundancy=None if prev is None else prev.get("semantic_redundancy"),
                        ),
                    }
                ]
            request_messages = [base_messages[0], *extra_messages, *base_messages[1:]]
            result = ensure_inference_results(
                engine,
                [request_messages],
                batch_size=1,
                sampling_kwargs={"max_tokens": CARD_MAX_TOKENS},
                model_role="generator",
            )[0]
            raw_result_text = str(result.text)
            card_call_meta = inference_result_metadata(result)
            try:
                card = parse_card_result(raw_result_text, descriptor=descriptor)
            except ValueError as exc:
                parse_error = str(exc)
                card = None
            if parse_error is not None:
                validation = ValidationResult(status="retry", reasons=[parse_error])
                validator_rows.append(
                    {
                        "persona_id": descriptor.persona_id,
                        "kind": "card_parse",
                        "status": "retry",
                        "reasons": [parse_error],
                        "attempt": attempt,
                    }
                )
                has_retry = True
            else:
                assert card is not None
                validation = validate_card(card)
                validator_rows.append(
                    {
                        "persona_id": descriptor.persona_id,
                        "kind": "card",
                        "status": validation.status,
                        "reasons": list(validation.reasons),
                        "attempt": attempt,
                    }
                )
                if validation.status in ("retry", "reject_hard"):
                    has_retry = True
                if validation.status == "accept":
                    accepted_cards[descriptor_idx] = card
            call_metadata.append(card_call_meta)
            current_attempt_rows.append(
                {
                    "attempt": attempt,
                    "persona_id": descriptor.persona_id,
                    "descriptor_index": descriptor_idx,
                    "request_messages": request_messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": parse_error,
                    "validation": None if validation is None else asdict(validation),
                    "card": None if parse_error is not None or card is None else asdict(card),
                }
            )
        cards = [card for card in accepted_cards if card is not None]
        retry_indices: set[int] = set()
        for row in validator_rows:
            if row.get("status") in ("retry", "reject_hard"):
                persona_id = str(row.get("persona_id") or "")
                if persona_id.startswith("persona_"):
                    try:
                        retry_indices.add(int(persona_id.split("_")[1]) - 1)
                    except ValueError:
                        pass
        accepted_indices = [idx for idx, card in enumerate(accepted_cards) if card is not None]
        exact_dupes: list[dict[str, float | int]] = []
        if not has_retry:
            exact_dupes = duplicate_diagnostics(card.system_prompt for card in cards)
        if exact_dupes:
            regenerated_cards = _regenerate_duplicate_cards(
                cards=cards,
                descriptors=[descriptors[idx] for idx in accepted_indices],
                duplicate_rows=exact_dupes,
                dataset=dataset,
                question=question,
                raw_task=raw_task,
                engine=engine,
            )
            for local_idx, card in enumerate(regenerated_cards):
                accepted_cards[accepted_indices[local_idx]] = card
            cards = [card for card in accepted_cards if card is not None]
            exact_dupes = duplicate_diagnostics(card.system_prompt for card in cards)
            if exact_dupes:
                retry_indices.update(accepted_indices[int(dupe["right"])] for dupe in exact_dupes)
                has_retry = True
        card_semantic_audit = semantic_redundancy_audit(
            [_card_semantic_text(card) for card in cards],
            label="card",
        )
        coverage_audit = card_coverage_audit(cards)
        semantic_pairs = list(card_semantic_audit.get("pairs", []))
        if semantic_pairs:
            regenerated_cards = _regenerate_duplicate_cards(
                cards=cards,
                descriptors=[descriptors[idx] for idx in accepted_indices],
                duplicate_rows=semantic_pairs,
                dataset=dataset,
                question=question,
                raw_task=raw_task,
                engine=engine,
            )
            for local_idx, card in enumerate(regenerated_cards):
                accepted_cards[accepted_indices[local_idx]] = card
            cards = [card for card in accepted_cards if card is not None]
            card_semantic_audit = semantic_redundancy_audit(
                [_card_semantic_text(card) for card in cards],
                label="card",
            )
            semantic_pairs = list(card_semantic_audit.get("pairs", []))
            if semantic_pairs:
                retry_indices.update(accepted_indices[int(pair.get("right") or 0)] for pair in semantic_pairs)
                has_retry = True
        if coverage_audit.get("redundant_indices"):
            regenerated_cards = _regenerate_duplicate_cards(
                cards=cards,
                descriptors=[descriptors[idx] for idx in accepted_indices],
                duplicate_rows=[{"right": idx} for idx in coverage_audit.get("redundant_indices", [])],
                dataset=dataset,
                question=question,
                raw_task=raw_task,
                engine=engine,
            )
            for local_idx, card in enumerate(regenerated_cards):
                accepted_cards[accepted_indices[local_idx]] = card
            cards = [card for card in accepted_cards if card is not None]
            coverage_audit = card_coverage_audit(cards)
            if coverage_audit.get("status") == "retry":
                retry_indices.update(
                    accepted_indices[int(idx)] for idx in coverage_audit.get("redundant_indices", [])
                )
                has_retry = True
        for idx in retry_indices:
            if 0 <= idx < len(accepted_cards):
                accepted_cards[idx] = None
        semantic_pair_map: dict[int, list[dict[str, Any]]] = {}
        for pair in semantic_pairs:
            left = accepted_indices[int(pair.get("left") or 0)]
            right = accepted_indices[int(pair.get("right") or 0)]
            semantic_pair_map.setdefault(left, []).append(pair)
            semantic_pair_map.setdefault(right, []).append(pair)
        for row in current_attempt_rows:
            idx = int(row.get("descriptor_index") or 0)
            row["semantic_redundancy"] = {
                "label": "card",
                "threshold": card_semantic_audit.get("threshold"),
                "max_similarity": card_semantic_audit.get("max_similarity"),
                "status": "retry" if semantic_pair_map.get(idx) else "accept",
                "pairs": semantic_pair_map.get(idx, []),
            }
            row["coverage_audit"] = coverage_audit
            row["accepted_card_ids"] = [
                card.persona_id for card in accepted_cards if card is not None
            ]
            row["pending_card_ids"] = [f"persona_{idx + 1}" for idx in sorted(retry_indices)]
            attempt_audits.append(row)
        dupes: list[dict[str, float | int]] = []
        if exact_dupes:
            dupes = exact_dupes
        last_meta = {
            "card_validations": validator_rows,
            "card_duplicates": dupes,
            "card_semantic_redundancy": card_semantic_audit,
            "card_coverage_audit": coverage_audit,
            "card_prompt_version": CARD_PROMPT_VERSION,
            "card_semantic_redundancy_version": SEMANTIC_REDUNDANCY_VERSION,
            "card_coverage_audit_version": COVERAGE_AUDIT_VERSION,
            "card_backend": backend,
            "card_call_metadata": call_metadata,
            "card_attempt_audits": [dict(audit) for audit in attempt_audits],
        }
        if not has_retry and not dupes and not semantic_pairs:
            return [card for card in accepted_cards if card is not None], last_meta
        pending_indices = sorted(retry_indices)
    assert last_meta is not None
    raise GenerationExhaustedError(
        f"Card generation exhausted retries: {last_meta}",
        stage="cards",
        metadata=last_meta,
    )


def build_persona_artifact(
    *,
    config: PersonaGenerationConfig,
    judge_card: JudgeCard | None = None,
    generator_engine: InferenceEngine | None = None,
) -> PersonaArtifact:
    descriptor_output, descriptor_meta = generate_descriptors(config=config, engine=generator_engine)
    descriptors = descriptor_output
    axis_selection = descriptor_meta["axis_selection"]
    sampled_points = descriptor_meta["sampled_points"]
    backend = _effective_backend(config=config, engine=generator_engine)
    cards, card_meta = expand_cards(
        descriptors,
        dataset=config.dataset,
        question=config.question,
        raw_task=config.raw_task,
        engine=generator_engine,
        backend=backend,
    )
    validator_metadata = build_persona_validator_metadata(
        axis_selection=axis_selection,
        descriptor_validator_metadata=descriptor_meta["validator_metadata"],
        card_metadata=card_meta,
        backend=backend,
        judge_card=judge_card,
    )
    n_plain = int(config.n_plain_agents)
    n_total = config.n_personas + n_plain
    slot_layout = build_slot_layout(n_agents=n_total, n_plain_agents=n_plain) if n_plain > 0 else None
    effective_seed = _effective_persona_seed(config=config)
    axis_role_counts = _axis_role_counts(axis_selection)

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
        sampled_points=sampled_points,
        descriptors=descriptors,
        cards=cards,
        judge_card=judge_card,
        prompt_versions={
            "axis": axis_selection.generator_prompt_version,
            "descriptor": DESCRIPTOR_PROMPT_VERSION,
            "card": CARD_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
        generation_settings={
            "generation_settings_version": GENERATION_SETTINGS_VERSION,
            "n_personas": int(config.n_personas),
            "persona_seed": int(config.persona_seed),
            "effective_persona_seed": effective_seed,
            "axis_mode": str(config.axis_mode),
            "fixed_axis_count": int(config.fixed_axis_count),
            "task_axis_count": int(config.task_axis_count),
            "sampling_method": str(config.sampling_method),
            "slot_sampling_version": SLOT_SAMPLING_VERSION,
            "slot_role_scheme": "generic_coverage_v1",
            "population_design_version": "generic_persona_coverage.v1",
            "generator_model": config.generator_model,
            "judge_generator_model": config.judge_generator_model,
            "judge_persona_mode": str(config.judge_persona_mode),
            "backend": backend,
            "benchmark_family": axis_selection.benchmark_family,
            "axis_bank_version": _axis_bank_version(axis_selection),
            "semantic_redundancy_version": SEMANTIC_REDUNDANCY_VERSION,
            "coverage_audit_version": COVERAGE_AUDIT_VERSION,
            "generic_axis_bank_version": _axis_bank_version(axis_selection),
            "card_schema_version": CARD_PROMPT_VERSION,
            **axis_role_counts,
            "axes_file": None if config.axes_file is None else str(config.axes_file),
            "n_plain_agents": n_plain,
        },
        validator_metadata=validator_metadata,
        slot_layout=slot_layout,
    )


def build_persona_validator_metadata(
    *,
    axis_selection: AxisSelection,
    descriptor_validator_metadata: dict[str, Any] | None,
    card_metadata: dict[str, Any] | None,
    backend: str,
    judge_card: JudgeCard | None = None,
) -> dict[str, Any]:
    validator_metadata: dict[str, Any] = {}
    validator_metadata.update(dict(descriptor_validator_metadata or {}))
    validator_metadata.update(dict(card_metadata or {}))
    validator_metadata["generator_backend"] = backend
    validator_metadata["generation_settings_version"] = GENERATION_SETTINGS_VERSION
    validator_metadata["semantic_redundancy_version"] = SEMANTIC_REDUNDANCY_VERSION
    validator_metadata["coverage_audit_version"] = COVERAGE_AUDIT_VERSION
    validator_metadata["slot_sampling_version"] = SLOT_SAMPLING_VERSION
    validator_metadata["axis_bank_version"] = _axis_bank_version(axis_selection)
    if judge_card is not None:
        validator_metadata["judge_card"] = asdict(judge_card)
    axis_call_meta = None
    for axis in axis_selection.axes:
        axis_call_meta = axis.source.get("call_metadata")
        if axis_call_meta is not None:
            break
    judge_call_meta = None if judge_card is None else judge_card.source.get("call_metadata")
    validator_metadata["token_usage_summary"] = {
        "axis_generation": _sum_token_counts([axis_call_meta]),
        "descriptor_generation": _sum_token_counts(
            [dict(descriptor_validator_metadata or {}).get("descriptor_call_metadata")]
        ),
        "card_generation": _sum_token_counts(
            list(dict(card_metadata or {}).get("card_call_metadata") or [])
        ),
        "judge_generation": _sum_token_counts([judge_call_meta]),
    }
    return validator_metadata
