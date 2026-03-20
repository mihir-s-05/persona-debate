from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, cast

from ..engines import ensure_inference_results, inference_result_metadata
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
from .sampling import sample_axis_points
from .schema import (
    Axis,
    PersonaArtifact,
    PersonaCard,
    PersonaDescriptor,
    PersonaGenerationConfig,
    ValidationResult,
    build_slot_layout,
)
from .validators import duplicate_diagnostics, validate_card, validate_descriptor, validate_descriptor_against_task


MAX_GENERATION_RETRIES = 2
DESCRIPTOR_MAX_TOKENS = 8192
CARD_MAX_TOKENS = 8192


class GenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, stage: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.metadata = metadata

_LOW_STYLE = {
    "symbolic_vs_intuitive": "use explicit symbolic structure before relying on intuition",
    "exhaustive_vs_pruning": "enumerate cases carefully before pruning",
    "propose_first_vs_verify_first": "propose a candidate path early and test it quickly",
    "local_vs_global_focus": "start from local components and build upward",
    "mechanistic_vs_elimination_reasoning": "construct the answer from mechanism rather than elimination",
    "low_vs_high_skepticism_of_intermediates": "treat intermediate steps as tentatively usable unless contradicted",
}

_HIGH_STYLE = {
    "symbolic_vs_intuitive": "use intuitive structure and plausibility before formal derivation",
    "exhaustive_vs_pruning": "prune aggressively and focus on the highest-value paths first",
    "propose_first_vs_verify_first": "verify constraints before committing to a candidate path",
    "local_vs_global_focus": "start from global structure and derive local details",
    "mechanistic_vs_elimination_reasoning": "lean on elimination and contradiction checks before full construction",
    "low_vs_high_skepticism_of_intermediates": "aggressively audit intermediate steps for hidden assumptions",
}


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
        return _LOW_STYLE.get(axis.axis_id, axis.low_desc)
    if bucket == "high":
        return _HIGH_STYLE.get(axis.axis_id, axis.high_desc)
    if value < 0.5:
        primary = _strip_sentence(_LOW_STYLE.get(axis.axis_id, axis.low_desc))
        secondary = _lower_sentence_start(_strip_sentence(_HIGH_STYLE.get(axis.axis_id, axis.high_desc)))
    else:
        primary = _strip_sentence(_HIGH_STYLE.get(axis.axis_id, axis.high_desc))
        secondary = _lower_sentence_start(_strip_sentence(_LOW_STYLE.get(axis.axis_id, axis.low_desc)))
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
    axis_selection: Any,
    points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    return build_stage1_messages(
        dataset=config.dataset,
        benchmark_family=axis_selection.benchmark_family or config.dataset,
        axes=_descriptor_prompt_axes(axis_selection),
        sampled_points=points,
    )


def parse_descriptor_result(
    result_text: str,
    *,
    n_personas: int,
    points: list[dict[str, float]],
) -> list[PersonaDescriptor]:
    payload = parse_json_payload(result_text)
    rows = payload.get("descriptors") or []
    if len(rows) < n_personas:
        raise ValueError(f"Stage-1 descriptor generation returned too few rows: {rows}")
    descriptors: list[PersonaDescriptor] = []
    for idx, row in enumerate(rows[:n_personas]):
        point = points[idx]
        descriptors.append(
            PersonaDescriptor(
                persona_id=str(row.get("persona_id") or f"persona_{idx + 1}"),
                name=str(row.get("name") or _persona_name(idx, point)),
                axis_values={k: float(v) for k, v in point.items()},
                axis_interpretation={str(k): str(v) for k, v in (row.get("axis_interpretation") or {}).items()},
                short_rule=str(row.get("short_rule") or "reason independently and verify the final answer"),
                reasoning_summary=str(row.get("reasoning_summary") or "apply a distinct operational reasoning policy"),
            )
        )
    return descriptors


def _llm_descriptors(
    *,
    config: PersonaGenerationConfig,
    axis_selection: Any,
    points: list[dict[str, float]],
    engine: Any,
) -> tuple[list[PersonaDescriptor], dict[str, Any]]:
    messages = build_descriptor_messages(
        config=config, axis_selection=axis_selection, points=points,
    )
    result = ensure_inference_results(
        engine,
        [messages],
        batch_size=1,
        sampling_kwargs={"max_tokens": DESCRIPTOR_MAX_TOKENS},
        model_role="generator",
    )[0]
    descriptors = parse_descriptor_result(
        str(result.text), n_personas=config.n_personas, points=points,
    )
    return descriptors, inference_result_metadata(result)


def _effective_backend(*, config: PersonaGenerationConfig, engine: Any | None) -> str:
    _ = config
    if engine is None:
        raise ValueError("persona generation requires a generator engine")
    return "llm"


def _descriptor_prompt_axis(axis: Any) -> dict[str, Any]:
    axis_id = str(getattr(axis, "axis_id", "") or "")
    name = str(getattr(axis, "name", "") or "")
    kind = str(getattr(axis, "kind", "") or "")
    if kind == "fixed":
        return {
            "axis_id": axis_id,
            "name": name,
            "kind": kind,
            "low_desc": str(getattr(axis, "low_desc", "") or ""),
            "high_desc": str(getattr(axis, "high_desc", "") or ""),
            "notes": None if getattr(axis, "notes", None) is None else str(getattr(axis, "notes", "") or ""),
        }

    return {
        "axis_id": axis_id,
        "name": name,
        "kind": kind,
        "low_desc": f"Low end of the '{name}' reasoning preference.",
        "high_desc": f"High end of the '{name}' reasoning preference.",
        "notes": (
            "Interpret this axis abstractly from its name only. "
            "Do not reuse task-specific objects, formulas, constants, or answer conditions."
        ),
    }


def _descriptor_prompt_axes(axis_selection: Any) -> list[dict[str, Any]]:
    return [_descriptor_prompt_axis(axis) for axis in getattr(axis_selection, "axes", []) or []]


def _descriptor_context_texts(axis_selection: Any) -> list[str]:
    # Descriptor generation is now benchmark-family-level and does not see the raw
    # question. Revalidating against task-axis text or question summaries creates
    # false positives on allowed shared vocabulary such as "differentiation" or
    # "abstraction", so live descriptor generation intentionally skips contextual
    # overlap checks beyond the base leakage validator.
    _ = axis_selection
    return []


def _validate_descriptor_for_generation(
    descriptor: PersonaDescriptor,
    *,
    config: PersonaGenerationConfig,
    axis_selection: Any,
    backend: str,
) -> ValidationResult:
    base = validate_descriptor(descriptor)
    if base.status != "accept":
        return base
    _ = backend
    return validate_descriptor_against_task(
        descriptor,
        question=None,
        raw_task=None,
        context_texts=_descriptor_context_texts(axis_selection),
    )


def prepare_descriptor_generation(
    *,
    config: PersonaGenerationConfig,
    engine: Any | None = None,
) -> tuple[Any, list[dict[str, float]], str]:
    backend = _effective_backend(config=config, engine=engine)
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
        seed=config.persona_seed,
        method=config.sampling_method,
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
    engine: Any | None = None,
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
    parts.append("")
    parts.append("Regenerate all descriptors. Requirements:")
    parts.append("- Describe reusable reasoning POLICY only (search order, verification timing, "
                 "pruning strategy, revision triggers, evidence standards)")
    parts.append("- No item-specific mathematical objects, equations, constants, or solve strategies")
    parts.append("- Each persona must be operationally distinct from every other")
    parts.append("- Return ONLY a valid JSON object—no prose, no markdown fencing, no explanation")
    return "\n".join(parts)


def generate_descriptors_from_state(
    *,
    config: PersonaGenerationConfig,
    axis_selection: Any,
    points: list[dict[str, float]],
    engine: Any | None = None,
) -> tuple[list[PersonaDescriptor], dict[str, Any]]:
    last_meta: dict[str, Any] | None = None
    backend = _effective_backend(config=config, engine=engine)
    attempt_audits: list[dict[str, Any]] = []
    retry_context: list[dict[str, Any]] | None = None
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        descriptor_messages: list[dict[str, Any]] | None = None
        raw_result_text: str | None = None
        parse_error: str | None = None
        base_messages = build_descriptor_messages(
            config=config,
            axis_selection=axis_selection,
            points=points,
        )
        if retry_context is not None:
            descriptor_messages = base_messages + retry_context
        else:
            descriptor_messages = base_messages
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
                n_personas=config.n_personas,
                points=points,
            )
        except ValueError as exc:
            parse_error = str(exc)
            descriptors = []
        validator_rows: list[dict[str, Any]] = []
        has_retry = parse_error is not None
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
            if validation.status == "retry":
                has_retry = True
        dupes = duplicate_diagnostics(d.reasoning_summary for d in descriptors)
        validator_meta = {
            "descriptor_validations": validator_rows,
            "descriptor_duplicates": dupes,
            "descriptor_prompt_version": DESCRIPTOR_PROMPT_VERSION,
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
                "validator_metadata": audit_validator_meta,
            }
        )
        validator_meta["attempt_audits"] = [dict(audit) for audit in attempt_audits]
        last_meta = {
            "axis_selection": axis_selection,
            "sampled_points": points,
            "validator_metadata": validator_meta,
        }
        if not has_retry and not dupes:
            return descriptors, last_meta
        retry_context = [
            {
                "role": "user",
                "content": _build_descriptor_retry_feedback(
                    parse_error=parse_error,
                    validator_rows=validator_rows,
                    duplicates=dupes,
                ),
            },
        ]
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
    return PersonaCard(
        persona_id=str(payload.get("persona_id") or descriptor.persona_id),
        title=str(payload.get("title") or descriptor.name),
        core_reasoning_strategy=str(payload.get("core_reasoning_strategy") or descriptor.reasoning_summary),
        priorities=[str(x) for x in payload.get("priorities", [])],
        distrusts=[str(x) for x in payload.get("distrusts", [])],
        decomposition_style=str(payload.get("decomposition_style") or descriptor.short_rule),
        revision_policy=str(payload.get("revision_policy") or "revise only on concrete evidence"),
        confidence_policy=str(payload.get("confidence_policy") or "be explicit about uncertainty"),
        failure_mode_to_avoid=str(payload.get("failure_mode_to_avoid") or "do not collapse into answer-first reasoning"),
        system_prompt=str(payload.get("system_prompt") or descriptor.reasoning_summary),
        card_version=CARD_PROMPT_VERSION,
    )


def _build_card_retry_feedback(
    *,
    parse_error: str | None,
    validation: ValidationResult | dict[str, Any] | None,
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
    parts.append("")
    parts.append("Regenerate this card. Requirements:")
    parts.append("- Return ONLY a valid JSON object matching the requested schema")
    parts.append("- No markdown fencing, no prose, no explanation")
    parts.append("- Keep the card compact and operational")
    parts.append("- Focus on reasoning policy, not biography or style")
    return "\n".join(parts)


def _llm_card(
    *,
    descriptor: PersonaDescriptor,
    question: str,
    question_media: list[dict[str, Any]] | None,
    engine: Any,
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
    engine: Any | None,
    backend: str,
) -> list[PersonaCard]:
    """Re-run card generation for duplicates with enhanced distinctiveness."""
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
            engine=cast(Any, engine),
        )
        updated[idx] = card
    return updated


def expand_cards(
    descriptors: list[PersonaDescriptor],
    *,
    dataset: str = "unknown",
    question: str,
    raw_task: dict[str, Any] | None = None,
    engine: Any | None = None,
    backend: str = "llm",
) -> tuple[list[PersonaCard], dict[str, Any]]:
    last_meta: dict[str, Any] | None = None
    question_media = _question_media_for_task(dataset=dataset, raw_task=raw_task)
    if engine is None:
        raise ValueError("persona generation requires a generator engine for card generation")
    attempt_audits: list[dict[str, Any]] = []
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        cards: list[PersonaCard] = []
        validator_rows: list[dict[str, Any]] = []
        call_metadata: list[dict[str, Any] | None] = []
        has_retry = False
        for descriptor in descriptors:
            card_call_meta: dict[str, Any] | None = None
            raw_result_text: str | None = None
            parse_error: str | None = None
            request_messages: list[dict[str, Any]] | None = None
            validation: ValidationResult | None = None
            request_messages = build_card_messages(
                descriptor=descriptor,
                question=question,
                question_media=question_media,
            )
            if attempt > 0:
                prev = next(
                    (
                        audit for audit in reversed(attempt_audits)
                        if audit.get("persona_id") == descriptor.persona_id
                    ),
                    None,
                )
                request_messages = request_messages + [
                    {
                        "role": "user",
                        "content": _build_card_retry_feedback(
                            parse_error=None if prev is None else prev.get("parse_error"),
                            validation=None if prev is None else prev.get("validation"),
                        ),
                    }
                ]
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
                if validation.status == "reject_hard":
                    raise ValueError(f"Card {card.persona_id} rejected: {validation.reasons}")
                if validation.status == "retry":
                    has_retry = True
                cards.append(card)
            call_metadata.append(card_call_meta)
            attempt_audits.append(
                {
                    "attempt": attempt,
                    "persona_id": descriptor.persona_id,
                    "request_messages": request_messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": parse_error,
                    "validation": None if validation is None else asdict(validation),
                    "card": None if parse_error is not None or card is None else asdict(card),
                }
            )
        dupes = duplicate_diagnostics(card.system_prompt for card in cards)
        if dupes:
            cards = _regenerate_duplicate_cards(
                cards=cards,
                descriptors=descriptors,
                duplicate_rows=dupes,
                dataset=dataset,
                question=question,
                raw_task=raw_task,
                engine=engine,
                backend=backend,
            )
            dupes = duplicate_diagnostics(card.system_prompt for card in cards)
        last_meta = {
            "card_validations": validator_rows,
            "card_duplicates": dupes,
            "card_prompt_version": CARD_PROMPT_VERSION,
            "card_backend": backend,
            "card_call_metadata": call_metadata,
            "card_attempt_audits": [dict(audit) for audit in attempt_audits],
        }
        if not has_retry and not dupes:
            return cards, last_meta
    assert last_meta is not None
    raise GenerationExhaustedError(
        f"Card generation exhausted retries: {last_meta}",
        stage="cards",
        metadata=last_meta,
    )


def build_persona_artifact(
    *,
    config: PersonaGenerationConfig,
    judge_card: Any | None = None,
    generator_engine: Any | None = None,
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
            "n_personas": int(config.n_personas),
            "persona_seed": int(config.persona_seed),
            "axis_mode": str(config.axis_mode),
            "fixed_axis_count": int(config.fixed_axis_count),
            "task_axis_count": int(config.task_axis_count),
            "sampling_method": str(config.sampling_method),
            "generator_model": config.generator_model,
            "judge_generator_model": config.judge_generator_model,
            "judge_persona_mode": str(config.judge_persona_mode),
            "backend": backend,
            "axes_file": None if config.axes_file is None else str(config.axes_file),
            "n_plain_agents": n_plain,
        },
        validator_metadata=validator_metadata,
        slot_layout=slot_layout,
    )


def build_persona_validator_metadata(
    *,
    axis_selection: Any,
    descriptor_validator_metadata: dict[str, Any] | None,
    card_metadata: dict[str, Any] | None,
    backend: str,
    judge_card: Any | None = None,
) -> dict[str, Any]:
    validator_metadata: dict[str, Any] = {}
    validator_metadata.update(dict(descriptor_validator_metadata or {}))
    validator_metadata.update(dict(card_metadata or {}))
    validator_metadata["generator_backend"] = backend
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
            cast(list[dict[str, Any] | None], dict(card_metadata or {}).get("card_call_metadata") or [])
        ),
        "judge_generation": _sum_token_counts([judge_call_meta]),
    }
    return validator_metadata
