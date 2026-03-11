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
)
from .validators import duplicate_diagnostics, validate_card, validate_descriptor


MAX_GENERATION_RETRIES = 2

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


def _heuristic_descriptors(*, config: PersonaGenerationConfig, axes: list[Axis], points: list[dict[str, float]]) -> list[PersonaDescriptor]:
    descriptors: list[PersonaDescriptor] = []
    for idx, point in enumerate(points):
        axis_interpretation = {
            axis.axis_id: _axis_interpretation(axis, float(point.get(axis.axis_id, 0.5)))
            for axis in axes
        }
        ordered_interps = list(axis_interpretation.values())
        short_rule = "; ".join(ordered_interps[:2]) if ordered_interps else "reason independently and verify the final answer"
        reasoning_summary = (
            "Focus on " + ", ".join(ordered_interps[:3])
            if ordered_interps
            else "focus on an independently reasoned solution"
        )
        descriptors.append(
            PersonaDescriptor(
                persona_id=f"persona_{idx + 1}",
                name=_persona_name(idx, point),
                axis_values={k: float(v) for k, v in point.items()},
                axis_interpretation=axis_interpretation,
                short_rule=short_rule,
                reasoning_summary=reasoning_summary,
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
    question_media = _question_media_for_task(dataset=config.dataset, raw_task=config.raw_task)
    messages = build_stage1_messages(
        dataset=config.dataset,
        benchmark_family=axis_selection.benchmark_family or config.dataset,
        question=config.question,
        axes=[asdict(axis) for axis in axis_selection.axes],
        sampled_points=points,
        question_media=question_media,
    )
    result = ensure_inference_results(
        engine,
        [messages],
        batch_size=1,
        sampling_kwargs={"max_tokens": 4096},
        model_role="generator",
    )[0]
    payload = parse_json_payload(str(result.text))
    rows = payload.get("descriptors") or []
    if len(rows) < config.n_personas:
        raise ValueError(f"Stage-1 descriptor generation returned too few rows: {rows}")
    descriptors: list[PersonaDescriptor] = []
    for idx, row in enumerate(rows[: config.n_personas]):
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
    return descriptors, inference_result_metadata(result)


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
    last_meta: dict[str, Any] | None = None
    backend = "llm" if config.backend == "llm" or (config.backend == "auto" and engine is not None) else "heuristic"
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
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        if backend == "llm" and engine is not None:
            descriptors, descriptor_call_meta = _llm_descriptors(
                config=config,
                axis_selection=axis_selection,
                points=points,
                engine=engine,
            )
        else:
            descriptors = _heuristic_descriptors(config=config, axes=axis_selection.axes, points=points)
            descriptor_call_meta = None
        validator_rows: list[dict[str, Any]] = []
        has_retry = False
        for descriptor in descriptors:
            validation = validate_descriptor(descriptor)
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
                raise ValueError(f"Descriptor {descriptor.persona_id} rejected: {validation.reasons}")
            if validation.status == "retry":
                has_retry = True
        dupes = duplicate_diagnostics(d.reasoning_summary for d in descriptors)
        validator_meta = {
            "descriptor_validations": validator_rows,
            "descriptor_duplicates": dupes,
            "descriptor_prompt_version": DESCRIPTOR_PROMPT_VERSION,
            "descriptor_backend": backend,
            "descriptor_call_metadata": descriptor_call_meta,
        }
        last_meta = {
            "axis_selection": axis_selection,
            "sampled_points": points,
            "validator_metadata": validator_meta,
        }
        if not has_retry and not dupes:
            return descriptors, last_meta
    assert last_meta is not None
    raise ValueError(f"Descriptor generation exhausted retries: {last_meta['validator_metadata']}")


def _heuristic_card(descriptor: PersonaDescriptor) -> PersonaCard:
    priorities = [
        descriptor.short_rule,
        "surface the main failure mode before finalizing",
    ]
    distrusts = [
        "unsupported leaps",
        "answer-first rationalization",
    ]
    return PersonaCard(
        persona_id=descriptor.persona_id,
        title=descriptor.name,
        core_reasoning_strategy=descriptor.reasoning_summary,
        priorities=priorities,
        distrusts=distrusts,
        decomposition_style=next(iter(descriptor.axis_interpretation.values()), descriptor.short_rule),
        revision_policy="update the answer only when another path exposes a concrete flaw in the current reasoning",
        confidence_policy="be explicit about uncertainty when a critical step is weak",
        failure_mode_to_avoid="do not collapse into decorative style or answer-first reasoning",
        system_prompt=(
            "You are solving as an operational reasoning persona.\n"
            f"Persona: {descriptor.name}\n"
            f"Core strategy: {descriptor.reasoning_summary}\n"
            f"Priorities: {'; '.join(priorities)}\n"
            f"Distrust: {'; '.join(distrusts)}\n"
            "Do not role-play biography. Focus on how to reason."
        ),
        card_version=CARD_PROMPT_VERSION,
    )


def _llm_card(
    *,
    descriptor: PersonaDescriptor,
    question: str,
    question_media: list[dict[str, Any]] | None,
    engine: Any,
) -> tuple[PersonaCard, dict[str, Any]]:
    messages = build_stage2_messages(
        question=question,
        descriptor=asdict(descriptor),
        question_media=question_media,
    )
    result = ensure_inference_results(
        engine,
        [messages],
        batch_size=1,
        sampling_kwargs={"max_tokens": 4096},
        model_role="generator",
    )[0]
    payload = parse_json_payload(str(result.text))
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
    ), inference_result_metadata(result)


def _heuristic_card_distinctive(descriptor: PersonaDescriptor) -> PersonaCard:
    """Generate a heuristic card that uses all axis interpretations for distinctiveness."""
    all_interps = list(descriptor.axis_interpretation.values())
    priorities = [
        descriptor.short_rule,
        "surface the main failure mode before finalizing",
    ]
    if len(all_interps) > 2:
        priorities.append(all_interps[-1])
    distrusts = [
        "unsupported leaps",
        "answer-first rationalization",
        f"ignoring the {descriptor.persona_id} operating policy",
    ]
    decomp = "; ".join(all_interps) if all_interps else descriptor.short_rule
    return PersonaCard(
        persona_id=descriptor.persona_id,
        title=descriptor.name,
        core_reasoning_strategy=descriptor.reasoning_summary,
        priorities=priorities,
        distrusts=distrusts,
        decomposition_style=decomp,
        revision_policy="update the answer only when another path exposes a concrete flaw in the current reasoning",
        confidence_policy="be explicit about uncertainty when a critical step is weak",
        failure_mode_to_avoid="do not collapse into decorative style or answer-first reasoning",
        system_prompt=(
            "You are solving as an operational reasoning persona.\n"
            f"Persona: {descriptor.name} ({descriptor.persona_id})\n"
            f"Core strategy: {descriptor.reasoning_summary}\n"
            f"Axis policy: {decomp}\n"
            f"Priorities: {'; '.join(priorities)}\n"
            f"Distrust: {'; '.join(distrusts)}\n"
            "Do not role-play biography. Focus on how to reason."
        ),
        card_version=CARD_PROMPT_VERSION,
    )


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
        if backend == "llm" and engine is not None:
            card, _ = _llm_card(
                descriptor=descriptor,
                question=question,
                question_media=question_media,
                engine=engine,
            )
        else:
            card = _heuristic_card_distinctive(descriptor)
        updated[idx] = card
    return updated


def expand_cards(
    descriptors: list[PersonaDescriptor],
    *,
    dataset: str = "unknown",
    question: str,
    raw_task: dict[str, Any] | None = None,
    engine: Any | None = None,
    backend: str = "heuristic",
) -> tuple[list[PersonaCard], dict[str, Any]]:
    last_meta: dict[str, Any] | None = None
    question_media = _question_media_for_task(dataset=dataset, raw_task=raw_task)
    for attempt in range(MAX_GENERATION_RETRIES + 1):
        cards: list[PersonaCard] = []
        validator_rows: list[dict[str, Any]] = []
        call_metadata: list[dict[str, Any] | None] = []
        has_retry = False
        for descriptor in descriptors:
            if backend == "llm" and engine is not None:
                card, card_call_meta = _llm_card(
                    descriptor=descriptor,
                    question=question,
                    question_media=question_media,
                    engine=engine,
                )
            else:
                card, card_call_meta = _heuristic_card(descriptor), None
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
        }
        if not has_retry and not dupes:
            return cards, last_meta
    assert last_meta is not None
    raise ValueError(f"Card generation exhausted retries: {last_meta}")


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
    backend = "llm" if config.backend == "llm" or (config.backend == "auto" and generator_engine is not None) else "heuristic"
    cards, card_meta = expand_cards(
        descriptors,
        dataset=config.dataset,
        question=config.question,
        raw_task=config.raw_task,
        engine=generator_engine,
        backend=backend,
    )
    validator_metadata: dict[str, Any] = {}
    validator_metadata.update(descriptor_meta["validator_metadata"])
    validator_metadata.update(card_meta)
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
        "descriptor_generation": _sum_token_counts([descriptor_meta["validator_metadata"].get("descriptor_call_metadata")]),
        "card_generation": _sum_token_counts(cast(list[dict[str, Any] | None], card_meta.get("card_call_metadata") or [])),
        "judge_generation": _sum_token_counts([judge_call_meta]),
    }
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
        },
        validator_metadata=validator_metadata,
    )
