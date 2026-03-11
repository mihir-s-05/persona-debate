from __future__ import annotations

import json
import io
import sys
from pathlib import Path

import pytest

from debate_v_majority.cli import main_impl as cli_main_impl
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.personas.axes import build_axis_selection
from debate_v_majority.personas import (
    FIXED_AXIS_BANK,
    PersonaGenerationConfig,
    artifact_path,
    build_judge_card,
    build_persona_artifact,
    ensure_judge_bank_card,
    duplicate_diagnostics,
    default_judge_bank_dir,
    expand_cards,
    get_fixed_axes,
    legacy_artifact_path,
    load_artifact,
    make_item_uid,
    resolve_judge_family_assignment,
    sample_axis_points,
    save_artifact,
    validate_text_for_leakage,
)
from debate_v_majority.personas.prompt_templates import (
    build_judge_messages,
    build_stage1_messages,
    build_stage2_messages,
    build_task_axis_messages,
)
from debate_v_majority.personas.generator import MAX_GENERATION_RETRIES, _axis_interpretation, generate_descriptors
from debate_v_majority.personas.schema import Axis, PersonaCard, PersonaDescriptor, ValidationResult


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class _FakeEngine:
    model_name = "gemini-3-flash"
    provider_name = "fake-provider"

    def __init__(self) -> None:
        self.calls = []

    def generate_batch(self, contexts, batch_size=1, sampling_kwargs=None, progress_callback=None, model_role=None):
        self.calls.append(
            {
                "contexts": contexts,
                "batch_size": batch_size,
                "sampling_kwargs": dict(sampling_kwargs or {}),
                "model_role": model_role,
            }
        )
        outputs = []
        for ctx in contexts:
            prompt_content = ctx[-1]["content"]
            if isinstance(prompt_content, list):
                prompt = "\n".join(
                    str(part.get("text") or "")
                    for part in prompt_content
                    if isinstance(part, dict) and str(part.get("type") or "text") == "text"
                )
            else:
                prompt = prompt_content
            if "Propose reasoning-relevant axes" in prompt:
                outputs.append(
                    json.dumps(
                        {
                            "axes": [
                                {
                                    "axis_id": "llm_axis",
                                    "name": "LLM Axis",
                                    "low_desc": "low mode",
                                    "high_desc": "high mode",
                                    "notes": "generated",
                                }
                            ]
                        }
                    )
                )
            elif "Generate the full persona population jointly" in prompt:
                outputs.append(
                    json.dumps(
                        {
                            "descriptors": [
                                {
                                    "persona_id": "persona_1",
                                    "name": "Verifier",
                                    "axis_interpretation": {"llm_axis": "leans low"},
                                    "short_rule": "verify before concluding",
                                    "reasoning_summary": "checks constraints before committing",
                                },
                                {
                                    "persona_id": "persona_2",
                                    "name": "Pruner",
                                    "axis_interpretation": {"llm_axis": "leans high"},
                                    "short_rule": "prune quickly and stress test",
                                    "reasoning_summary": "moves fast but checks contradictions",
                                },
                            ]
                        }
                    )
                )
            elif "Expand each descriptor into a compact" in prompt:
                persona_id = "persona_1" if "persona_1" in prompt else "persona_2"
                outputs.append(
                    json.dumps(
                        {
                            "persona_id": persona_id,
                            "title": f"Generated Card {persona_id}",
                            "core_reasoning_strategy": f"use a generated reasoning strategy for {persona_id}",
                            "priorities": ["track constraints", "check failure modes"],
                            "distrusts": ["unsupported jumps", "answer-first thinking"],
                            "decomposition_style": "generated decomposition",
                            "revision_policy": "revise on concrete counterevidence",
                            "confidence_policy": "state uncertainty directly",
                            "failure_mode_to_avoid": "do not leak answer hints",
                            "system_prompt": f"Generated operational system prompt for {persona_id}.",
                        }
                    )
                )
            elif "Generate a constrained judge card" in prompt:
                outputs.append(
                    json.dumps(
                        {
                            "judge_id": "judge_generated",
                            "judge_family": "competition_math",
                            "domain_scope": "competition_math",
                            "evaluation_priorities": ["prefer transcript-supported answers"],
                            "tie_break_policy": "prefer clearer support",
                            "independent_resolve_policy": "limited_check_only",
                            "answer_format_policy": "strict",
                            "confidence_policy": "optional",
                            "system_prompt": "Generated judge prompt",
                        }
                    )
                )
            else:
                raise AssertionError(f"Unexpected prompt: {prompt}")
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def test_sample_axis_points_is_deterministic():
    points1 = sample_axis_points(axes=[], num_personas=3, seed=7)
    points2 = sample_axis_points(axes=[], num_personas=3, seed=7)
    assert points1 == points2 == [{}, {}, {}]


def test_sample_axis_points_with_axes_are_bounded_and_distinct():
    axes = FIXED_AXIS_BANK[:2]
    points = sample_axis_points(axes=axes, num_personas=4, seed=3, method="maximin")
    assert len(points) == 4
    assert len({tuple(sorted(p.items())) for p in points}) == 4
    assert all(0.0 <= value <= 1.0 for point in points for value in point.values())


def test_halton_sampling_is_deterministic():
    axes = FIXED_AXIS_BANK[:3]
    points1 = sample_axis_points(axes=axes, num_personas=4, seed=5, method="halton")
    points2 = sample_axis_points(axes=axes, num_personas=4, seed=5, method="halton")
    assert points1 == points2


def test_balanced_principle_option_axis_interpretation_is_grammatical():
    axis = Axis(
        axis_id="principle_first_vs_option_first",
        name="Principle First vs Option First",
        kind="task",
        low_desc="Identify the governing principle before evaluating options.",
        high_desc="Scan options first and use them to guide which principle matters.",
    )
    interpretation = _axis_interpretation(axis, 0.4206)

    assert interpretation == (
        "blend two moves: Identify the governing principle before evaluating options; "
        "also scan options first and use them to guide which principle matters."
    )
    assert "balance identify" not in interpretation


def test_validate_text_for_leakage_rejects_answer_like_content():
    result = validate_text_for_leakage("The likely answer is A, so choose option A.")
    assert result.status == "reject_hard"


def test_duplicate_diagnostics_flags_near_duplicates():
    dupes = duplicate_diagnostics(
        [
            "reason carefully and verify the final answer",
            "reason carefully and verify the final answer before committing",
        ],
        threshold=0.5,
    )
    assert dupes


def test_judge_generation_modes_differ():
    raw_task = {"problem": "What is 1+1?", "answer": "2"}
    neutral = build_judge_card(dataset="aime25", raw_task=raw_task, question="q", mode="neutral_baseline")
    family = build_judge_card(dataset="aime25", raw_task=raw_task, question="q", mode="task_family_generated")
    question = build_judge_card(dataset="aime25", raw_task=raw_task, question="What is 1+1?", mode="question_conditioned_generated")
    assert neutral is not None and family is not None and question is not None
    assert neutral.judge_family != family.judge_family
    assert question.domain_scope != family.domain_scope


def test_benchmark_family_bank_artifact_and_family_routing(tmp_path: Path):
    judge_card, artifact, path = ensure_judge_bank_card(
        judge_bank_dir=default_judge_bank_dir(artifacts_dir=tmp_path),
        dataset="aime25",
        judge_family="math",
        engine=None,
        generator_model=None,
        backend="heuristic",
        refresh=False,
    )
    assert judge_card.judge_family == "math"
    assert artifact.judge_card.judge_id == judge_card.judge_id
    assert path.exists()

    assignment = resolve_judge_family_assignment(
        dataset="hle",
        item_uid="hle:item-1",
        question="What is 1+1?",
        raw_task={"category": "physics"},
    )
    assert assignment.judge_family == "physical_sciences"

    prepared_assignment = resolve_judge_family_assignment(
        dataset="hle",
        item_uid="hle:item-2",
        question="What is Bayes theorem?",
        raw_task={"category": "humanities/social science", "domain_family": "humanities"},
    )
    assert prepared_assignment.judge_family == "humanities"


def test_build_persona_artifact_and_roundtrip(tmp_path: Path):
    raw_task = {
        "problem": "What is 1+1?",
        "answer": "2",
    }
    config = PersonaGenerationConfig(
        dataset="aime25",
        question="Solve the problem carefully and give the final answer in the form \\boxed{answer}.\nProblem: What is 1+1?",
        raw_task=raw_task,
        item_uid="aime25:h:test",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=3,
        persona_seed=11,
        axis_mode="hybrid",
        fixed_axis_count=2,
        task_axis_count=1,
    )
    judge_card = build_judge_card(dataset="aime25", raw_task=raw_task, question=config.question)
    artifact = build_persona_artifact(config=config, judge_card=judge_card)
    assert artifact.item_uid == "aime25:h:test"
    assert len(artifact.cards) == 3
    path = save_artifact(artifacts_dir=tmp_path, artifact=artifact)
    restored = load_artifact(path=path)
    assert restored.to_dict() == artifact.to_dict()


def test_save_artifact_keys_path_by_generation_config(tmp_path: Path):
    raw_task = {"problem": "What is 1+1?", "answer": "2"}
    judge_card = build_judge_card(dataset="aime25", raw_task=raw_task, question="What is 1+1?")
    config_a = PersonaGenerationConfig(
        dataset="aime25",
        question="What is 1+1?",
        raw_task=raw_task,
        item_uid="aime25:h:shared-item",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=3,
        persona_seed=11,
        axis_mode="hybrid",
        fixed_axis_count=2,
        task_axis_count=1,
    )
    config_b = PersonaGenerationConfig(
        dataset="aime25",
        question="What is 1+1?",
        raw_task=raw_task,
        item_uid="aime25:h:shared-item",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=3,
        persona_seed=42,
        axis_mode="hybrid",
        fixed_axis_count=2,
        task_axis_count=1,
    )
    artifact_a = build_persona_artifact(config=config_a, judge_card=judge_card)
    artifact_b = build_persona_artifact(config=config_b, judge_card=judge_card)

    path_a = save_artifact(artifacts_dir=tmp_path, artifact=artifact_a)
    path_b = save_artifact(artifacts_dir=tmp_path, artifact=artifact_b)

    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()
    assert "--cfg-" in path_a.name
    assert "--cfg-" in path_b.name


def test_resolve_persona_artifact_replay_falls_back_to_legacy_path(tmp_path: Path):
    dataset_path = tmp_path / "gpqa.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-legacy-replay",
            }
        ],
    )
    items, _ = _make_dataset_subset(
        dataset="gpqa",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    item = items[0]
    question, _, raw_task = cli_main_impl._parse_question_answer("gpqa", item.raw_task)
    artifact, _ = cli_main_impl._resolve_persona_artifact(
        dataset="gpqa",
        item=item,
        question=question,
        raw_task=raw_task,
        artifacts_dir=tmp_path / "artifacts",
        n_personas=3,
        persona_seed=7,
        axis_mode="hybrid",
        fixed_axis_count=2,
        task_axis_count=1,
        sampling_method="maximin",
        judge_persona_mode="task_family_generated",
        backend="heuristic",
        generator_model=None,
        judge_generator_model=None,
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        save_artifact=False,
        replay=False,
    )
    legacy_path = legacy_artifact_path(
        artifacts_dir=tmp_path / "artifacts",
        dataset="gpqa",
        item_uid=item.item_uid,
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    replayed, replay_path = cli_main_impl._resolve_persona_artifact(
        dataset="gpqa",
        item=item,
        question=question,
        raw_task=raw_task,
        artifacts_dir=tmp_path / "artifacts",
        n_personas=3,
        persona_seed=7,
        axis_mode="hybrid",
        fixed_axis_count=2,
        task_axis_count=1,
        sampling_method="maximin",
        judge_persona_mode="task_family_generated",
        backend="heuristic",
        generator_model=None,
        judge_generator_model=None,
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        save_artifact=False,
        replay=True,
    )

    assert replayed.to_dict() == artifact.to_dict()
    assert replay_path == legacy_path
    assert artifact_path(
        artifacts_dir=tmp_path / "artifacts",
        dataset="gpqa",
        item_uid=item.item_uid,
        dataset_revision=item.dataset_revision,
        generation_settings=artifact.generation_settings,
    ) != legacy_path


def test_build_persona_artifact_llm_path():
    raw_task = {"problem": "What is 1+1?", "answer": "2"}
    config = PersonaGenerationConfig(
        dataset="aime25",
        question="Solve the problem carefully and give the final answer in the form \\boxed{answer}.\nProblem: What is 1+1?",
        raw_task=raw_task,
        item_uid="aime25:h:test-llm",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=2,
        persona_seed=5,
        axis_mode="task",
        backend="llm",
        generator_model="gemini-3-flash",
        judge_generator_model="gemini-3-flash",
    )
    engine = _FakeEngine()
    judge_card = build_judge_card(
        dataset="aime25",
        raw_task=raw_task,
        question=config.question,
        mode="task_family_generated",
        engine=engine,
        generator_model="gemini-3-flash",
        backend="llm",
    )
    artifact = build_persona_artifact(config=config, judge_card=judge_card, generator_engine=engine)
    assert artifact.generator_model == "gemini-3-flash"
    assert artifact.axes.axes[0].axis_id == "llm_axis"
    assert artifact.axes.axes[0].source["call_metadata"]["provider_name"] == "fake-provider"
    assert artifact.axes.axes[0].source["call_metadata"]["model_role"] == "generator"
    assert artifact.judge_card is not None
    assert artifact.judge_card.source["call_metadata"]["provider_name"] == "fake-provider"
    assert artifact.judge_card.source["call_metadata"]["model_role"] == "judge_generator"
    assert artifact.cards[0].system_prompt.startswith("Generated operational system prompt")
    assert len(engine.calls) >= 4
    assert all(call["sampling_kwargs"]["max_tokens"] == 4096 for call in engine.calls)
    assert all("temperature" not in call["sampling_kwargs"] for call in engine.calls)


def test_hle_llm_persona_and_judge_generation_attach_question_images():
    raw_task = {
        "id": "hle-image-1",
        "question": "Which option is correct?\nA) red\nB) blue",
        "answer": "blue",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
        "image": "https://example.invalid/hle-image.png",
    }
    question = (
        "Solve the following HLE-Verified task carefully.\n"
        "Return your final answer in the required strict format.\n"
        "If you report confidence, use `Confidence: <number between 0 and 1>`.\n\n"
        "Which option is correct?\nA) red\nB) blue\n\n"
        "Image references present (not attached in this text-only run):\n"
        "- image=https://example.invalid/hle-image.png\n\n"
        "Put your final answer in the form \\boxed{A} using exactly one choice label from A, B."
    )
    config = PersonaGenerationConfig(
        dataset="hle",
        question=question,
        raw_task=raw_task,
        item_uid="hle:h:multimodal-test",
        item_display_id="hle-image-1",
        dataset_revision="rev-test",
        n_personas=2,
        persona_seed=5,
        axis_mode="task",
        backend="llm",
        generator_model="gemini-3-flash",
        judge_generator_model="gemini-3-flash",
    )
    engine = _FakeEngine()
    judge_card = build_judge_card(
        dataset="hle",
        raw_task=raw_task,
        question=question,
        mode="task_family_generated",
        engine=engine,
        generator_model="gemini-3-flash",
        backend="llm",
    )
    artifact = build_persona_artifact(config=config, judge_card=judge_card, generator_engine=engine)
    assert artifact.judge_card is not None
    assert artifact.cards
    assert len(engine.calls) >= 4
    for call in engine.calls:
        user_message = call["contexts"][0][-1]
        assert isinstance(user_message["content"], list)
        assert user_message["content"][0]["type"] == "text"
        assert any(part.get("type") == "image" for part in user_message["content"][1:])


def test_make_dataset_subset_populates_stable_identity(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": "What is correct?",
            "Correct Answer": "blue",
            "Incorrect Answer 1": "red",
            "Incorrect Answer 2": "green",
            "Incorrect Answer 3": "yellow",
            "id": "gpqa-item-1",
        }
    ]
    _write_jsonl(path, rows)
    items, meta = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=1,
        seed=123,
        ids=[0],
        range_str=None,
    )
    assert items[0].item_uid == "gpqa:gpqa-item-1"
    assert items[0].dataset_revision == meta["dataset_revision"]
    assert items[0].item_display_id == "gpqa-item-1"


def test_make_item_uid_falls_back_to_content_hash_for_aime():
    row = {"problem": "What is 1+1?", "answer": "2"}
    uid = make_item_uid(dataset="aime25", raw_task=row, dataset_revision="rev-a")
    assert uid.startswith("aime25:h:")


def test_fixed_axis_bank_has_expected_shape():
    assert len(FIXED_AXIS_BANK) == 6
    assert FIXED_AXIS_BANK[0].axis_id == "symbolic_vs_intuitive"


def test_get_fixed_axes_returns_requested_prefix():
    assert get_fixed_axes(0) == []
    assert get_fixed_axes(2) == FIXED_AXIS_BANK[:2]
    assert get_fixed_axes(999) == FIXED_AXIS_BANK


def test_expand_cards_regenerates_only_weaker_duplicates(monkeypatch: pytest.MonkeyPatch):
    descriptors = [
        PersonaDescriptor(
            persona_id="persona_1",
            name="First",
            axis_values={"a": 0.2},
            axis_interpretation={"a": "low"},
            short_rule="verify before concluding",
            reasoning_summary="checks constraints before committing",
        ),
        PersonaDescriptor(
            persona_id="persona_2",
            name="Second",
            axis_values={"a": 0.8},
            axis_interpretation={"a": "high"},
            short_rule="prune quickly and test contradictions",
            reasoning_summary="moves fast but checks contradictions",
        ),
    ]

    def _duplicate_card(descriptor: PersonaDescriptor) -> PersonaCard:
        return PersonaCard(
            persona_id=descriptor.persona_id,
            title=descriptor.name,
            core_reasoning_strategy="shared strategy",
            priorities=["track constraints"],
            distrusts=["unsupported jumps"],
            decomposition_style="shared decomposition",
            revision_policy="revise on concrete evidence",
            confidence_policy="state uncertainty directly",
            failure_mode_to_avoid="do not leak hints",
            system_prompt="Shared operational prompt",
            card_version="test",
        )

    monkeypatch.setattr("debate_v_majority.personas.generator._heuristic_card", _duplicate_card)

    cards, meta = expand_cards(descriptors, question="What is 1+1?")
    assert meta["card_duplicates"] == []
    assert cards[0].system_prompt == "Shared operational prompt"
    # Regenerated card uses _heuristic_card_distinctive which produces a different system prompt
    assert cards[1].system_prompt != "Shared operational prompt"
    assert "persona_2" in cards[1].system_prompt
    # Core strategy preserved from descriptor
    assert cards[1].core_reasoning_strategy == "moves fast but checks contradictions"


def test_generate_descriptors_retries_after_validator_retry(monkeypatch: pytest.MonkeyPatch):
    config = PersonaGenerationConfig(
        dataset="aime25",
        question="What is 1+1?",
        raw_task={"problem": "What is 1+1?", "answer": "2"},
        item_uid="aime25:h:retry-pass",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=2,
        persona_seed=9,
        axis_mode="fixed",
        fixed_axis_count=1,
        backend="heuristic",
    )
    statuses = iter(["retry", "accept", "accept", "accept"])
    call_count = {"n": 0}

    def _fake_validate_descriptor(_descriptor):
        call_count["n"] += 1
        status = next(statuses)
        return ValidationResult(status=status, reasons=[f"status={status}"])

    monkeypatch.setattr("debate_v_majority.personas.generator.validate_descriptor", _fake_validate_descriptor)

    descriptors, meta = generate_descriptors(config=config)
    assert len(descriptors) == 2
    assert call_count["n"] == 4
    assert meta["validator_metadata"]["descriptor_validations"][-1]["attempt"] == 1


def test_generate_descriptors_retry_keeps_axes_and_sampled_points_stable(monkeypatch: pytest.MonkeyPatch):
    config = PersonaGenerationConfig(
        dataset="aime25",
        question="What is 1+1?",
        raw_task={"problem": "What is 1+1?", "answer": "2"},
        item_uid="aime25:h:retry-stable",
        item_display_id=1,
        dataset_revision="rev-test",
        n_personas=2,
        persona_seed=7,
        axis_mode="fixed",
        fixed_axis_count=1,
        backend="heuristic",
    )
    statuses = iter(["retry", "accept", "accept", "accept"])
    axis_calls = {"n": 0}
    sample_calls = {"n": 0}

    def _fake_build_axis_selection(**kwargs):
        axis_calls["n"] += 1
        return build_axis_selection(
            mode=kwargs["mode"],
            question=kwargs["question"],
            dataset=kwargs["dataset"],
            raw_task=kwargs["raw_task"],
            fixed_count=kwargs["fixed_count"],
            task_count=kwargs["task_count"],
            generator_model=kwargs["generator_model"],
            engine=kwargs["engine"],
            backend=kwargs["backend"],
            axes_file=kwargs["axes_file"],
        )

    def _fake_sample_axis_points(**kwargs):
        sample_calls["n"] += 1
        return sample_axis_points(**kwargs)

    def _fake_validate_descriptor(_descriptor):
        status = next(statuses)
        return ValidationResult(status=status, reasons=[f"status={status}"])

    monkeypatch.setattr("debate_v_majority.personas.generator.build_axis_selection", _fake_build_axis_selection)
    monkeypatch.setattr("debate_v_majority.personas.generator.sample_axis_points", _fake_sample_axis_points)
    monkeypatch.setattr("debate_v_majority.personas.generator.validate_descriptor", _fake_validate_descriptor)

    descriptors, meta = generate_descriptors(config=config)
    expected_points = sample_axis_points(
        axes=get_fixed_axes(1),
        num_personas=2,
        seed=7,
        method=config.sampling_method,
    )
    assert len(descriptors) == 2
    assert axis_calls["n"] == 1
    assert sample_calls["n"] == 1
    assert meta["sampled_points"] == expected_points


def test_expand_cards_retry_cap_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch):
    descriptors = [
        PersonaDescriptor(
            persona_id="persona_1",
            name="First",
            axis_values={},
            axis_interpretation={},
            short_rule="verify first",
            reasoning_summary="reason carefully",
        )
    ]
    call_count = {"n": 0}

    def _always_retry(_card):
        call_count["n"] += 1
        return ValidationResult(status="retry", reasons=["still too vague"])

    monkeypatch.setattr("debate_v_majority.personas.generator.validate_card", _always_retry)

    with pytest.raises(ValueError, match="Card generation exhausted retries"):
        expand_cards(descriptors, question="What is 1+1?")

    assert call_count["n"] == MAX_GENERATION_RETRIES + 1


def test_arg_parser_accepts_judge_model_alias():
    parser = cli_main_impl._build_arg_parser()
    args = parser.parse_args(["--dataset", "gpqa", "--mode", "personas", "--judge_model", "gemini-3-flash"])
    assert args.judge_generator_model == "gemini-3-flash"


def test_arg_parser_accepts_debater_model_alias():
    parser = cli_main_impl._build_arg_parser()
    args = parser.parse_args(["--dataset", "gpqa", "--mode", "single", "--debater_model", "gemini-3-flash"])
    assert args.model_name == "gemini-3-flash"


def test_persona_mode_main_generates_and_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": "What is correct?",
            "Correct Answer": "blue",
            "Incorrect Answer 1": "red",
            "Incorrect Answer 2": "green",
            "Incorrect Answer 3": "yellow",
            "id": "gpqa-item-1",
        }
    ]
    _write_jsonl(dataset_path, rows)
    out_dir = tmp_path / "out"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda dataset, **_kw: dataset_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "gpqa",
            "--mode",
            "personas",
            "--subset_ids",
            "0",
            "--out_dir",
            str(out_dir),
            "--persona_artifacts_dir",
            str(artifact_dir),
            "--persona_save_artifacts",
        ],
    )
    cli_main_impl.main()
    outputs = sorted(out_dir.glob("personas_*.jsonl"))
    assert len(outputs) == 1
    lines = outputs[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["mode"] == "personas"
    assert row["item_uid"] == "gpqa:gpqa-item-1"
    assert Path(row["artifact_path"]).exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "gpqa",
            "--mode",
            "personas",
            "--subset_ids",
            "0",
            "--out_dir",
            str(out_dir),
            "--persona_artifacts_dir",
            str(artifact_dir),
            "--persona_replay",
        ],
    )
    cli_main_impl.main()
    outputs_after = sorted(out_dir.glob("personas_*.jsonl"))
    assert len(outputs_after) == 2


def test_persona_generation_emits_benchmark_bank_judge_card(tmp_path: Path):
    dataset_path = tmp_path / "aime_benchmark_bank.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 1+1?", "answer": "2", "id": "aime-bank-1"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )

    rows = cli_main_impl.run_persona_generation(
        dataset="aime25",
        items=items,
        artifacts_dir=tmp_path / "artifacts",
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=1,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="benchmark_family_bank",
        backend="heuristic",
        generator_model=None,
        judge_generator_model=None,
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        judge_bank_dir=tmp_path / "judge_bank",
        judge_bank_refresh=False,
        gpqa_family_cache_path=None,
        save_artifacts=False,
        replay=False,
        dump_cards=False,
        summary_file=io.StringIO(),
    )

    row = rows[0]
    assert row["judge_card"] is not None
    assert row["judge_card"]["judge_family"] == "math"
    assert row["validator_metadata"]["judge_bank"]["judge_family_assignment"]["judge_family"] == "math"
    assert row["validator_metadata"]["judge_bank"]["judge_bank_path"].endswith("math.json")


def test_prompt_templates_emphasize_support_coverage_and_operational_behavior():
    axis_messages = build_task_axis_messages(
        dataset="gpqa",
        benchmark_family="science_multiple_choice",
        question="Which option is correct?",
        count=2,
    )
    axis_user = axis_messages[-1]["content"]
    assert "support coverage" in axis_user
    assert "different first-pass solution paths" in axis_user

    descriptor_messages = build_stage1_messages(
        dataset="gpqa",
        benchmark_family="science_multiple_choice",
        question="Which option is correct?",
        axes=[{"axis_id": "a", "name": "Axis A", "low_desc": "low", "high_desc": "high"}],
        sampled_points=[{"a": 0.1}, {"a": 0.9}],
    )
    descriptor_user = descriptor_messages[-1]["content"]
    assert "hard anchors" in descriptor_user
    assert "near-duplicate safe personas" in descriptor_user
    assert "support coverage" in descriptor_user

    stage2_messages = build_stage2_messages(
        question="Which option is correct?",
        descriptor={
            "persona_id": "persona_1",
            "name": "Verifier",
            "axis_interpretation": {"a": "checks before committing"},
            "short_rule": "verify before committing",
            "reasoning_summary": "checks critical steps before updating",
        },
    )
    stage2_user = stage2_messages[-1]["content"]
    assert "operating rules" in stage2_user
    assert "visible in execution" in stage2_user

    judge_messages = build_judge_messages(
        dataset="gpqa",
        benchmark_family="science_multiple_choice",
        question="Which option is correct?",
        mode="task_family_generated",
    )
    judge_user = judge_messages[-1]["content"]
    assert "transcript-grounded evidence" in judge_user
    assert "superficial consensus" in judge_user
