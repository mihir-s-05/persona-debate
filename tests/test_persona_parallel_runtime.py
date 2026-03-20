from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

from debate_v_majority.cli.persona_runtime import run_persona_generation, run_persona_generation_staged
from debate_v_majority.cli.stage_state import load_latest_stage_entry_of_type
from debate_v_majority.personas import Axis, AxisSelection, JudgeCard, PersonaCard, PersonaDescriptor
from debate_v_majority.cli.subset import SubsetItem


class _ParallelFakeEngine:
    model_name = "gemini-3-flash-preview"
    provider_name = "fake"

    def __init__(self) -> None:
        self.call_log: list[dict[str, Any]] = []

    def generate_batch(
        self,
        contexts,
        batch_size=None,
        sampling_kwargs=None,
        progress_callback=None,
        model_role=None,
    ):
        del batch_size, sampling_kwargs
        self.call_log.append({"model_role": model_role, "n_contexts": len(contexts)})
        outputs = []
        for ctx in contexts:
            prompt = str(ctx[-1].get("content", ""))
            all_content = " ".join(str(message.get("content", "")) for message in ctx)
            if "Generate the full persona population jointly" in all_content:
                outputs.append(
                    json.dumps(
                        {
                            "descriptors": [
                                {
                                    "persona_id": f"persona_{i + 1}",
                                    "name": f"P{i + 1}",
                                    "axis_interpretation": {"x": f"interp_{i}"},
                                    "short_rule": f"rule_{i}",
                                    "reasoning_summary": f"unique summary {i}",
                                }
                                for i in range(2)
                            ]
                        }
                    )
                )
            elif "Expand each descriptor into a compact" in prompt:
                pid = "persona_1"
                if "persona_2" in prompt:
                    pid = "persona_2"
                outputs.append(
                    json.dumps(
                        {
                            "persona_id": pid,
                            "title": f"Card {pid}",
                            "core_reasoning_strategy": f"strategy {pid}",
                            "priorities": ["p1"],
                            "distrusts": ["d1"],
                            "decomposition_style": "step",
                            "revision_policy": "revise",
                            "confidence_policy": "explicit",
                            "failure_mode_to_avoid": "none",
                            "system_prompt": f"System prompt for {pid}.",
                        }
                    )
                )
            else:
                outputs.append("{}")
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def _items() -> list[SubsetItem]:
    return [
        SubsetItem(
            subset_id=0,
            orig_id=0,
            item_uid="aime25:test_0",
            dataset_revision=None,
            item_display_id=0,
            raw_task={"question": "What is 2+2?", "answer": "4"},
            dataset_meta={},
        ),
        SubsetItem(
            subset_id=1,
            orig_id=1,
            item_uid="aime25:test_1",
            dataset_revision=None,
            item_display_id=1,
            raw_task={"question": "What is 3+3?", "answer": "6"},
            dataset_meta={},
        ),
    ]


def test_staged_persona_multi_item_parallel_pipeline_preserves_mapping(tmp_path: Path):
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    engine = _ParallelFakeEngine()
    items = _items()

    for stage in ("axes", "descriptors", "cards"):
        entry = run_persona_generation_staged(
            dataset="aime25",
            items=items,
            artifacts_dir=artifacts_dir,
            stage_state_file=state_file,
            persona_stage=stage,
            n_personas=2,
            persona_seed=0,
            axis_mode="fixed",
            fixed_axis_count=2,
            task_axis_count=0,
            sampling_method="maximin",
            judge_persona_mode="neutral_baseline",
            backend="llm",
            generator_model="fake-generator",
            judge_generator_model="fake-judge-generator",
            generator_engine=engine,
            judge_engine=engine,
            axes_file=None,
            summary_file=summary,
        )

    assert set(entry.persona_data) == {"aime25:test_0", "aime25:test_1"}
    for item_uid in entry.persona_data:
        cards = entry.persona_data[item_uid]["cards"]
        assert [card["title"] for card in cards] == ["Card persona_1", "Card persona_2"]


def test_full_persona_generation_multi_item_parallel_pipeline_preserves_mapping(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    engine = _ParallelFakeEngine()
    items = _items()

    rows = run_persona_generation(
        dataset="aime25",
        items=items,
        artifacts_dir=artifacts_dir,
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=2,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="neutral_baseline",
        backend="llm",
        generator_model="fake-generator",
        judge_generator_model="fake-judge-generator",
        generator_engine=engine,
        judge_engine=engine,
        axes_file=None,
        save_artifacts=False,
        replay=False,
        dump_cards=False,
        summary_file=summary,
    )

    assert [row["item_uid"] for row in rows] == ["aime25:test_0", "aime25:test_1"]
    for row in rows:
        assert [card["title"] for card in row["cards"]] == ["Card persona_1", "Card persona_2"]


def test_staged_persona_resume_preserves_completed_parallel_descriptor_work(tmp_path: Path, monkeypatch):
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    items = _items()
    call_counts = {"aime25:test_0": 0, "aime25:test_1": 0}

    def _fake_prepare_descriptor_generation(*, config, engine):
        del engine
        return (
            AxisSelection(
                mode="fixed",
                axes=[
                    Axis(
                        axis_id="x",
                        name="Axis X",
                        kind="fixed",
                        low_desc="low",
                        high_desc="high",
                    )
                ],
                benchmark_family=None,
                question_summary=config.question,
                generator_prompt_version="v1",
                generator_model="fake-generator",
            ),
            [{"x": 0.0}, {"x": 1.0}],
            "llm",
        )

    def _fake_generate_descriptors_from_state(*, config, axis_selection, points, engine):
        del axis_selection, points, engine
        item_uid = str(config.item_uid)
        call_counts[item_uid] += 1
        if item_uid == "aime25:test_1" and call_counts[item_uid] == 1:
            raise RuntimeError("descriptor failure")
        return (
            [
                PersonaDescriptor(
                    persona_id="persona_1",
                    name=f"{item_uid} p1",
                    axis_values={"x": 0.0},
                    axis_interpretation={"x": "steady"},
                    short_rule=f"rule {item_uid}",
                    reasoning_summary=f"reasoning {item_uid}",
                ),
                PersonaDescriptor(
                    persona_id="persona_2",
                    name=f"{item_uid} p2",
                    axis_values={"x": 1.0},
                    axis_interpretation={"x": "bold"},
                    short_rule=f"rule2 {item_uid}",
                    reasoning_summary=f"reasoning2 {item_uid}",
                ),
            ],
            {
                "validator_metadata": {
                    "descriptor_call_metadata": {"item_uid": item_uid},
                    "descriptor_validations": [],
                }
            },
        )

    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.prepare_descriptor_generation",
        _fake_prepare_descriptor_generation,
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.generate_descriptors_from_state",
        _fake_generate_descriptors_from_state,
    )

    run_persona_generation_staged(
        dataset="aime25",
        items=items,
        artifacts_dir=artifacts_dir,
        stage_state_file=state_file,
        persona_stage="axes",
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=2,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="neutral_baseline",
        backend="llm",
        generator_model="fake-generator",
        judge_generator_model="fake-judge-generator",
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        summary_file=summary,
    )

    try:
        run_persona_generation_staged(
            dataset="aime25",
            items=items,
            artifacts_dir=artifacts_dir,
            stage_state_file=state_file,
            persona_stage="descriptors",
            n_personas=2,
            persona_seed=0,
            axis_mode="fixed",
            fixed_axis_count=2,
            task_axis_count=0,
            sampling_method="maximin",
            judge_persona_mode="neutral_baseline",
            backend="llm",
            generator_model="fake-generator",
            judge_generator_model="fake-judge-generator",
            generator_engine=None,
            judge_engine=None,
            axes_file=None,
            summary_file=summary,
        )
        assert False, "expected descriptor failure"
    except RuntimeError as exc:
        assert str(exc) == "descriptor failure"

    partial_entry = load_latest_stage_entry_of_type(state_file, "persona")
    assert partial_entry is not None
    assert partial_entry.completed_stage == "axes"
    assert partial_entry.meta["active_stage"] == "descriptors"
    assert partial_entry.meta["active_stage_complete"] is False
    assert "descriptors" in partial_entry.persona_data["aime25:test_0"]
    assert "descriptors" not in partial_entry.persona_data["aime25:test_1"]

    entry = run_persona_generation_staged(
        dataset="aime25",
        items=items,
        artifacts_dir=artifacts_dir,
        stage_state_file=state_file,
        persona_stage="descriptors",
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=2,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="neutral_baseline",
        backend="llm",
        generator_model="fake-generator",
        judge_generator_model="fake-judge-generator",
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        summary_file=summary,
    )

    assert call_counts["aime25:test_0"] == 1
    assert call_counts["aime25:test_1"] == 2
    assert set(entry.persona_data) == {"aime25:test_0", "aime25:test_1"}
    assert entry.meta["active_stage"] == "descriptors"
    assert entry.meta["active_stage_complete"] is True


def test_full_persona_generation_resume_does_not_rerun_earlier_stages_after_judge_failure(
    tmp_path: Path,
    monkeypatch,
):
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    items = _items()
    stage_counts = {"axes": 0, "descriptors": 0, "cards": 0}
    judge_counts = {"aime25:test_0": 0, "aime25:test_1": 0}

    def _fake_prepare_descriptor_generation(*, config, engine):
        del engine
        stage_counts["axes"] += 1
        return (
            AxisSelection(
                mode="fixed",
                axes=[
                    Axis(
                        axis_id="x",
                        name="Axis X",
                        kind="fixed",
                        low_desc="low",
                        high_desc="high",
                    )
                ],
                benchmark_family=None,
                question_summary=config.question,
                generator_prompt_version="v1",
                generator_model="fake-generator",
            ),
            [{"x": 0.0}, {"x": 1.0}],
            "llm",
        )

    def _fake_generate_descriptors_from_state(*, config, axis_selection, points, engine):
        del axis_selection, points, engine
        stage_counts["descriptors"] += 1
        return (
            [
                PersonaDescriptor(
                    persona_id="persona_1",
                    name=f"{config.item_uid} p1",
                    axis_values={"x": 0.0},
                    axis_interpretation={"x": "steady"},
                    short_rule="rule 1",
                    reasoning_summary="reasoning 1",
                ),
                PersonaDescriptor(
                    persona_id="persona_2",
                    name=f"{config.item_uid} p2",
                    axis_values={"x": 1.0},
                    axis_interpretation={"x": "bold"},
                    short_rule="rule 2",
                    reasoning_summary="reasoning 2",
                ),
            ],
            {
                "validator_metadata": {
                    "descriptor_call_metadata": {"item_uid": config.item_uid},
                    "descriptor_validations": [],
                }
            },
        )

    def _fake_expand_cards(descriptors, *, dataset, question, raw_task, engine, backend):
        del dataset, question, raw_task, engine, backend
        stage_counts["cards"] += 1
        cards = [
            PersonaCard(
                persona_id=descriptor.persona_id,
                title=f"Card {descriptor.persona_id}",
                core_reasoning_strategy=f"strategy {descriptor.persona_id}",
                priorities=["p1"],
                distrusts=["d1"],
                decomposition_style="step",
                revision_policy="revise",
                confidence_policy="explicit",
                failure_mode_to_avoid="none",
                system_prompt=f"System prompt {descriptor.persona_id}",
                card_version="v1",
            )
            for descriptor in descriptors
        ]
        return cards, {"card_call_metadata": [{"count": len(cards)}], "card_validations": []}

    def _fake_resolve_runtime_judge_card(
        *,
        dataset,
        item,
        question,
        raw_task,
        persona_artifacts_dir,
        judge_bank_dir,
        judge_bank_refresh,
        gpqa_family_cache_path,
        judge_persona_mode,
        persona_backend,
        judge_generator_model,
        judge_engine,
    ):
        del (
            dataset,
            question,
            raw_task,
            persona_artifacts_dir,
            judge_bank_dir,
            judge_bank_refresh,
            gpqa_family_cache_path,
            judge_persona_mode,
            persona_backend,
            judge_generator_model,
            judge_engine,
        )
        judge_counts[item.item_uid] += 1
        if item.item_uid == "aime25:test_1" and judge_counts[item.item_uid] == 1:
            raise RuntimeError("judge failure")
        judge_card = JudgeCard(
            judge_id=f"judge_{item.item_uid}",
            judge_family="math",
            domain_scope="math",
            evaluation_priorities=["support"],
            tie_break_policy="prefer support",
            independent_resolve_policy="limited_check_only",
            answer_format_policy="strict",
            confidence_policy=None,
            system_prompt="Judge prompt",
            card_version="v1",
            source={"backend": "llm"},
        )
        return judge_card, {"judge_bank_path": f"{item.item_uid}.json"}

    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.prepare_descriptor_generation",
        _fake_prepare_descriptor_generation,
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.generate_descriptors_from_state",
        _fake_generate_descriptors_from_state,
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.expand_cards",
        _fake_expand_cards,
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime._resolve_runtime_judge_card",
        _fake_resolve_runtime_judge_card,
    )

    try:
        run_persona_generation(
            dataset="aime25",
            items=items,
            artifacts_dir=artifacts_dir,
            n_personas=2,
            persona_seed=0,
            axis_mode="fixed",
            fixed_axis_count=2,
            task_axis_count=0,
            sampling_method="maximin",
            judge_persona_mode="neutral_baseline",
            backend="llm",
            generator_model="fake-generator",
            judge_generator_model="fake-judge-generator",
            generator_engine=None,
            judge_engine=None,
            axes_file=None,
            save_artifacts=False,
            replay=False,
            dump_cards=False,
            summary_file=summary,
        )
        assert False, "expected judge failure"
    except RuntimeError as exc:
        assert str(exc) == "judge failure"

    assert stage_counts == {"axes": 2, "descriptors": 2, "cards": 2}
    assert judge_counts == {"aime25:test_0": 1, "aime25:test_1": 1}

    rows = run_persona_generation(
        dataset="aime25",
        items=items,
        artifacts_dir=artifacts_dir,
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=2,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="neutral_baseline",
        backend="llm",
        generator_model="fake-generator",
        judge_generator_model="fake-judge-generator",
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        save_artifacts=False,
        replay=False,
        dump_cards=False,
        summary_file=summary,
    )

    assert [row["item_uid"] for row in rows] == ["aime25:test_0", "aime25:test_1"]
    assert stage_counts == {"axes": 2, "descriptors": 2, "cards": 2}
    assert judge_counts == {"aime25:test_0": 1, "aime25:test_1": 2}

