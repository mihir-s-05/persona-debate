"""Tests for staged execution: stage_state, staged persona generation, and debate stop/resume."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from debate_v_majority.cli.stage_state import (
    STAGE_ENTRY_SCHEMA_VERSION,
    STAGE_TRACE_SCHEMA_VERSION,
    StageEntry,
    append_stage_entry,
    load_all_stage_entries,
    load_latest_stage_entry,
    load_latest_stage_entry_of_type,
    load_stage_entry_by_name,
    make_stage_entry,
)
from debate_v_majority.personas.generator import (
    build_card_messages,
    build_descriptor_messages,
    parse_card_result,
    parse_descriptor_result,
)
from debate_v_majority.personas.schema import (
    Axis,
    AxisSelection,
    JudgeCard,
    PersonaArtifact,
    PersonaCard,
    PersonaDescriptor,
    PersonaGenerationConfig,
)
from debate_v_majority.personas.generator import MAX_GENERATION_RETRIES
from debate_v_majority.personas import load_artifact as load_persona_artifact


# ---------------------------------------------------------------------------
# StageEntry unit tests
# ---------------------------------------------------------------------------


def test_stage_entry_roundtrip():
    entry = make_stage_entry(
        stage_type="persona",
        completed_stage="axes",
        dataset="aime25",
        items=[{"item_uid": "aime25:1"}],
        persona_data={"aime25:1": {"axis_selection": {"mode": "fixed"}}},
    )
    as_dict = entry.to_dict()
    restored = StageEntry.from_dict(as_dict)
    assert restored.schema_version == STAGE_ENTRY_SCHEMA_VERSION
    assert restored.stage_type == "persona"
    assert restored.completed_stage == "axes"
    assert restored.dataset == "aime25"
    assert len(restored.items) == 1
    assert restored.persona_data["aime25:1"]["axis_selection"]["mode"] == "fixed"
    assert restored.created_at


def test_append_and_load(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    e1 = make_stage_entry(
        stage_type="persona", completed_stage="axes",
        dataset="aime25", items=[],
    )
    e2 = make_stage_entry(
        stage_type="persona", completed_stage="descriptors",
        dataset="aime25", items=[],
    )
    append_stage_entry(path, e1)
    append_stage_entry(path, e2)
    all_entries = load_all_stage_entries(path)
    assert len(all_entries) == 2
    assert all_entries[0].completed_stage == "axes"
    assert all_entries[1].completed_stage == "descriptors"
    latest = load_latest_stage_entry(path)
    assert latest.completed_stage == "descriptors"


def test_append_stage_entry_writes_new_v2_trace_file_per_stage(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    e1 = make_stage_entry(
        stage_type="persona",
        completed_stage="axes",
        dataset="aime25",
        items=[{"item_uid": "aime25:1"}],
        persona_data={
            "aime25:1": {
                "axis_selection": {"mode": "hybrid", "axes": [{"axis_id": "a1", "name": "Axis 1", "kind": "fixed"}]},
                "sampled_points": [{"a1": 0.1}],
            }
        },
    )
    e2 = make_stage_entry(
        stage_type="persona",
        completed_stage="descriptors",
        dataset="aime25",
        items=[{"item_uid": "aime25:1"}],
        persona_data={
            "aime25:1": {
                "descriptors": [
                    {
                        "persona_id": "persona_1",
                        "name": "Tester",
                        "short_rule": "Rule",
                        "reasoning_summary": "Summary",
                    }
                ]
            }
        },
    )

    append_stage_entry(path, e1)
    append_stage_entry(path, e2)

    trace_files = sorted(tmp_path.glob("state.trace.*.v2.*.md"))
    assert len(trace_files) == 2
    assert trace_files[0] != trace_files[1]
    assert STAGE_TRACE_SCHEMA_VERSION in trace_files[0].read_text(encoding="utf-8")
    assert "Completed stage: `axes`" in trace_files[0].read_text(encoding="utf-8")
    assert "Completed stage: `descriptors`" in trace_files[1].read_text(encoding="utf-8")


def test_append_stage_entry_writes_human_readable_debate_trace(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    entry = make_stage_entry(
        stage_type="debate",
        completed_stage="round_0",
        dataset="aime25",
        items=[{"item_uid": "aime25:1"}],
        debate_data={
            "n_agents": 2,
            "n_rounds": 3,
            "judge_rounds": [3],
            "persona_artifacts": [
                {
                    "cards": [
                        {"title": "Persona One"},
                        {"title": "Persona Two"},
                    ]
                }
            ],
            "agent_round_outputs_by_q": [
                [
                    [
                        {
                            "final_answer": "240",
                            "visible_output": "First agent full visible output",
                            "public_rationale": "First agent rationale",
                            "thought_summary": "First agent summary",
                            "scoring_result": {"correct": 1},
                        }
                    ],
                    [
                        {
                            "final_answer": "188",
                            "visible_output": "Second agent full visible output",
                            "public_rationale": "Second agent rationale",
                            "thought_summary": "Second agent summary",
                            "scoring_result": {"correct": 0},
                        }
                    ],
                ]
            ],
            "results_by_round": {},
        },
    )

    append_stage_entry(path, entry)

    trace_files = list(tmp_path.glob("state.trace.debate.round_0.v2.*.md"))
    assert len(trace_files) == 1
    trace_text = trace_files[0].read_text(encoding="utf-8")
    assert "Debater Outputs For Round `0`" in trace_text
    assert "Agent 1: Persona One" in trace_text
    assert "Final answer: `240`" in trace_text
    assert "First agent full visible output" in trace_text
    assert "First agent rationale" in trace_text


def test_append_stage_entry_recovers_from_missing_trailing_newline(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    first = make_stage_entry(
        stage_type="persona",
        completed_stage="axes",
        dataset="aime25",
        items=[],
    )
    second = make_stage_entry(
        stage_type="persona",
        completed_stage="descriptors",
        dataset="aime25",
        items=[],
    )
    path.write_text(json.dumps(first.to_dict(), ensure_ascii=False), encoding="utf-8")

    append_stage_entry(path, second)

    raw = path.read_text(encoding="utf-8")
    assert '}\n{' in raw
    entries = load_all_stage_entries(path)
    assert [entry.completed_stage for entry in entries] == ["axes", "descriptors"]


def test_load_by_name(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    for stage in ("axes", "descriptors", "cards"):
        append_stage_entry(
            path,
            make_stage_entry(
                stage_type="persona", completed_stage=stage,
                dataset="aime25", items=[],
            ),
        )
    entry = load_stage_entry_by_name(path, "descriptors")
    assert entry.completed_stage == "descriptors"


def test_load_by_name_missing(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    append_stage_entry(
        path,
        make_stage_entry(
            stage_type="persona", completed_stage="axes",
            dataset="aime25", items=[],
        ),
    )
    with pytest.raises(ValueError, match="No stage entry named"):
        load_stage_entry_by_name(path, "cards")


def test_load_latest_empty(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_latest_stage_entry(path)


def test_load_latest_by_type(tmp_path: Path):
    path = tmp_path / "state.jsonl"
    append_stage_entry(
        path,
        make_stage_entry(stage_type="persona", completed_stage="axes", dataset="aime25", items=[]),
    )
    append_stage_entry(
        path,
        make_stage_entry(stage_type="debate", completed_stage="round_0", dataset="aime25", items=[]),
    )
    latest_persona = load_latest_stage_entry_of_type(path, "persona")
    latest_debate = load_latest_stage_entry_of_type(path, "debate")
    assert latest_persona is not None
    assert latest_persona.completed_stage == "axes"
    assert latest_debate is not None
    assert latest_debate.completed_stage == "round_0"


# ---------------------------------------------------------------------------
# Generator build/parse helpers
# ---------------------------------------------------------------------------


def _sample_config() -> PersonaGenerationConfig:
    return PersonaGenerationConfig(
        dataset="aime25",
        question="What is 2+2?",
        raw_task={"question": "What is 2+2?"},
        item_uid="aime25:1",
        item_display_id=1,
        dataset_revision=None,
        n_personas=2,
    )


def _sample_axis_selection() -> AxisSelection:
    return AxisSelection(
        mode="fixed",
        axes=[
            Axis(
                axis_id="symbolic_vs_intuitive",
                name="Symbolic vs Intuitive",
                kind="fixed",
                low_desc="symbolic",
                high_desc="intuitive",
            )
        ],
        benchmark_family="competition_math",
        question_summary="What is 2+2?",
        generator_prompt_version="v0",
    )


def _sample_persona_artifact(*, item_uid: str, n_personas: int = 1) -> PersonaArtifact:
    descriptors = [
        PersonaDescriptor(
            persona_id=f"persona_{idx + 1}",
            name=f"Persona {idx + 1}",
            axis_values={"symbolic_vs_intuitive": 0.25 + (0.5 * idx)},
            axis_interpretation={"symbolic_vs_intuitive": "balanced"},
            short_rule="verify the key inference before committing",
            reasoning_summary=f"reasoning summary {idx + 1}",
        )
        for idx in range(n_personas)
    ]
    cards = [
        PersonaCard(
            persona_id=descriptor.persona_id,
            title=descriptor.name,
            core_reasoning_strategy=descriptor.reasoning_summary,
            priorities=[descriptor.short_rule],
            distrusts=["unsupported leaps"],
            decomposition_style="stepwise",
            revision_policy="revise only on concrete evidence",
            confidence_policy="be explicit",
            failure_mode_to_avoid="answer-first reasoning",
            system_prompt=f"System prompt for {descriptor.persona_id}.",
            card_version="v0",
        )
        for descriptor in descriptors
    ]
    return PersonaArtifact(
        artifact_version="v0",
        dataset="aime25",
        item_uid=item_uid,
        dataset_revision=None,
        item_display_id=0,
        persona_seed=0,
        generator_model="fake-generator",
        judge_generator_model="fake-judge-generator",
        axes=_sample_axis_selection(),
        sampled_points=[descriptor.axis_values for descriptor in descriptors],
        descriptors=descriptors,
        cards=cards,
        judge_card=JudgeCard(
            judge_id="judge_1",
            judge_family="neutral",
            domain_scope="math",
            evaluation_priorities=["accuracy"],
            tie_break_policy="prefer the more supported answer",
            independent_resolve_policy="resolve independently before comparing",
            answer_format_policy="boxed final answer",
            confidence_policy="state uncertainty",
            system_prompt="You are a careful judge.",
            card_version="v0",
        ),
        prompt_versions={"axis": "v0", "descriptor": "v0", "card": "v0", "judge": "v0"},
        created_at="2026-03-11T12:00:00+00:00",
    )


def test_build_descriptor_messages_returns_list():
    config = _sample_config()
    axis_selection = _sample_axis_selection()
    points = [{"symbolic_vs_intuitive": 0.2}, {"symbolic_vs_intuitive": 0.8}]
    msgs = build_descriptor_messages(
        config=config, axis_selection=axis_selection, points=points,
    )
    assert isinstance(msgs, list)
    assert len(msgs) > 0
    assert msgs[0]["role"] in ("system", "user")


def test_build_descriptor_messages_sanitizes_task_axis_descriptions():
    config = _sample_config()
    axis_selection = AxisSelection(
        mode="hybrid",
        benchmark_family="competition_math",
        question_summary="question summary",
        generator_prompt_version="phase0.axes.v3",
        axes=[
            Axis(
                axis_id="symbolic_vs_intuitive",
                name="Symbolic vs Intuitive",
                kind="fixed",
                low_desc="Lean on equations.",
                high_desc="Lean on intuition.",
            ),
            Axis(
                axis_id="differentiation_methodology",
                name="Differentiation Methodology",
                kind="task",
                low_desc="Uses standard product rules to differentiate f(x).",
                high_desc="Uses 1/(x-18) + 1/(x-72) = 1/x.",
                notes="Specific constants from the item.",
            ),
        ],
    )
    points = [{"symbolic_vs_intuitive": 0.2, "differentiation_methodology": 0.8}]
    msgs = build_descriptor_messages(
        config=config, axis_selection=axis_selection, points=points,
    )
    user_content = str(msgs[-1]["content"])
    assert "1/(x-18)" not in user_content
    assert "Specific constants from the item." not in user_content
    assert "Interpret this axis abstractly from its name only." in user_content
    assert "What is 2+2?" not in user_content
    assert "Question:" not in user_content


def test_parse_descriptor_result():
    payload = json.dumps({
        "descriptors": [
            {
                "persona_id": "persona_1",
                "name": "A",
                "axis_interpretation": {"x": "y"},
                "short_rule": "rule",
                "reasoning_summary": "summary A unique",
            },
            {
                "persona_id": "persona_2",
                "name": "B",
                "axis_interpretation": {"x": "z"},
                "short_rule": "rule2",
                "reasoning_summary": "summary B unique",
            },
        ]
    })
    points = [{"x": 0.3}, {"x": 0.7}]
    descriptors = parse_descriptor_result(payload, n_personas=2, points=points)
    assert len(descriptors) == 2
    assert descriptors[0].persona_id == "persona_1"
    assert descriptors[1].axis_values == {"x": 0.7}


def test_parse_descriptor_result_too_few():
    payload = json.dumps({"descriptors": [{"persona_id": "p1"}]})
    with pytest.raises(ValueError, match="too few rows"):
        parse_descriptor_result(payload, n_personas=3, points=[{}, {}, {}])


def test_build_card_messages_returns_list():
    desc = PersonaDescriptor(
        persona_id="persona_1", name="Test",
        axis_values={}, axis_interpretation={},
        short_rule="rule", reasoning_summary="summary",
    )
    msgs = build_card_messages(
        descriptor=desc, question="What is 2+2?", question_media=None,
    )
    assert isinstance(msgs, list)
    assert len(msgs) > 0


def test_parse_card_result():
    desc = PersonaDescriptor(
        persona_id="persona_1", name="Test",
        axis_values={}, axis_interpretation={},
        short_rule="rule", reasoning_summary="summary",
    )
    payload = json.dumps({
        "persona_id": "persona_1",
        "title": "Generated Card",
        "core_reasoning_strategy": "test strategy",
        "priorities": ["a"],
        "distrusts": ["b"],
        "decomposition_style": "step by step",
        "revision_policy": "revise on evidence",
        "confidence_policy": "be explicit",
        "failure_mode_to_avoid": "no leaks",
        "system_prompt": "You are persona_1.",
    })
    card = parse_card_result(payload, descriptor=desc)
    assert card.persona_id == "persona_1"
    assert card.title == "Generated Card"
    assert card.system_prompt == "You are persona_1."


# ---------------------------------------------------------------------------
# Debate stop-after
# ---------------------------------------------------------------------------


def test_debate_stopped_exception_carries_results():
    from debate_v_majority.cli.debate_runner import _DebateStopped

    results = {0: [{"answer": "x"}]}
    exc = _DebateStopped(results)
    assert exc.results_by_round == results


# ---------------------------------------------------------------------------
# Staged persona generation (integration-level with fake engine)
# ---------------------------------------------------------------------------


class _FakeEngine:
    model_name = "gemini-3-flash-preview"
    provider_name = "fake"

    def __init__(self):
        self.call_log: list[dict[str, Any]] = []

    def generate_batch(
        self, contexts, batch_size=None, sampling_kwargs=None,
        progress_callback=None, model_role=None,
    ):
        self.call_log.append(
            {
                "model_role": model_role,
                "batch_size": len(contexts),
            }
        )
        outputs = []
        for ctx in contexts:
            all_content = " ".join(str(m.get("content", "")) for m in ctx)
            prompt = str(ctx[-1].get("content", ""))
            if "Generate the full persona population jointly" in all_content:
                outputs.append(json.dumps({
                    "descriptors": [
                        {
                            "persona_id": f"persona_{i+1}",
                            "name": f"P{i+1}",
                            "axis_interpretation": {"x": f"interp_{i}"},
                            "short_rule": f"rule_{i}",
                            "reasoning_summary": f"unique summary {i}",
                        }
                        for i in range(5)
                    ]
                }))
            elif "Expand each descriptor into a compact" in prompt:
                pid = "persona_1"
                for i in range(5):
                    if f"persona_{i+1}" in prompt:
                        pid = f"persona_{i+1}"
                        break
                outputs.append(json.dumps({
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
                }))
            else:
                outputs.append("{}")
        return outputs


def test_staged_persona_axes_only(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    entry = run_persona_generation_staged(
        dataset="aime25",
        items=[item],
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
        backend="heuristic",
        generator_model=None,
        judge_generator_model=None,
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        summary_file=summary,
    )
    assert entry.completed_stage == "axes"
    assert state_file.exists()
    lines = [l for l in state_file.read_text().strip().split("\n") if l.strip()]
    assert len(lines) == 1
    loaded = load_latest_stage_entry(state_file)
    assert loaded.completed_stage == "axes"
    assert "aime25:test_0" in loaded.persona_data


def test_staged_persona_resume_from_axes(tmp_path: Path):
    """Run axes stage, then resume and run descriptors stage."""
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir, stage_state_file=state_file,
        persona_stage="axes", n_personas=2, persona_seed=0,
        axis_mode="fixed", fixed_axis_count=2, task_axis_count=0,
        sampling_method="maximin", judge_persona_mode="neutral_baseline",
        backend="heuristic", generator_model=None,
        judge_generator_model=None, generator_engine=None,
        judge_engine=None, axes_file=None, summary_file=summary,
    )

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir, stage_state_file=state_file,
        persona_stage="descriptors", n_personas=2, persona_seed=0,
        axis_mode="fixed", fixed_axis_count=2, task_axis_count=0,
        sampling_method="maximin", judge_persona_mode="neutral_baseline",
        backend="heuristic", generator_model=None,
        judge_generator_model=None, generator_engine=None,
        judge_engine=None, axes_file=None, summary_file=summary,
    )
    assert entry.completed_stage == "descriptors"
    all_entries = load_all_stage_entries(state_file)
    assert len(all_entries) == 2
    item_data = entry.persona_data["aime25:test_0"]
    assert "descriptors" in item_data
    assert len(item_data["descriptors"]) == 2


def test_staged_persona_full_heuristic_pipeline(tmp_path: Path):
    """Run all four stages sequentially through heuristic backend."""
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    for stage in ("axes", "descriptors", "cards", "judge_card"):
        entry = run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir, stage_state_file=state_file,
            persona_stage=stage, n_personas=2, persona_seed=0,
            axis_mode="fixed", fixed_axis_count=2, task_axis_count=0,
            sampling_method="maximin", judge_persona_mode="neutral_baseline",
            backend="heuristic", generator_model=None,
            judge_generator_model=None, generator_engine=None,
            judge_engine=None, axes_file=None, summary_file=summary,
        )

    assert entry.completed_stage == "judge_card"
    all_entries = load_all_stage_entries(state_file)
    assert len(all_entries) == 4
    stages = [e.completed_stage for e in all_entries]
    assert stages == ["axes", "descriptors", "cards", "judge_card"]
    item_data = entry.persona_data["aime25:test_0"]
    assert "axis_selection" in item_data
    assert "descriptors" in item_data
    assert "cards" in item_data
    assert "judge_card" in item_data


def test_staged_persona_judge_card_saves_artifact(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    for stage in ("axes", "descriptors", "cards", "judge_card"):
        entry = run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage=stage, n_personas=2,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="heuristic",
            generator_model=None, judge_generator_model=None,
            generator_engine=None, judge_engine=None, axes_file=None,
            save_artifacts=True,
            summary_file=summary,
        )

    item_data = entry.persona_data["aime25:test_0"]
    artifact_path = Path(item_data["artifact_path"])
    assert artifact_path.exists()
    artifact = load_persona_artifact(path=artifact_path)
    assert artifact.item_uid == "aime25:test_0"
    assert len(artifact.cards) == 2
    assert artifact.judge_card is not None


def test_staged_persona_llm_pipeline_with_fake_engine(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    fake_engine = _FakeEngine()

    for stage in ("axes", "descriptors", "cards", "judge_card"):
        entry = run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage=stage, n_personas=2,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="llm",
            generator_model="fake-generator", judge_generator_model=None,
            generator_engine=fake_engine, judge_engine=None, axes_file=None,
            save_artifacts=True,
            summary_file=summary,
        )

    item_data = entry.persona_data["aime25:test_0"]
    assert [call["model_role"] for call in fake_engine.call_log] == ["generator", "generator", "generator"]
    assert [call["batch_size"] for call in fake_engine.call_log] == [1, 1, 1]
    assert len(item_data["descriptors"]) == 2
    assert len(item_data["cards"]) == 2
    assert item_data["cards"][0]["title"] == "Card persona_1"
    assert item_data["cards"][1]["title"] == "Card persona_2"
    assert item_data["descriptor_validator_metadata"]["descriptor_call_metadata"] is not None
    assert len(item_data["card_metadata"]["card_call_metadata"]) == 2
    artifact = load_persona_artifact(path=Path(item_data["artifact_path"]))
    assert [card.title for card in artifact.cards] == ["Card persona_1", "Card persona_2"]
    assert artifact.judge_card is not None


def test_staged_persona_descriptors_preserve_validator_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    statuses = iter(["retry", "accept", "accept", "accept"])

    def _fake_validate_descriptor(_descriptor):
        status = next(statuses)
        from debate_v_majority.personas.schema import ValidationResult

        return ValidationResult(status=status, reasons=[f"status={status}"])

    monkeypatch.setattr("debate_v_majority.personas.generator.validate_descriptor", _fake_validate_descriptor)

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="descriptors", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    meta = entry.persona_data["aime25:test_0"]["descriptor_validator_metadata"]
    assert meta["descriptor_validations"][-1]["attempt"] == 1
    assert meta["descriptor_duplicates"] == []


def test_staged_persona_descriptors_retry_after_reject_hard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem
    from debate_v_majority.personas.schema import ValidationResult

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()
    statuses = iter(["reject_hard", "accept", "accept", "accept"])

    def _fake_validate_descriptor(_descriptor):
        status = next(statuses)
        return ValidationResult(status=status, reasons=[f"status={status}"])

    monkeypatch.setattr("debate_v_majority.personas.generator.validate_descriptor", _fake_validate_descriptor)

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="descriptors", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    meta = entry.persona_data["aime25:test_0"]["descriptor_validator_metadata"]
    assert meta["descriptor_validations"][-1]["attempt"] == 1
    assert meta["descriptor_validations"][0]["status"] == "accept"


def test_staged_persona_descriptor_failure_writes_audit_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem
    from debate_v_majority.personas.schema import ValidationResult

    class _DescriptorFailEngine:
        model_name = "fake-generator"
        provider_name = "fake"

        def generate_batch(self, contexts, batch_size=1, sampling_kwargs=None, progress_callback=None, model_role=None):
            outputs = []
            for ctx in contexts:
                all_content = " ".join(str(m.get("content", "")) for m in ctx)
                if "Generate the full persona population jointly" in all_content:
                    outputs.append(
                        json.dumps(
                            {
                                "descriptors": [
                                    {
                                        "persona_id": "persona_1",
                                        "name": "Leaky",
                                        "axis_interpretation": {"symbolic_vs_intuitive": "leans symbolic"},
                                        "short_rule": "the answer is 240",
                                        "reasoning_summary": "the answer is 240",
                                    }
                                ]
                            }
                        )
                    )
                else:
                    outputs.append(json.dumps({"axes": []}))
            if progress_callback is not None:
                progress_callback(len(outputs))
            return outputs

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    def _always_reject(_descriptor):
        return ValidationResult(status="reject_hard", reasons=["contains answer-oriented leakage indicators"])

    monkeypatch.setattr("debate_v_majority.personas.generator.validate_descriptor", _always_reject)

    with pytest.raises(ValueError, match="Descriptor generation exhausted retries"):
        run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage="descriptors", n_personas=1,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=1,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="llm",
            generator_model="fake-generator", judge_generator_model=None,
            generator_engine=_DescriptorFailEngine(), judge_engine=None, axes_file=None,
            summary_file=summary,
        )

    audit_path = tmp_path / "state.persona_failures.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["failed_stage"] == "descriptors"
    assert rows[0]["error_type"] == "GenerationExhaustedError"
    attempt_audits = rows[0]["failure_metadata"]["validator_metadata"]["attempt_audits"]
    assert len(attempt_audits) == MAX_GENERATION_RETRIES + 1
    assert "the answer is 240" in attempt_audits[-1]["raw_result_text"]


def test_staged_persona_cards_preserve_duplicate_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem
    from debate_v_majority.personas.schema import PersonaCard

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    def _duplicate_card(descriptor):
        return PersonaCard(
            persona_id=descriptor.persona_id,
            title=descriptor.name,
            core_reasoning_strategy=descriptor.reasoning_summary,
            priorities=[descriptor.short_rule],
            distrusts=["unsupported leaps"],
            decomposition_style=descriptor.short_rule,
            revision_policy="revise on evidence",
            confidence_policy="be explicit",
            failure_mode_to_avoid="avoid duplication",
            system_prompt="Shared operational prompt",
            card_version="v0",
        )

    monkeypatch.setattr("debate_v_majority.personas.generator._heuristic_card", _duplicate_card)

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="cards", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    cards = entry.persona_data["aime25:test_0"]["cards"]
    card_meta = entry.persona_data["aime25:test_0"]["card_metadata"]
    assert card_meta["card_duplicates"] == []
    assert cards[0]["system_prompt"] == "Shared operational prompt"
    assert cards[1]["system_prompt"] != "Shared operational prompt"


def test_staged_persona_replay_uses_saved_artifact_without_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.persona_runtime import _resolve_persona_artifact, run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    artifacts_dir = tmp_path / "artifacts"
    state_file = tmp_path / "state.jsonl"
    summary = StringIO()
    question = "What is 2+2?"
    raw_task = {"question": "What is 2+2?", "answer": "4"}
    artifact, artifact_path = _resolve_persona_artifact(
        dataset="aime25",
        item=item,
        question=question,
        raw_task=raw_task,
        artifacts_dir=artifacts_dir,
        n_personas=2,
        persona_seed=0,
        axis_mode="fixed",
        fixed_axis_count=2,
        task_axis_count=0,
        sampling_method="maximin",
        judge_persona_mode="neutral_baseline",
        backend="heuristic",
        generator_model=None,
        judge_generator_model=None,
        generator_engine=None,
        judge_engine=None,
        axes_file=None,
        save_artifact=True,
        replay=False,
    )

    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.prepare_descriptor_generation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected regeneration")),
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.generate_descriptors_from_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected regeneration")),
    )
    monkeypatch.setattr(
        "debate_v_majority.cli.persona_runtime.expand_cards",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected regeneration")),
    )

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="judge_card", n_personas=2,
        persona_seed=0, axis_mode="replay", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        replay=True,
        summary_file=summary,
    )

    entries = load_all_stage_entries(state_file)
    assert [row.completed_stage for row in entries] == ["axes", "descriptors", "cards", "judge_card"]
    item_data = entry.persona_data["aime25:test_0"]
    assert item_data["artifact"] == artifact.to_dict()
    assert item_data["artifact_path"] == str(artifact_path)


def test_staged_persona_completed_stage_is_noop(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    for stage in ("axes", "descriptors", "cards", "judge_card"):
        run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage=stage, n_personas=2,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="heuristic",
            generator_model=None, judge_generator_model=None,
            generator_engine=None, judge_engine=None, axes_file=None,
            summary_file=summary,
        )

    entries_before = load_all_stage_entries(state_file)
    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="judge_card", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )
    entries_after = load_all_stage_entries(state_file)
    assert len(entries_after) == len(entries_before)
    assert entry.completed_stage == "judge_card"


def test_staged_persona_completed_stage_is_noop_before_replay_lookup(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    summary = StringIO()
    append_stage_entry(
        state_file,
        make_stage_entry(
            stage_type="persona",
            completed_stage="judge_card",
            dataset="aime25",
            items=[asdict(item)],
            persona_data={"aime25:test_0": {"artifact": {"item_uid": "aime25:test_0"}}},
                meta={
                    "resume_settings": {
                        "n_personas": 2,
                        "persona_seed": 0,
                        "axis_mode": "replay",
                    "fixed_axis_count": 2,
                    "task_axis_count": 0,
                    "sampling_method": "maximin",
                    "judge_persona_mode": "neutral_baseline",
                    "backend": "heuristic",
                        "generator_model": None,
                        "judge_generator_model": None,
                        "axes_file": None,
                        "effective_backend": "heuristic",
                    }
                },
            ),
        )

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=tmp_path / "artifacts",
        stage_state_file=state_file, persona_stage="judge_card", n_personas=2,
        persona_seed=0, axis_mode="replay", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        replay=True, summary_file=summary,
    )

    assert entry.completed_stage == "judge_card"


def test_staged_persona_resume_rejects_mismatched_generation_settings(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="axes", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    with pytest.raises(ValueError, match="generation settings mismatch"):
        run_persona_generation_staged(
            dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage="judge_card", n_personas=5,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="heuristic",
            generator_model=None, judge_generator_model=None,
            generator_engine=None, judge_engine=None, axes_file=None,
            summary_file=summary,
        )


def test_staged_persona_resume_rejects_mismatched_items(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item_a = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    item_b = SubsetItem(
        subset_id=1, orig_id=1, item_uid="aime25:test_1",
        dataset_revision=None, item_display_id=1,
        raw_task={"question": "What is 3+3?", "answer": "6"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    run_persona_generation_staged(
        dataset="aime25", items=[item_a], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="axes", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    with pytest.raises(ValueError, match="items do not match"):
        run_persona_generation_staged(
            dataset="aime25", items=[item_b], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage="descriptors", n_personas=2,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="heuristic",
            generator_model=None, judge_generator_model=None,
            generator_engine=None, judge_engine=None, axes_file=None,
            summary_file=summary,
        )


# ---------------------------------------------------------------------------
# Full-mode persona generation still works (regression)
# ---------------------------------------------------------------------------


def test_full_persona_generation_unchanged(tmp_path: Path):
    """The original non-staged run_persona_generation still works."""
    from debate_v_majority.cli.persona_runtime import run_persona_generation
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    summary = StringIO()

    rows = run_persona_generation(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        n_personas=2, persona_seed=0, axis_mode="fixed",
        fixed_axis_count=2, task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        save_artifacts=False, replay=False, dump_cards=False,
        summary_file=summary,
    )
    assert len(rows) == 1
    assert rows[0]["item_uid"] == "aime25:test_0"
    assert len(rows[0]["cards"]) == 2


# ---------------------------------------------------------------------------
# Debate stop-after integration
# ---------------------------------------------------------------------------


def test_debate_append_state(tmp_path: Path):
    """_debate_append_state writes a valid JSONL line."""
    from debate_v_majority.cli.debate_runner import _debate_append_state
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "x", "answer": "y"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    _debate_append_state(
        stage_state_file=state_file,
        completed_stage="round_0",
        dataset="aime25",
        items=[item],
        all_contexts=[[
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "\\boxed{4}"},
            ],
        ]],
        agent_round_outputs_by_q=[[
            [{"final_answer": "4", "raw_response": "4"}],
        ]],
        results_by_round={},
        prev_judge_by_q=[None],
        round1_majority_by_q=[None],
        persona_artifacts=[None],
        persona_artifact_paths=[None],
        runtime_judge_cards=[None],
        runtime_judge_bank_meta=[None],
        persona_settings={
            "use_personas": False,
            "runtime_judge_persona_enabled": False,
            "persona_seed": 0,
            "persona_axis_mode": "fixed",
            "persona_fixed_axis_count": 2,
            "persona_task_axis_count": 0,
            "persona_sampling_method": "maximin",
            "persona_judge_mode": "neutral_baseline",
            "persona_backend": "heuristic",
            "generator_model": None,
            "judge_generator_model": None,
            "persona_axes_file": None,
            "judge_bank_dir": None,
            "judge_bank_refresh": False,
            "gpqa_family_cache_path": None,
        },
        runtime_settings={
            "debater_model": "fake-model",
            "debater_backend": "fake",
            "judge_model": "fake-model",
            "judge_backend": "fake",
            "judge_block_size": None,
            "judge_sampling_kwargs": {},
            "judge_strict_final_only": True,
            "judge_recovery_parse_enabled": True,
            "judge_trace_mode": "visible_plus_thought_summary",
            "public_rationale_max_tokens": 96,
        },
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
    )
    assert state_file.exists()
    entry = load_latest_stage_entry(state_file)
    assert entry.stage_type == "debate"
    assert entry.completed_stage == "round_0"
    assert entry.debate_data["n_agents"] == 1
    assert entry.debate_data["judge_rounds"] == [1]
    assert entry.debate_data["persona_settings"]["use_personas"] is False


class _ResumeEngine:
    model_name = "fake-model"
    provider_name = "fake"

    def __init__(self):
        self.call_log: list[dict[str, Any]] = []

    def generate_batch(
        self, contexts, batch_size=None, sampling_kwargs=None,
        progress_callback=None, model_role=None,
    ):
        self.call_log.append(
            {
                "model_role": model_role,
                "last_messages": [str(ctx[-1].get("content", "")) for ctx in contexts],
                "sampling_kwargs": None if sampling_kwargs is None else dict(sampling_kwargs),
            }
        )
        if progress_callback is not None:
            progress_callback(len(contexts))
        if model_role == "judge":
            return ["Decision: \\boxed{4}" for _ in contexts]
        outputs = []
        for ctx in contexts:
            latest = str(ctx[-1].get("content", ""))
            if "These are the solutions to the problem from other agents:" in latest:
                outputs.append("Updated solve. Final answer \\boxed{4}")
            else:
                outputs.append("Initial solve. Final answer \\boxed{4}")
        return outputs

    def count_prompt_tokens(self, messages):
        return 64

    def shutdown(self):
        return None


def test_debate_uses_precomputed_personas_without_regenerating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from debate_v_majority.cli.debate_runner import run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    artifact = _sample_persona_artifact(item_uid=item.item_uid)

    def _unexpected_resolve(**kwargs):
        raise AssertionError("run_debate should reuse the provided persona artifact")

    monkeypatch.setattr("debate_v_majority.cli.debate_runner._resolve_persona_artifact", _unexpected_resolve)

    results = run_debate(
        dataset="aime25",
        items=[item],
        engine=_ResumeEngine(),
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=None,
        judge_engine=_ResumeEngine(),
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_artifacts_by_item={item.item_uid: artifact},
        progress_file=StringIO(),
    )

    assert len(results[1]) == 1
    assert results[1][0]["persona_ids"] == ["persona_1"]


def test_debate_defaults_debater_max_output_tokens_to_32768():
    from debate_v_majority.cli.debate_runner import run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    engine = _ResumeEngine()
    results = run_debate(
        dataset="aime25",
        items=[item],
        engine=engine,
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=None,
        judge_engine=engine,
        progress_file=StringIO(),
    )

    assert len(results[1]) == 1
    debater_calls = [call for call in engine.call_log if call["model_role"] == "debater"]
    assert debater_calls
    assert all(call["sampling_kwargs"]["max_tokens"] == 32768 for call in debater_calls)


def test_debate_resume_after_round_0(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    first_engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=first_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=first_engine,
            progress_file=StringIO(),
            debate_stop_after="round_0",
            stage_state_file=state_file,
        )

    resume_engine = _ResumeEngine()
    results = run_debate(
        dataset="aime25",
        items=[item],
        engine=resume_engine,
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=None,
        judge_engine=resume_engine,
        progress_file=StringIO(),
        stage_state_file=state_file,
    )

    entries = load_all_stage_entries(state_file)
    assert [entry.completed_stage for entry in entries] == ["round_0", "round_1", "round_1_judge"]
    assert [call["model_role"] for call in resume_engine.call_log] == ["debater", "judge"]
    assert len(results[1]) == 1


def test_debate_resume_pending_judge_step(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    first_engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=first_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=first_engine,
            progress_file=StringIO(),
            debate_stop_after="round_1",
            stage_state_file=state_file,
        )

    resume_engine = _ResumeEngine()
    results = run_debate(
        dataset="aime25",
        items=[item],
        engine=resume_engine,
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=None,
        judge_engine=resume_engine,
        progress_file=StringIO(),
        stage_state_file=state_file,
    )

    entries = load_all_stage_entries(state_file)
    assert [entry.completed_stage for entry in entries] == ["round_0", "round_1", "round_1_judge"]
    assert [call["model_role"] for call in resume_engine.call_log] == ["judge"]
    assert len(results[1]) == 1


def test_debate_resume_rejects_mismatched_persona_settings(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _debate_append_state, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    _debate_append_state(
        stage_state_file=state_file,
        completed_stage="round_0",
        dataset="aime25",
        items=[item],
        all_contexts=[[[{"role": "user", "content": "question"}]]],
        agent_round_outputs_by_q=[[[{"final_answer": "4", "private_raw_response": "Initial solve. Final answer \\boxed{4}"}]]],
        results_by_round={},
        prev_judge_by_q=[None],
        round1_majority_by_q=[None],
        persona_artifacts=[None],
        persona_artifact_paths=[None],
        runtime_judge_cards=[None],
        runtime_judge_bank_meta=[None],
        persona_settings={
            "use_personas": False,
            "runtime_judge_persona_enabled": False,
            "persona_seed": 0,
            "persona_axis_mode": "fixed",
            "persona_fixed_axis_count": 2,
            "persona_task_axis_count": 0,
            "persona_sampling_method": "maximin",
            "persona_judge_mode": "neutral_baseline",
            "persona_backend": "heuristic",
            "generator_model": None,
            "judge_generator_model": None,
            "persona_axes_file": None,
            "judge_bank_dir": None,
            "judge_bank_refresh": False,
            "gpqa_family_cache_path": None,
        },
        runtime_settings={
            "debater_model": "fake-model",
            "debater_backend": "fake",
            "judge_model": "fake-model",
            "judge_backend": "fake",
            "judge_block_size": None,
            "judge_sampling_kwargs": {},
            "judge_strict_final_only": True,
            "judge_recovery_parse_enabled": True,
            "judge_trace_mode": "visible_plus_thought_summary",
            "public_rationale_max_tokens": 96,
        },
        n_agents=1,
        n_rounds=1,
        judge_rounds=[1],
    )

    with pytest.raises(ValueError, match="persona settings mismatch"):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=_ResumeEngine(),
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=_ResumeEngine(),
            use_personas=True,
            artifacts_dir=tmp_path / "artifacts",
            persona_seed=0,
            persona_axis_mode="fixed",
            persona_fixed_axis_count=2,
            persona_task_axis_count=0,
            persona_sampling_method="maximin",
            persona_judge_mode="neutral_baseline",
            persona_backend="heuristic",
            progress_file=StringIO(),
            stage_state_file=state_file,
        )


def test_debate_resume_rejects_mismatched_judge_rounds(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    first_engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=first_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=first_engine,
            progress_file=StringIO(),
            debate_stop_after="round_1",
            stage_state_file=state_file,
        )

    with pytest.raises(ValueError, match="judge_rounds mismatch"):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=_ResumeEngine(),
            n_agents=1,
            n_rounds=1,
            judge_rounds=[0, 1],
            batch_size=None,
            judge_engine=_ResumeEngine(),
            progress_file=StringIO(),
            stage_state_file=state_file,
        )


def test_staged_persona_resume_rejects_changed_item_payload(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    original_item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    changed_item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision="rev-2", item_display_id=0,
        raw_task={"question": "What is 2+3?", "answer": "5"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    run_persona_generation_staged(
        dataset="aime25", items=[original_item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="axes", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        summary_file=summary,
    )

    with pytest.raises(ValueError, match="items do not match"):
        run_persona_generation_staged(
            dataset="aime25", items=[changed_item], artifacts_dir=artifacts_dir,
            stage_state_file=state_file, persona_stage="descriptors", n_personas=2,
            persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
            task_axis_count=0, sampling_method="maximin",
            judge_persona_mode="neutral_baseline", backend="heuristic",
            generator_model=None, judge_generator_model=None,
            generator_engine=None, judge_engine=None, axes_file=None,
            summary_file=summary,
        )


def test_staged_persona_resume_allows_judge_bank_flag_changes(tmp_path: Path):
    from debate_v_majority.cli.persona_runtime import run_persona_generation_staged
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "state.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    summary = StringIO()

    run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="cards", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        judge_bank_dir=tmp_path / "judge_bank_a",
        gpqa_family_cache_path=tmp_path / "gpqa_a.json",
        summary_file=summary,
    )

    entry = run_persona_generation_staged(
        dataset="aime25", items=[item], artifacts_dir=artifacts_dir,
        stage_state_file=state_file, persona_stage="cards", n_personas=2,
        persona_seed=0, axis_mode="fixed", fixed_axis_count=2,
        task_axis_count=0, sampling_method="maximin",
        judge_persona_mode="neutral_baseline", backend="heuristic",
        generator_model=None, judge_generator_model=None,
        generator_engine=None, judge_engine=None, axes_file=None,
        judge_bank_dir=tmp_path / "judge_bank_b",
        judge_bank_refresh=True,
        gpqa_family_cache_path=tmp_path / "gpqa_b.json",
        summary_file=summary,
    )

    assert entry.completed_stage == "cards"


def test_debate_resume_rejects_changed_item_payload(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    original_item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    changed_item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision="rev-2", item_display_id=0,
        raw_task={"question": "What is 2+3?", "answer": "5"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    first_engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[original_item],
            engine=first_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=first_engine,
            progress_file=StringIO(),
            debate_stop_after="round_0",
            stage_state_file=state_file,
        )

    with pytest.raises(ValueError, match="items do not match"):
        run_debate(
            dataset="aime25",
            items=[changed_item],
            engine=_ResumeEngine(),
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=_ResumeEngine(),
            progress_file=StringIO(),
            stage_state_file=state_file,
        )


def test_debate_resume_rejects_runtime_model_mismatch(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    first_engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=first_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=first_engine,
            progress_file=StringIO(),
            debate_stop_after="round_0",
            stage_state_file=state_file,
        )

    resume_engine = _ResumeEngine()
    resume_engine.model_name = "other-model"
    with pytest.raises(ValueError, match="runtime settings mismatch"):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=resume_engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=resume_engine,
            progress_file=StringIO(),
            stage_state_file=state_file,
        )


def test_debate_resume_rejects_judge_only_runtime_flag_mismatch(tmp_path: Path):
    from debate_v_majority.cli.debate_runner import _DebateStopped, run_debate
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    state_file = tmp_path / "debate_state.jsonl"
    engine = _ResumeEngine()

    with pytest.raises(_DebateStopped):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=engine,
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=engine,
            progress_file=StringIO(),
            debate_stop_after="round_0",
            stage_state_file=state_file,
        )

    with pytest.raises(ValueError, match="runtime settings mismatch"):
        run_debate(
            dataset="aime25",
            items=[item],
            engine=_ResumeEngine(),
            n_agents=1,
            n_rounds=1,
            judge_rounds=[1],
            batch_size=None,
            judge_engine=_ResumeEngine(),
            judge_block_size=1,
            progress_file=StringIO(),
            stage_state_file=state_file,
        )


def test_validate_debate_stop_after_rejects_malformed_stage():
    from debate_v_majority.cli.main_impl import _validate_debate_stop_after

    with pytest.raises(ValueError, match="Invalid --debate_stop_after stage"):
        _validate_debate_stop_after(
            debate_stop_after="round1",
            n_rounds=3,
            judge_rounds=[3],
        )


def test_validate_debate_stop_after_rejects_unjudged_stage():
    from debate_v_majority.cli.main_impl import _validate_debate_stop_after

    with pytest.raises(ValueError, match="not a judged step"):
        _validate_debate_stop_after(
            debate_stop_after="round_1_judge",
            n_rounds=3,
            judge_rounds=[3],
        )


def test_prompt_continue_persona_stage_accepts_enter(monkeypatch):
    from debate_v_majority.cli.main_impl import _prompt_continue_persona_stage

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert _prompt_continue_persona_stage(next_stage="descriptors", status_file=StringIO()) is True


def test_prompt_continue_persona_stage_stops_on_q(monkeypatch):
    from debate_v_majority.cli.main_impl import _prompt_continue_persona_stage

    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    assert _prompt_continue_persona_stage(next_stage="descriptors", status_file=StringIO()) is False


def test_prompt_continue_persona_stage_stops_on_ctrl_c(monkeypatch):
    from debate_v_majority.cli.main_impl import _prompt_continue_persona_stage

    status = StringIO()

    def _raise_interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)
    assert _prompt_continue_persona_stage(next_stage="descriptors", status_file=status) is False
    assert "Progress has been saved" in status.getvalue()


def test_format_stage_cost_summary_includes_remaining_budget(tmp_path: Path):
    from debate_v_majority.accounting import CostTracker
    from debate_v_majority.cli.main_impl import _format_stage_cost_summary

    tracker = CostTracker(
        ledger_path=tmp_path / "ledger.jsonl",
        session_name="test",
        max_run_cost_usd=1.0,
        max_total_cost_usd=10.0,
    )
    before = {
        "session": {
            "n_calls": 1,
            "estimated_cost_usd": 0.25,
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "billable_output_tokens": 30,
        },
        "cumulative": {"estimated_cost_usd": 4.25},
    }
    after = {
        "session": {
            "n_calls": 3,
            "estimated_cost_usd": 0.40,
            "input_tokens": 150,
            "cached_input_tokens": 50,
            "billable_output_tokens": 70,
        },
        "cumulative": {"estimated_cost_usd": 4.40},
    }

    line = _format_stage_cost_summary(
        label="debate:round_0",
        before=before,
        after=after,
        cost_tracker=tracker,
    )

    assert line is not None
    assert "debate:round_0=$0.150000" in line
    assert "run_remaining=$0.600000" in line
    assert "cumulative_remaining=$5.600000" in line


def test_main_skips_persona_output_for_staged_runs(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    stage_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_persona_generation_staged",
        lambda **kwargs: (
            stage_calls.append(kwargs)
            or make_stage_entry(
                stage_type="persona",
                completed_stage=kwargs["persona_stage"],
                dataset="aime25",
                items=[asdict(item)],
            )
        ),
    )
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "personas",
            "--persona_backend",
            "heuristic",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--persona_stage",
            "axes",
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert len(stage_calls) == 1
    assert not list(tmp_path.glob("personas_*.jsonl"))


def test_main_writes_persona_output_after_final_staged_run(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_persona_generation_staged",
        lambda **kwargs: make_stage_entry(
            stage_type="persona",
            completed_stage="judge_card",
            dataset="aime25",
            items=[asdict(item)],
        ),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "persona_rows_from_stage_entry",
        lambda entry: [{"item_uid": "aime25:test_0", "cards": [], "judge_card": None}],
    )
    monkeypatch.setattr(cli_main_impl, "persona_artifacts_from_rows", lambda rows: {})
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "personas",
            "--persona_backend",
            "heuristic",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--persona_stage",
            "judge_card",
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert list(tmp_path.glob("personas_*.jsonl"))


def test_main_interactive_persona_staging_advances_without_rerun(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    stage_calls: list[str] = []
    prompt_results = iter([True, False])

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )

    def _fake_stage_runner(**kwargs):
        stage_calls.append(kwargs["persona_stage"])
        return make_stage_entry(
            stage_type="persona",
            completed_stage=kwargs["persona_stage"],
            dataset="aime25",
            items=[asdict(item)],
        )

    monkeypatch.setattr(cli_main_impl, "run_persona_generation_staged", _fake_stage_runner)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120002")
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_persona_stage", lambda **kwargs: next(prompt_results))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "personas",
            "--persona_backend",
            "heuristic",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--persona_stage",
            "axes",
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert stage_calls == ["axes", "descriptors"]


def test_main_runs_personas_then_debate_in_one_flow(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    call_order: list[str] = []
    debate_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_persona_generation",
        lambda **kwargs: (
            call_order.append("personas")
            or [{"item_uid": "aime25:test_0", "cards": [], "judge_card": None}]
        ),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "persona_artifacts_from_rows",
        lambda rows: {"aime25:test_0": "persona-sentinel"},
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_debate",
        lambda **kwargs: (
            call_order.append("debate")
            or debate_calls.append(kwargs)
            or {3: [{"final_correct": 1}]}
        ),
    )
    monkeypatch.setattr(cli_main_impl, "_augment_output_rows", lambda records, **kwargs: records)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120100")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "personas,debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert call_order == ["personas", "debate"]
    assert len(debate_calls) == 1
    assert debate_calls[0]["use_personas"] is True
    assert debate_calls[0]["persona_artifacts_by_item"] == {"aime25:test_0": "persona-sentinel"}
    assert list(tmp_path.glob("personas_*.jsonl"))
    assert list(tmp_path.glob("debate_*.jsonl"))


def test_main_interactive_persona_then_debate_staging_in_one_flow(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.debate_runner import _DebateStopped
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    persona_stage_calls: list[str] = []
    debate_stage_calls: list[str | None] = []
    debate_calls: list[dict[str, Any]] = []
    persona_prompts = iter([True, True, True])
    handoff_prompts = iter([True])
    debate_prompts = iter([True, True])

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )

    def _fake_stage_runner(**kwargs):
        persona_stage_calls.append(kwargs["persona_stage"])
        return make_stage_entry(
            stage_type="persona",
            completed_stage=kwargs["persona_stage"],
            dataset="aime25",
            items=[asdict(item)],
        )

    def _fake_run_debate(**kwargs):
        stop_after = kwargs.get("debate_stop_after")
        debate_calls.append(kwargs)
        debate_stage_calls.append(stop_after)
        if stop_after in {"round_0", "round_1"}:
            raise _DebateStopped({1: []})
        return {1: [{"final_correct": 1}]}

    monkeypatch.setattr(cli_main_impl, "run_persona_generation_staged", _fake_stage_runner)
    monkeypatch.setattr(
        cli_main_impl,
        "persona_rows_from_stage_entry",
        lambda entry: [{"item_uid": "aime25:test_0", "cards": [], "judge_card": None}],
    )
    monkeypatch.setattr(
        cli_main_impl,
        "persona_artifacts_from_rows",
        lambda rows: {"aime25:test_0": "staged-persona-sentinel"},
    )
    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(cli_main_impl, "_augment_output_rows", lambda records, **kwargs: records)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_persona_stage", lambda **kwargs: next(persona_prompts))
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_to_debate", lambda **kwargs: next(handoff_prompts))
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_debate_stage", lambda **kwargs: next(debate_prompts))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120101")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "personas,debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--persona_stage",
            "axes",
            "--n_rounds",
            "1",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert persona_stage_calls == ["axes", "descriptors", "cards", "judge_card"]
    assert debate_stage_calls == ["round_0", "round_1", "round_1_judge"]
    assert all(call["persona_artifacts_by_item"] == {"aime25:test_0": "staged-persona-sentinel"} for call in debate_calls)
    assert list(tmp_path.glob("personas_*.jsonl"))
    assert list(tmp_path.glob("debate_*.jsonl"))


def test_main_skips_debate_output_when_stopped_early(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.debate_runner import _DebateStopped
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    majority_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_sampled",
        lambda **kwargs: (majority_calls.append(kwargs) or [{"final_correct": 1}]),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "run_debate",
        lambda **kwargs: (_ for _ in ()).throw(_DebateStopped({1: []})),
    )
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120001")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate,majority",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--debate_stop_after",
            "round_0",
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert not majority_calls
    assert not list(tmp_path.glob("debate_*.jsonl"))
    assert not list(tmp_path.glob("majority_*.jsonl"))


def test_main_reuses_auto_debate_stage_file_on_resume(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.debate_runner import _DebateStopped
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    stage_state_calls: list[Path | None] = []
    timestamp_values = iter(["20260311_120010", "20260311_120011"])

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: next(timestamp_values))

    def _fake_run_debate(**kwargs):
        state_file = kwargs.get("stage_state_file")
        stage_state_calls.append(state_file)
        if len(stage_state_calls) == 1:
            assert state_file is not None
            append_stage_entry(
                state_file,
                make_stage_entry(
                    stage_type="debate",
                    completed_stage="round_0",
                    dataset="aime25",
                    items=[asdict(item)],
                    debate_data={
                        "n_agents": kwargs["n_agents"],
                        "n_rounds": kwargs["n_rounds"],
                        "judge_rounds": list(kwargs["judge_rounds"]),
                    },
                ),
            )
            raise _DebateStopped({1: []})
        return {1: [{"final_correct": 1}]}

    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(cli_main_impl, "_augment_output_rows", lambda records, **kwargs: records)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--debate_stop_after",
            "round_0",
            "--quiet",
        ],
    )
    cli_main_impl.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )
    cli_main_impl.main()

    assert len(stage_state_calls) == 2
    assert stage_state_calls[0] is not None
    assert stage_state_calls[1] == stage_state_calls[0]
    assert not (tmp_path / "stage_state_aime_debate.latest").exists()


def test_main_reuses_completed_auto_debate_stage_file(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    pointer_path = tmp_path / "stage_state_aime_debate.latest"
    stage_state_file = tmp_path / "debate_state_complete.jsonl"
    append_stage_entry(
        stage_state_file,
        make_stage_entry(
            stage_type="debate",
            completed_stage="round_1_judge",
            dataset="aime25",
            items=[asdict(item)],
            debate_data={
                "n_agents": 5,
                "n_rounds": 1,
                "judge_rounds": [1],
                "results_by_round": {"1": [{"final_correct": 1}]},
                "runtime_settings": {"debater_model": "fake-model", "debater_backend": "fake"},
                "persona_settings": {
                    "use_personas": False,
                    "runtime_judge_persona_enabled": False,
                    "persona_seed": 0,
                    "persona_axis_mode": "hybrid",
                    "persona_fixed_axis_count": 3,
                    "persona_task_axis_count": 3,
                    "persona_sampling_method": "maximin",
                    "persona_judge_mode": "benchmark_family_bank",
                    "persona_backend": "llm",
                    "generator_model": None,
                    "judge_generator_model": None,
                    "persona_axes_file": None,
                    "judge_bank_dir": None,
                    "judge_bank_refresh": False,
                    "gpqa_family_cache_path": None,
                },
                "all_contexts": [[[] for _ in range(5)]],
                "agent_round_outputs_by_q": [[[] for _ in range(5)]],
                "prev_judge_by_q": [None],
                "round1_majority_by_q": [None],
                "persona_artifacts": [None],
                "persona_artifact_paths": [None],
                "runtime_judge_cards": [None],
                "runtime_judge_bank_meta": [None],
            },
        ),
    )
    pointer_path.write_text(str(stage_state_file), encoding="utf-8")
    stage_state_calls: list[Path | None] = []

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260311_120030")
    monkeypatch.setattr(cli_main_impl, "_augment_output_rows", lambda records, **kwargs: records)

    def _fake_run_debate(**kwargs):
        stage_state_calls.append(kwargs.get("stage_state_file"))
        return {1: [{"final_correct": 1}]}

    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--n_rounds",
            "1",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert stage_state_calls == [stage_state_file]
    assert not pointer_path.exists()
    assert list(tmp_path.glob("debate_*.jsonl"))


def test_main_discards_auto_debate_stage_file_on_settings_mismatch(tmp_path: Path, monkeypatch):
    from debate_v_majority.cli import main_impl as cli_main_impl
    from debate_v_majority.cli.debate_runner import _DebateStopped
    from debate_v_majority.cli.subset import SubsetItem

    item = SubsetItem(
        subset_id=0, orig_id=0, item_uid="aime25:test_0",
        dataset_revision=None, item_display_id=0,
        raw_task={"question": "What is 2+2?", "answer": "4"},
        dataset_meta={},
    )
    stage_state_calls: list[Path | None] = []
    timestamp_values = iter(["20260311_120020", "20260311_120021"])
    first_stage_state_file: Path | None = None

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: tmp_path / "dataset.jsonl")
    monkeypatch.setattr(
        cli_main_impl,
        "_make_dataset_subset",
        lambda **kwargs: ([item], {"subset_size": 1, "seed": 0}),
    )
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", lambda **kwargs: _ResumeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "_reuse_or_create_role_engine",
        lambda **kwargs: kwargs["existing_engine"],
    )
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: next(timestamp_values))
    monkeypatch.setattr(cli_main_impl, "_augment_output_rows", lambda records, **kwargs: records)

    def _fake_run_debate(**kwargs):
        nonlocal first_stage_state_file
        state_file = kwargs.get("stage_state_file")
        stage_state_calls.append(state_file)
        if len(stage_state_calls) == 1:
            assert state_file is not None
            first_stage_state_file = state_file
            append_stage_entry(
                state_file,
                make_stage_entry(
                    stage_type="debate",
                    completed_stage="round_0",
                    dataset="aime25",
                    items=[asdict(item)],
                    debate_data={
                        "n_agents": kwargs["n_agents"],
                        "n_rounds": kwargs["n_rounds"],
                        "judge_rounds": list(kwargs["judge_rounds"]),
                    },
                ),
            )
            raise _DebateStopped({1: []})
        if len(stage_state_calls) == 2:
            assert state_file == first_stage_state_file
            raise ValueError("Debate state runtime settings mismatch")
        assert state_file is None
        return {1: [{"final_correct": 1}]}

    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--debate_stop_after",
            "round_0",
            "--quiet",
        ],
    )
    cli_main_impl.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_ids",
            "0",
            "--out_dir",
            str(tmp_path),
            "--judge_trace_mode",
            "assistant_transcript",
            "--quiet",
        ],
    )
    cli_main_impl.main()

    assert len(stage_state_calls) == 3
    assert stage_state_calls[0] == first_stage_state_file
    assert stage_state_calls[1] == first_stage_state_file
    assert stage_state_calls[2] is None
    assert not (tmp_path / "stage_state_aime_debate.latest").exists()
