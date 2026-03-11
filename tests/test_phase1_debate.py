from __future__ import annotations

import json
from pathlib import Path

import pytest

from debate_v_majority.cli.dataset_eval import _parse_question_answer
from debate_v_majority.cli.debate_runner import run_debate
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.engines import InferenceResult


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class _FakeDebateEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "fake-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name
        self.calls: list[tuple[list[list[dict[str, str]]], int | None, dict[str, object] | None]] = []

    def count_prompt_tokens(self, messages):
        del messages
        return 11

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        call_idx = len(self.calls)
        assert call_idx < len(self.outputs_by_call), f"Unexpected extra engine call {call_idx}"
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts), (call_idx, len(outputs), len(contexts))
        self.calls.append(([[dict(msg) for msg in ctx] for ctx in contexts], batch_size, sampling_kwargs))
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


class _TypedFakeDebateEngine:
    def __init__(self, outputs_by_call: list[list[InferenceResult]], *, model_name: str = "fake-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name
        self.provider_name = "fake-provider"
        self.calls: list[tuple[list[list[dict[str, str]]], int | None, dict[str, object] | None, str | None]] = []
        self._context_len_tokens = 32768

    @property
    def context_len_tokens(self) -> int:
        return self._context_len_tokens

    def count_prompt_tokens(self, messages):
        del messages
        return 11

    def generate_batch_results(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None, model_role=None):
        call_idx = len(self.calls)
        assert call_idx < len(self.outputs_by_call), f"Unexpected extra engine call {call_idx}"
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts), (call_idx, len(outputs), len(contexts))
        self.calls.append(([[dict(msg) for msg in ctx] for ctx in contexts], batch_size, sampling_kwargs, model_role))
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def test_run_debate_with_personas_shares_visible_full_peer_outputs(tmp_path: Path):
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
                "id": "gpqa-item-1",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    wrong_letter = next(letter for letter in ("A", "B", "C", "D") if letter != gt_letter)
    engine = _FakeDebateEngine(
        outputs_by_call=[
            ['{"judge_family":"physics"}'],
            [
                f"Reason: the stem best matches option {gt_letter}.\nConfidence: 0.70\n\\boxed{{{gt_letter}}}",
                f"Reason: I suspect option {wrong_letter}.\nConfidence: 0.40\n\\boxed{{{wrong_letter}}}",
                f"Reason: option {gt_letter} is most consistent.\nConfidence: 0.80\n\\boxed{{{gt_letter}}}",
            ],
            [
                f"After comparing the public critiques, option {gt_letter} still has the strongest support.\nConfidence: 0.88\n\\boxed{{{gt_letter}}}",
                f"The other agents exposed the flaw in {wrong_letter}, so I switch to {gt_letter}.\nConfidence: 0.72\n\\boxed{{{gt_letter}}}",
                f"I still prefer {gt_letter} after checking the objections.\nConfidence: 0.91\n\\boxed{{{gt_letter}}}",
            ],
            [f"\\boxed{{{gt_letter}}}"],
        ]
    )
    artifact_dir = tmp_path / "artifacts"

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=3,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=artifact_dir,
        persona_seed=7,
        persona_axis_mode="hybrid",
        persona_fixed_axis_count=2,
        persona_task_axis_count=1,
        persona_sampling_method="maximin",
        persona_judge_mode="task_family_generated",
        persona_backend="heuristic",
        persona_save_artifacts=True,
        public_rationale_max_tokens=12,
    )

    row = results[1][0]
    assert row["schema_version"] == "phase2.debate.v1"
    assert row["dataset"] == "gpqa"
    assert row["debater_model"] == "fake-model"
    assert row["item_uid"] == "gpqa:gpqa-item-1"
    assert row["round1_majority_answer"] == gt_letter
    assert row["round1_majority_correct"] == 1
    assert row["round1_majority_origin"] == "shared_debate_round1"
    assert row["round1_vote_counts"][gt_letter] == 2
    assert row["round1_vote_counts"][wrong_letter] == 1
    assert row["round1_strict_majority_answer"] == gt_letter
    assert row["round1_plurality_answer"] == gt_letter
    assert row["round1_majority_result"]["majority_answer"] == gt_letter
    assert row["final_round_majority_answer"] == gt_letter
    assert row["final_round_majority_result"]["majority_answer"] == gt_letter
    assert row["judge_final_answer"] == gt_letter
    assert row["judge_result"]["answer"] == gt_letter
    assert row["final_correct"] == 1
    assert Path(row["persona_meta"]["artifact_path"]).exists()
    assert row["judge_meta"]["judge_persona_mode"] == "task_family_generated"
    assert row["judge_trace"]["judge_backend"] == "_FakeDebateEngine"
    assert row["judge_trace"]["judge_card"] is not None
    assert len(row["persona_ids"]) == 3
    assert len(row["persona_summaries"]) == 3
    assert len(row["agent_round_outputs"]) == 3
    assert all(len(agent_outputs) == 2 for agent_outputs in row["agent_round_outputs"])
    assert len(row["convergence_per_round"]) == 2
    assert len(row["answer_changes_per_agent"]) == 3
    assert row["agent_responses"][0][0]["role"] == "system"

    second_round_contexts = engine.calls[2][0]
    debate_prompt = second_round_contexts[0][-1]["content"]
    assert f"\\boxed{{{wrong_letter}}}" in debate_prompt
    assert "Public rationale:" not in debate_prompt
    assert "Final answer:" not in debate_prompt
    assert "private scratch agent2" not in debate_prompt


def test_peer_shared_output_contains_full_visible_response(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_full.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-2",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    engine = _FakeDebateEngine(
        outputs_by_call=[
            ['{"judge_family":"physics"}'],
            [
                f"Reason one.\n\\boxed{{{gt_letter}}}",
                f"Reason two.\n\\boxed{{{gt_letter}}}",
            ],
            [
                f"Updated response.\n\\boxed{{{gt_letter}}}",
                f"Updated response.\n\\boxed{{{gt_letter}}}",
            ],
            [f"\\boxed{{{gt_letter}}}"],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = results[1][0]
    debate_prompt = engine.calls[2][0][0][-1]["content"]
    assert "Reason two." in debate_prompt


def test_judge_context_includes_all_round_outputs(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_judge_ctx.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-judge-ctx",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    engine = _FakeDebateEngine(
        outputs_by_call=[
            ['{"judge_family":"physics"}'],
            [
                f"Visible explanation one.\n\\boxed{{{gt_letter}}}",
                f"Visible explanation two.\n\\boxed{{{gt_letter}}}",
            ],
            [
                f"Visible explanation one, round two.\n\\boxed{{{gt_letter}}}",
                f"Visible explanation two, round two.\n\\boxed{{{gt_letter}}}",
            ],
            [f"\\boxed{{{gt_letter}}}"],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        judge_block_size=1,
        batch_size=4,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = results[1][0]
    judge_context = row["judge_trace"]["judge_context"][-1]["content"]
    assert "Visible explanation one." in judge_context
    assert "Visible explanation two." in judge_context
    assert "Visible explanation one, round two." in judge_context
    assert "Visible explanation two, round two." in judge_context
    assert "ROUND 1" in judge_context
    assert "ROUND 2" in judge_context


def test_benchmark_family_bank_shares_visible_outputs_and_gives_judge_thought_summaries(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_bank.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option best describes DNA replication?",
                "Correct Answer": "semi-conservative",
                "Incorrect Answer 1": "fully conservative",
                "Incorrect Answer 2": "dispersive only",
                "Incorrect Answer 3": "random copying",
                "id": "gpqa-item-bank-1",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    wrong_letter = next(letter for letter in ("A", "B", "C", "D") if letter != gt_letter)
    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text='{"judge_family":"biology"}'),
            ],
            [
                InferenceResult(text=f"Initial answer {gt_letter}.", thought_summary="I favored the biology mechanism.", thought_summary_available=True),
                InferenceResult(text=f"Initial answer {wrong_letter}.", thought_summary="I over-weighted an incorrect distractor.", thought_summary_available=True),
                InferenceResult(text=f"Initial answer {gt_letter}.", thought_summary="I checked the replication description.", thought_summary_available=True),
            ],
            [
                InferenceResult(text=f"Updated answer {gt_letter}.", thought_summary="The peer outputs reinforced the biology answer.", thought_summary_available=True),
                InferenceResult(text=f"Updated answer {gt_letter}.", thought_summary="I corrected my distractor mistake after comparing outputs.", thought_summary_available=True),
                InferenceResult(text=f"Updated answer {gt_letter}.", thought_summary="The visible outputs stayed consistent with the right choice.", thought_summary_available=True),
            ],
            [
                InferenceResult(text=f"\\boxed{{{gt_letter}}}"),
            ],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=3,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
        persona_judge_mode="benchmark_family_bank",
        judge_trace_mode="visible_plus_thought_summary",
    )

    row = results[1][0]
    debate_prompt = engine.calls[2][0][0][-1]["content"]
    assert "Initial answer" in debate_prompt
    assert "Thought summary" not in debate_prompt
    assert row["judge_meta"]["judge_persona_mode"] == "benchmark_family_bank"
    assert row["judge_meta"]["judge_trace_mode"] == "visible_plus_thought_summary"
    assert row["judge_meta"]["judge_bank"]["judge_family_assignment"]["judge_family"] == "biology"
    assert Path(row["judge_meta"]["judge_bank"]["judge_bank_path"]).exists()
    assert row["agent_round_outputs"][0][0]["thought_summary"] == "I favored the biology mechanism."
    judge_context = row["judge_trace"]["judge_context"][-1]["content"]
    assert "Thought summary:" in judge_context
    assert "I corrected my distractor mistake after comparing outputs." in judge_context


def test_visible_plus_thought_summary_mode_reaches_non_benchmark_judges(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_visible_plus_thought_summary.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option best describes DNA replication?",
                "Correct Answer": "semi-conservative",
                "Incorrect Answer 1": "fully conservative",
                "Incorrect Answer 2": "dispersive only",
                "Incorrect Answer 3": "random copying",
                "id": "gpqa-item-visible-thought-summary",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text='{"judge_family":"biology"}'),
            ],
            [
                InferenceResult(text=f"Round one visible {gt_letter}.", thought_summary="Initial mechanism check.", thought_summary_available=True),
                InferenceResult(text=f"Round one visible {gt_letter}.", thought_summary="Cross-checking distractors.", thought_summary_available=True),
            ],
            [
                InferenceResult(text=f"Round two visible {gt_letter}.", thought_summary="Peers reinforced the same answer.", thought_summary_available=True),
                InferenceResult(text=f"Round two visible {gt_letter}.", thought_summary="No contradiction survived round two.", thought_summary_available=True),
            ],
            [
                InferenceResult(text=f"\\boxed{{{gt_letter}}}"),
            ],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
        judge_trace_mode="visible_plus_thought_summary",
    )

    row = results[1][0]
    judge_context = row["judge_trace"]["judge_context"][-1]["content"]
    assert row["judge_meta"]["judge_trace_mode"] == "visible_plus_thought_summary"
    assert "Thought summary:" in judge_context
    assert "Initial mechanism check." in judge_context
    assert "No contradiction survived round two." in judge_context
    assert "ROUND 1" in judge_context
    assert "ROUND 2" in judge_context


def test_benchmark_family_bank_assistant_transcript_keeps_full_round_history(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_bank_full_history.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option best describes DNA replication?",
                "Correct Answer": "semi-conservative",
                "Incorrect Answer 1": "fully conservative",
                "Incorrect Answer 2": "dispersive only",
                "Incorrect Answer 3": "random copying",
                "id": "gpqa-item-bank-full-history",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text='{"judge_family":"biology"}'),
            ],
            [
                InferenceResult(text=f"Round one visible {gt_letter}.", thought_summary="Private first-round thinking.", thought_summary_available=True),
                InferenceResult(text=f"Round one visible {gt_letter}."),
            ],
            [
                InferenceResult(text=f"Round two visible {gt_letter}."),
                InferenceResult(text=f"Round two visible {gt_letter}."),
            ],
            [
                InferenceResult(text=f"\\boxed{{{gt_letter}}}"),
            ],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
        judge_trace_mode="assistant_transcript",
        persona_judge_mode="benchmark_family_bank",
    )

    row = results[1][0]
    judge_context = row["judge_trace"]["judge_context"][-1]["content"]
    assert row["judge_meta"]["judge_trace_mode"] == "assistant_transcript"
    assert "ROUND 1" in judge_context
    assert "ROUND 2" in judge_context
    assert "Round one visible" in judge_context
    assert "Round two visible" in judge_context
    assert "Private first-round thinking" not in judge_context


def test_run_debate_with_personas_works_on_aime(tmp_path: Path):
    dataset_path = tmp_path / "aime.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "problem": "What is 1+1?",
                "answer": "2",
                "id": "aime-item-1",
            }
        ],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    engine = _FakeDebateEngine(
        outputs_by_call=[
            [
                "Compute directly. \\boxed{2}",
                "The sum is two. \\boxed{2}",
                "1 plus 1 equals 3. \\boxed{3}",
            ],
            ["\\boxed{2}"],
        ]
    )

    results = run_debate(
        dataset="aime25",
        items=items,
        engine=engine,
        n_agents=3,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = results[0][0]
    assert row["round1_majority_answer"] == "2"
    assert row["final_round_majority_answer"] == "2"
    assert row["judge_final_answer"] == "2"
    assert row["round1_majority_correct"] == 1
    assert row["judge_final_correct"] == 1
    assert row["persona_meta"]["artifact_path"] is not None


def test_run_debate_propagates_judge_retry_generation_failures(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_retry_failure.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-retry-failure",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)

    class _RetryFailureEngine(_TypedFakeDebateEngine):
        def generate_batch_results(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None, model_role=None):
            if len(self.calls) == 2:
                raise RuntimeError("judge retry failed")
            return super().generate_batch_results(
                contexts,
                batch_size=batch_size,
                sampling_kwargs=sampling_kwargs,
                progress_callback=progress_callback,
                model_role=model_role,
            )

    engine = _RetryFailureEngine(
        outputs_by_call=[
            [
                InferenceResult(text=f"Round one {gt_letter}."),
                InferenceResult(text=f"Round one {gt_letter}."),
            ],
            [
                InferenceResult(text="Judge forgot to box the answer"),
            ],
        ]
    )

    try:
            run_debate(
                dataset="gpqa",
                items=items,
                engine=engine,
                n_agents=2,
                n_rounds=0,
                judge_rounds=[0],
                batch_size=4,
                judge_block_size=1,
            )
    except RuntimeError as exc:
        assert "judge retry failed" in str(exc)
    else:
        raise AssertionError("Expected judge retry generation failure to propagate")


def test_run_debate_raises_when_judge_output_remains_unparsable(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_retry_unparsed.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-retry-unparsed",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)

    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text=f"Round one {gt_letter}."),
                InferenceResult(text=f"Round one {gt_letter}."),
            ],
            [
                InferenceResult(text="Judge forgot to box the answer"),
            ],
            [
                InferenceResult(text="Still not boxed after retry"),
            ],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
    )
    row = results[0][0]
    assert row["judge_final_answer"] is None
    assert row["judge_trace"]["judge_parse_failed"] is True


def test_run_debate_accepts_recovered_initial_judge_answer_without_retry(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_recover_initial_judge.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-recover-initial-judge",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)

    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text=f"Round one {gt_letter}."),
                InferenceResult(text=f"Round one {gt_letter}."),
            ],
            [
                InferenceResult(text=f"Final answer: {gt_letter}"),
            ],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
        judge_recovery_parse_enabled=True,
    )
    row = results[0][0]
    assert len(engine.calls) == 2
    assert row["judge_final_answer"] == gt_letter
    assert row["judge_trace"]["judge_parse_mode"] == "recover"
    assert row["judge_trace"]["judge_retry_raw_response"] is None
    assert row["judge_trace"]["judge_parse_failed"] is False


def test_run_debate_without_native_prompt_counter_does_not_crash_prejudge(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_no_prompt_counter.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-item-no-prompt-counter",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)

    class _NoPromptCounterEngine:
        def __init__(self) -> None:
            self.model_name = "fake-model"
            self.provider_name = "legacy"
            self.calls = 0

        def generate_batch_results(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None, model_role=None):
            del batch_size, sampling_kwargs, model_role
            self.calls += 1
            if progress_callback is not None:
                progress_callback(len(contexts))
            if self.calls == 1:
                return [
                    InferenceResult(text=f"Round one {gt_letter}."),
                    InferenceResult(text=f"Round one {gt_letter}."),
                ]
            return [InferenceResult(text=f"\\boxed{{{gt_letter}}}")]

    engine = _NoPromptCounterEngine()
    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
    )

    row = results[0][0]
    assert row["judge_final_answer"] == gt_letter


def test_run_debate_generated_judge_card_without_personas(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_generated_judge.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option best describes DNA replication?",
                "Correct Answer": "semi-conservative",
                "Incorrect Answer 1": "fully conservative",
                "Incorrect Answer 2": "dispersive only",
                "Incorrect Answer 3": "random copying",
                "id": "gpqa-generated-judge",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    engine = _TypedFakeDebateEngine(
        outputs_by_call=[
            [
                InferenceResult(text='{"judge_family":"biology"}'),
            ],
            [
                InferenceResult(text=f"Agent one picks {gt_letter}. \\boxed{{{gt_letter}}}"),
                InferenceResult(text=f"Agent two picks {gt_letter}. \\boxed{{{gt_letter}}}"),
            ],
            [InferenceResult(text=f"\\boxed{{{gt_letter}}}")],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        persona_judge_mode="task_family_generated",
        persona_backend="heuristic",
        enable_runtime_judge_persona=True,
        gpqa_family_cache_path=tmp_path / "gpqa_family_cache.json",
    )

    row = results[0][0]
    assert row["judge_meta"]["judge_persona_mode"] == "task_family_generated"
    assert row["judge_meta"]["judge_family_assignment"]["judge_family"] == "biology"
    assert row["judge_trace"]["judge_card"] is not None
    assert row["judge_trace"]["judge_card"]["judge_family"] == "biology"


def test_run_debate_requires_full_judge_transcript_when_context_is_too_long(tmp_path: Path):
    dataset_path = tmp_path / "gpqa_long_judge_window.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-long-window",
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
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)

    class _TightContextJudgeEngine(_TypedFakeDebateEngine):
        def __init__(self, outputs_by_call):
            super().__init__(outputs_by_call)
            self._context_len_tokens = 80

        def count_prompt_tokens(self, messages):
            text = "\n".join(str(msg.get("content", "")) for msg in messages)
            if "ROUND 1" in text and "ROUND 2" in text:
                return 500
            return 10

    engine = _TightContextJudgeEngine(
        outputs_by_call=[
            [
                InferenceResult(text=f"Very long round one analysis for {gt_letter}. \\boxed{{{gt_letter}}}"),
                InferenceResult(text=f"Different long round one analysis for {gt_letter}. \\boxed{{{gt_letter}}}"),
            ],
            [
                InferenceResult(text=f"Round two converges on {gt_letter}. \\boxed{{{gt_letter}}}"),
                InferenceResult(text=f"Round two also picks {gt_letter}. \\boxed{{{gt_letter}}}"),
            ],
            [InferenceResult(text=f"\\boxed{{{gt_letter}}}")],
        ]
    )

    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        judge_trace_mode="assistant_transcript",
    )
    row = results[1][0]
    assert row["final_judge_answer"] == gt_letter
    assert row["judge_trace"]["judge_context_is_full_transcript"] is False
    assert row["judge_trace"]["judge_context_start_round"] == 2
