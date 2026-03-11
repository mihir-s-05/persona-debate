from __future__ import annotations

import json
from pathlib import Path

from debate_v_majority.cli.debate_runner import run_debate
from debate_v_majority.cli.subset import _make_dataset_subset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _FakeDebateEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "fake-hle-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name
        self.calls: list[list[list[dict[str, str]]]] = []

    def count_prompt_tokens(self, messages):
        return sum(len(str(message.get("content") or "")) for message in messages) + len(messages)

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        call_idx = getattr(self, "_call_idx", 0)
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts)
        setattr(self, "_call_idx", call_idx + 1)
        self.calls.append([[dict(message) for message in context] for context in contexts])
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def test_run_debate_hle_emits_phase4_metadata_and_audit_traces(tmp_path: Path):
    dataset_path = tmp_path / "verified.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "hle-item-1",
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "blue",
                "answer_type": "multipleChoice",
                "category": "physics",
                "Verified_Classes": "Gold subset",
                "source_variant": "verified",
                "source_subset_label": "Gold subset",
                "source_dataset_id": "skylenage/HLE-Verified",
                "source_dataset_revision": "0bc83643672d4f68a5f89998617a639d85e7318b",
                "source_paper_version": "arXiv:2602.13964v3",
            }
        ],
    )

    items, meta = _make_dataset_subset(
        dataset="hle",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
        hle_variant="verified",
    )

    engine = _FakeDebateEngine(
        outputs_by_call=[
            [
                "Initial analysis.\nConfidence: 0.80\n\\boxed{B}",
                "I suspect blue is right.\nConfidence: 67%\nFinal answer: blue",
            ],
            [
                "Transcript support favors B.\nConfidence: 0.88\n\\boxed{B}",
            ],
        ]
    )

    results = run_debate(
        dataset="hle",
        items=items,
        engine=engine,
        n_agents=2,
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
    assert meta["dataset_variant"] == "verified"
    assert row["dataset"] == "hle"
    assert row["source_variant"] == "verified"
    assert row["source_subset_label"] == "Gold subset"
    assert row["canonical_item_id"] == "hle-item-1"
    assert row["answer_format_type"] == "multiple_choice"
    assert row["domain_family"] == "physical_sciences"
    assert row["round1_majority_answer"] == "B"
    assert row["round1_majority_correct"] == 1
    assert row["final_judge_answer"] == "B"
    assert row["final_judge_correct"] == 1
    assert row["judge_trace"]["judge_extractor_trace"]["normalized_answer"] == "B"
    assert row["judge_trace"]["judge_scoring_result"]["correct"] == 1
    assert row["judge_trace"]["judge_extractor_trace"]["extractor_provenance"] == "hle_verified.extractor.v1"
    agent_output = row["agent_round_outputs"][1][0]
    assert agent_output["final_answer"] == "B"
    assert agent_output["confidence"] == 0.67
    assert agent_output["confidence_raw_text"] == "Confidence: 67%"
    assert agent_output["parse_success"] is True
    assert agent_output["extractor_trace"]["answer_format_type"] == "multiple_choice"
    assert agent_output["scoring_result"]["correct"] == 1


def test_run_debate_hle_retry_keeps_primary_judge_trace_aligned_with_final_answer(tmp_path: Path):
    dataset_path = tmp_path / "verified_retry.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "hle-item-retry",
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "blue",
                "answer_type": "multipleChoice",
                "category": "physics",
                "Verified_Classes": "Gold subset",
                "source_variant": "verified",
                "source_subset_label": "Gold subset",
                "source_dataset_id": "skylenage/HLE-Verified",
                "source_dataset_revision": "0bc83643672d4f68a5f89998617a639d85e7318b",
                "source_paper_version": "arXiv:2602.13964v3",
            }
        ],
    )
    items, _ = _make_dataset_subset(
        dataset="hle",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
        hle_variant="verified",
    )
    engine = _FakeDebateEngine(
        outputs_by_call=[
            [
                "Confidence: 0.80\n\\boxed{B}",
                "Confidence: 0.60\n\\boxed{B}",
            ],
            [
                "The strongest support is for blue.",
            ],
            [
                "Confidence: 0.91\n\\boxed{B}",
            ],
        ]
    )

    results = run_debate(
        dataset="hle",
        items=items,
        engine=engine,
        n_agents=2,
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
    assert row["final_judge_answer"] == "B"
    assert row["judge_trace"]["judge_extractor_trace_source"] == "retry"
    assert row["judge_trace"]["judge_extractor_trace"]["normalized_answer"] == "B"
    assert row["judge_trace"]["judge_scoring_result"]["correct"] == 1
    assert row["judge_trace"]["judge_raw_extractor_trace"]["parse_success"] is False
    assert row["judge_trace"]["judge_retry_extractor_trace"]["normalized_answer"] == "B"


def test_run_debate_hle_multi_round_uses_chat_message_shape_for_debate_sharing(tmp_path: Path):
    dataset_path = tmp_path / "verified_multiround.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "hle-item-round2",
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "blue",
                "answer_type": "multipleChoice",
                "category": "physics",
                "Verified_Classes": "Gold subset",
                "source_variant": "verified",
                "source_subset_label": "Gold subset",
                "source_dataset_id": "skylenage/HLE-Verified",
                "source_dataset_revision": "0bc83643672d4f68a5f89998617a639d85e7318b",
                "source_paper_version": "arXiv:2602.13964v3",
            }
        ],
    )
    items, _ = _make_dataset_subset(
        dataset="hle",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
        hle_variant="verified",
    )
    engine = _FakeDebateEngine(
        outputs_by_call=[
            [
                "Round 1 agent 1.\nConfidence: 0.80\n\\boxed{B}",
                "Round 1 agent 2.\nConfidence: 0.60\n\\boxed{B}",
            ],
            [
                "Round 2 agent 1.\nConfidence: 0.82\n\\boxed{B}",
                "Round 2 agent 2.\nConfidence: 0.61\n\\boxed{B}",
            ],
            [
                "Judge picks blue.\nConfidence: 0.90\n\\boxed{B}",
            ],
        ]
    )

    results = run_debate(
        dataset="hle",
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
    assert row["final_judge_answer"] == "B"
    round2_context = engine.calls[1][0]
    assert round2_context[-1]["role"] == "user"
    assert "These are the prior-round outputs from other agents:" in round2_context[-1]["content"]
    assert "Agent 1:" in round2_context[-1]["content"]
