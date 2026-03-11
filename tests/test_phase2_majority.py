from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from debate_v_majority.cli.dataset_eval import _parse_question_answer
from debate_v_majority.cli.sample_runner import run_sampled
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.cli.main_impl import _ensure_dataset_test_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class _FakeBatchEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "fake-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name
        self.calls: list[tuple[list[list[dict[str, str]]], int | None, dict[str, object] | None]] = []

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        call_idx = len(self.calls)
        assert call_idx < len(self.outputs_by_call), f"Unexpected extra engine call {call_idx}"
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts), (call_idx, len(outputs), len(contexts))
        self.calls.append(([[dict(msg) for msg in ctx] for ctx in contexts], batch_size, sampling_kwargs))
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def test_persona_majority_gpqa_emits_vote_details_and_persona_metadata(tmp_path: Path):
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
                "id": "gpqa-majority-1",
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
    engine = _FakeBatchEngine(
        outputs_by_call=[
            [
                f"Reason one. \\boxed{{{gt_letter}}}",
                f"Reason two. \\boxed{{{gt_letter}}}",
                f"Reason three. \\boxed{{{wrong_letter}}}",
            ]
        ]
    )

    rows = run_sampled(
        dataset="gpqa",
        items=items,
        engine=engine,
        n_samples=3,
        batch_size=4,
        mode_label="majority",
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = rows[0]
    assert row["schema_version"] == "phase2.majority.v1"
    assert row["dataset"] == "gpqa"
    assert row["row_origin"] == "standalone_persona_majority"
    assert row["majority_origin"] == "standalone_persona"
    assert row["model_name"] == "fake-model"
    assert row["item_uid"] == "gpqa:gpqa-majority-1"
    assert row["vote_counts"][gt_letter] == 2
    assert row["vote_counts"][wrong_letter] == 1
    assert row["strict_majority_answer"] == gt_letter
    assert row["plurality_answer"] == gt_letter
    assert row["majority_result"]["majority_answer"] == gt_letter
    assert row["majority_result"]["result_origin"] == "standalone_persona"
    assert row["final_majority_answer"] == gt_letter
    assert row["final_majority_correct"] == 1
    assert row["final_answer"] == gt_letter
    assert row["final_correct"] == 1
    assert row["persona_meta"]["artifact_path"] is not None
    assert len(row["sample_persona_ids"]) == 3
    assert len(row["sample_persona_summaries"]) == 3
    assert row["persona_ids"] == row["sample_persona_ids"]
    assert row["persona_summaries"] == row["sample_persona_summaries"]
    assert engine.calls[0][0][0][0]["role"] == "system"


def test_persona_majority_aime_records_tie_case(tmp_path: Path):
    dataset_path = tmp_path / "aime.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 1+1?", "answer": "2", "id": "aime-majority-1"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    engine = _FakeBatchEngine(
        outputs_by_call=[
            [
                "Answer one. \\boxed{2}",
                "Answer two. \\boxed{2}",
                "Answer three. \\boxed{3}",
                "Answer four. \\boxed{3}",
            ]
        ]
    )

    rows = run_sampled(
        dataset="aime25",
        items=items,
        engine=engine,
        n_samples=4,
        batch_size=4,
        mode_label="majority",
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = rows[0]
    assert row["strict_majority_answer"] is None
    assert row["plurality_answer"] is None
    assert row["final_majority_answer"] is None
    assert row["final_majority_correct"] == 0
    assert row["final_answer"] is None
    assert row["final_correct"] == 0


def test_persona_majority_records_plurality_without_strict_majority(tmp_path: Path):
    dataset_path = tmp_path / "aime_plurality.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 1+1?", "answer": "2", "id": "aime-majority-2"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    engine = _FakeBatchEngine(
        outputs_by_call=[["\\boxed{2}", "\\boxed{2}", "\\boxed{3}", "\\boxed{4}"]]
    )

    rows = run_sampled(
        dataset="aime25",
        items=items,
        engine=engine,
        n_samples=4,
        batch_size=4,
        mode_label="majority",
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="heuristic",
        persona_save_artifacts=True,
    )

    row = rows[0]
    assert row["strict_majority_answer"] is None
    assert row["plurality_answer"] == "2"
    assert row["final_majority_answer"] == "2"
    assert row["final_majority_correct"] == 1


def test_gpqa_download_prefers_diamond_config(tmp_path: Path, monkeypatch):
    chosen: dict[str, str] = {}

    def _load_dataset(_name, config, split):
        chosen["config"] = config
        return [{"id": "x"}]

    fake_module = SimpleNamespace(
        get_dataset_config_names=lambda _name: ["gpqa_main", "gpqa_diamond"],
        load_dataset=_load_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    path = tmp_path / "gpqa.jsonl"
    _ensure_dataset_test_jsonl("gpqa", path)

    assert chosen["config"] == "gpqa_diamond"
