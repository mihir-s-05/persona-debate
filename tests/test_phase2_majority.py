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


def _fake_persona_response(prompt: str, all_content: str) -> str | None:
    text = f"{prompt}\n{all_content}".lower()
    if "classify this gpqa item into exactly one domain family" in text:
        return json.dumps({"judge_family": "physics"})
    if "you generate reasoning-diversity axes" in text or "propose reasoning-relevant axes" in text:
        return json.dumps(
            {
                "axes": [
                    {
                        "axis_id": "verification_style",
                        "name": "Verification Style",
                        "low_desc": "Validate cautiously before committing.",
                        "high_desc": "Advance quickly, then verify later.",
                        "notes": "Keep the axis answer-agnostic.",
                    }
                ]
            }
        )
    if "generate the full persona population jointly" in text:
        return json.dumps(
            {
                "descriptors": [
                    {
                        "persona_id": f"persona_{idx + 1}",
                        "name": f"P{idx + 1}",
                        "axis_interpretation": {"verification_style": f"mode_{idx}"},
                        "short_rule": f"rule_{idx}",
                        "reasoning_summary": f"reasoning summary {idx}",
                    }
                    for idx in range(5)
                ]
            }
        )
    if "expand each descriptor into a compact" in text:
        persona_id = next(
            (f"persona_{idx}" for idx in range(1, 6) if f"persona_{idx}" in text),
            "persona_1",
        )
        return json.dumps(
            {
                "persona_id": persona_id,
                "title": f"Card {persona_id}",
                "core_reasoning_strategy": f"strategy {persona_id}",
                "priorities": ["track support"],
                "distrusts": ["unsupported jumps"],
                "decomposition_style": "stepwise",
                "revision_policy": "revise on evidence",
                "confidence_policy": "explicit",
                "failure_mode_to_avoid": "answer leakage",
                "system_prompt": f"System prompt for {persona_id}.",
            }
        )
    if (
        "generate a constrained judge card" in text
        or "generate reusable benchmark-level judge cards" in text
        or "generate a reusable benchmark-level judge card" in text
    ):
        return json.dumps(
            {
                "judge_id": "judge_physics",
                "judge_family": "physics",
                "domain_scope": "physics",
                "evaluation_priorities": ["score transcript support"],
                "tie_break_policy": "prefer explicit support",
                "independent_resolve_policy": "limited_check_only",
                "answer_format_policy": "strict",
                "confidence_policy": "optional",
                "system_prompt": "Judge prompt",
            }
        )
    return None


def _is_special_output_batch(outputs: list[object]) -> bool:
    markers = ('"judge_family"', '"judge_id"', '"descriptors"', '"axes"', '"persona_id"')
    return bool(outputs) and all(any(marker in str(output or "") for marker in markers) for output in outputs)


class _FakeBatchEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "fake-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name
        self.calls: list[tuple[list[list[dict[str, str]]], int | None, dict[str, object] | None]] = []

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        special_outputs: list[str] = []
        for context in contexts:
            prompt = str(context[-1].get("content", ""))
            all_content = " ".join(str(msg.get("content", "")) for msg in context)
            response = _fake_persona_response(prompt, all_content)
            if response is None:
                special_outputs = []
                break
            special_outputs.append(response)
        if special_outputs:
            output_idx = getattr(self, "_output_idx", 0)
            if output_idx < len(self.outputs_by_call) and _is_special_output_batch(self.outputs_by_call[output_idx]):
                setattr(self, "_output_idx", output_idx + 1)
            if progress_callback is not None:
                progress_callback(len(special_outputs))
            return special_outputs
        call_idx = getattr(self, "_output_idx", 0)
        assert call_idx < len(self.outputs_by_call), f"Unexpected extra engine call {call_idx}"
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts), (call_idx, len(outputs), len(contexts))
        self.calls.append(([[dict(msg) for msg in ctx] for ctx in contexts], batch_size, sampling_kwargs))
        setattr(self, "_output_idx", call_idx + 1)
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
        persona_backend="llm",
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
        persona_backend="llm",
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
        persona_backend="llm",
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


def test_single_mode_requests_32768_max_tokens(tmp_path: Path):
    dataset_path = tmp_path / "aime_single.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 1+1?", "answer": "2", "id": "aime-single-1"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    engine = _FakeBatchEngine(outputs_by_call=[["Answer. \\boxed{2}"]])

    rows = run_sampled(
        dataset="aime25",
        items=items,
        engine=engine,
        n_samples=1,
        batch_size=4,
        mode_label="single",
    )

    assert rows[0]["final_answer"] == "2"
    assert engine.calls[0][2] == {"max_tokens": 32768}


