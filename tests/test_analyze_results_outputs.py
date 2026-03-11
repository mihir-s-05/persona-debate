from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from debate_v_majority.tools import analyze_results as ar
from debate_v_majority.tools._analysis.debate import resolve_debate_round_outputs
from debate_v_majority.tools._analysis.runmeta import load_adapters


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_plurality_vote_keeps_unparsed_answers_in_tally():
    assert ar.plurality_vote(["A", None]) is None
    assert ar.plurality_vote(["A", "A", None]) == "A"
    assert ar.plurality_vote([None, None, "A"]) is None


def test_resolve_debate_round_outputs_keeps_missing_final_answer_explicit():
    outputs = resolve_debate_round_outputs(
        {
            "dataset": "gpqa",
            "n_agents": 1,
            "n_rounds": 1,
            "agent_round_outputs": [[{"public_rationale": "reason only"}]],
            "agent_round_parsed_answers": [[None]],
            "agent_responses": [[{"role": "assistant", "content": "Final answer is probably A"}]],
        },
        dataset="gpqa",
        adapters=load_adapters(),
    )

    assert outputs[0][0]["final_answer"] is None
    assert outputs[0][0]["analysis_missing_final_answer"] is True


def test_analyze_results_raises_on_malformed_run_metadata(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    _write_jsonl(
        results_dir / "broken.jsonl",
        [
            {
                "schema_version": "phase2.single.v1",
                "dataset": "gpqa",
                "mode": "mystery-mode",
                "item_uid": "gpqa:item-bad",
            }
        ],
    )

    with pytest.raises(ValueError, match="Unrecognized run metadata"):
        ar.analyze(results_dir, out_dir)


def test_analyze_results_writes_metrics_only_outputs(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    gpqa_quick = results_dir / "gpqa_quick"

    run_name = "single_custom_name.jsonl"
    _write_jsonl(
        gpqa_quick / run_name,
        [
            {
                "schema_version": "phase2.single.v1",
                "dataset": "gpqa",
                "mode": "single",
                "row_origin": "single_independent",
                "model_name": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-0",
                "dataset_revision": "test",
                "orig_id": 1,
                "answer": "A",
                "final_answer": "A",
                "final_correct": 1,
                "sample_completions": ["\\boxed{A}"],
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "gemini-3-flash-preview"
    ar.analyze(results_dir, out_dir)

    summary_path = out_dir / "summary.json"
    tables_path = out_dir / "tables.md"
    assert summary_path.exists()
    assert tables_path.exists()
    assert not (out_dir / "examples.json").exists()
    assert not (out_dir / "case_studies.md").exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "examples" not in payload


def test_analyze_results_uses_row_schema_and_item_uid_for_phase2_comparisons(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    runs_dir = results_dir / "custom_runs"

    _write_jsonl(
        runs_dir / "standalone_persona_results.jsonl",
        [
            {
                "schema_version": "phase2.majority.v1",
                "dataset": "gpqa",
                "mode": "majority",
                "row_origin": "standalone_persona_majority",
                "majority_origin": "standalone_persona",
                "model_name": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-1",
                "dataset_revision": "test",
                "orig_id": 11,
                "answer": "A",
                "sample_completions": ["\\boxed{A}", "\\boxed{A}", "\\boxed{B}"],
                "sample_parsed_answers": ["A", "A", "B"],
                "majority_result": {
                    "result_kind": "standalone_majority",
                    "result_origin": "standalone_persona",
                    "vote_counts": {"A": 2, "B": 1},
                    "strict_majority_answer": "A",
                    "plurality_answer": "A",
                    "majority_answer": "A",
                    "majority_correct": 1,
                },
                "final_majority_answer": "A",
                "final_majority_correct": 1,
                "final_answer": "A",
                "final_correct": 1,
            }
        ],
    )
    _write_jsonl(
        runs_dir / "standalone_sampling_results.jsonl",
        [
            {
                "schema_version": "phase2.majority_compat.v1",
                "dataset": "gpqa",
                "mode": "majority",
                "row_origin": "standalone_sampling_majority",
                "majority_origin": "standalone_sampling",
                "model_name": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-2",
                "dataset_revision": "test",
                "orig_id": 22,
                "answer": "A",
                "sample_completions": ["\\boxed{A}", "\\boxed{B}", "\\boxed{C}"],
                "sample_parsed_answers": ["A", "B", "C"],
                "majority_result": {
                    "result_kind": "standalone_majority",
                    "result_origin": "standalone_sampling",
                    "vote_counts": {"A": 1, "B": 1, "C": 1},
                    "strict_majority_answer": None,
                    "plurality_answer": None,
                    "majority_answer": None,
                    "majority_correct": 0,
                },
                "final_majority_answer": None,
                "final_majority_correct": 0,
                "final_answer": None,
                "final_correct": 0,
            }
        ],
    )
    _write_jsonl(
        runs_dir / "debate_shared_round1_results.jsonl",
        [
            {
                "schema_version": "phase2.debate.v1",
                "dataset": "gpqa",
                "mode": "debate",
                "row_origin": "debate_judge",
                "debater_model": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-1",
                "dataset_revision": "test",
                "orig_id": 999,
                "answer": "A",
                "n_agents": 3,
                "n_rounds": 1,
                "final_majority_answer": "B",
                "final_majority_correct": 0,
                "final_judge_answer": "A",
                "final_judge_correct": 1,
                "agent_round_parsed_answers": [["A"], ["B"], ["C"]],
                "agent_responses": [
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{A}"}],
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{B}"}],
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{C}"}],
                ],
                "judge_trace": {"judge_raw_response": "\\boxed{A}"},
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "gemini-3-flash-preview"
    ar.analyze(results_dir, out_dir)

    with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    summaries = payload["run_summaries"]
    assert len(summaries) == 3
    debate_summary = next(s for s in summaries if s["meta"]["mode"] == "debate")
    assert debate_summary["round1_majority_correct"] == 0
    assert debate_summary["round1_majority_none"] == 1
    assert debate_summary["majority_correct"] == 0

    comparison_pooled = payload["comparison_pooled"]
    assert "('gpqa', 'standalone_persona_majority')" in comparison_pooled
    assert "('gpqa', 'standalone_sampling_majority')" in comparison_pooled
    assert "('gpqa', 'debate_round1_majority:3a-1r')" in comparison_pooled

    per_question_keys = payload["per_question"].keys()
    assert any("gpqa:item-1" in key for key in per_question_keys)
    assert not any("legacy_orig_id" in key for key in per_question_keys)


def test_analyze_results_emits_phase3_persona_fidelity_metrics(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    runs_dir = results_dir / "custom_runs"

    _write_jsonl(
        runs_dir / "debate_phase3_metrics.jsonl",
        [
            {
                "schema_version": "phase2.debate.v1",
                "dataset": "gpqa",
                "mode": "debate",
                "row_origin": "debate_judge",
                "debater_model": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-phase3",
                "dataset_revision": "test",
                "orig_id": 123,
                "question": "Which option is correct?",
                "answer": "A",
                "n_agents": 3,
                "n_rounds": 2,
                "persona_summaries": [
                    {"persona_id": "p1", "title": "Careful verifier", "short_rule": "Double-check."},
                    {"persona_id": "p2", "title": "Fast heuristic", "short_rule": "Use the first pattern."},
                    {"persona_id": "p3", "title": "Skeptic", "short_rule": "Attack the common view."},
                ],
                "agent_round_outputs": [
                    [
                        {
                            "private_raw_response": "Reason alpha. \\boxed{A}",
                            "public_rationale": "alpha evidence",
                            "final_answer": "A",
                            "confidence": 0.9,
                        },
                        {
                            "private_raw_response": "Still A. \\boxed{A}",
                            "public_rationale": "alpha corrected",
                            "final_answer": "A",
                            "confidence": 0.95,
                        },
                    ],
                    [
                        {
                            "private_raw_response": "Reason beta. \\boxed{B}",
                            "public_rationale": "beta clue",
                            "final_answer": "B",
                            "confidence": 0.4,
                        },
                        {
                            "private_raw_response": "I switch. \\boxed{A}",
                            "public_rationale": "beta corrected",
                            "final_answer": "A",
                            "confidence": 0.7,
                        },
                    ],
                    [
                        {
                            "private_raw_response": "Reason gamma. \\boxed{B}",
                            "public_rationale": "gamma alternative",
                            "final_answer": "B",
                            "confidence": 0.5,
                        },
                        {
                            "private_raw_response": "Still B. \\boxed{B}",
                            "public_rationale": "gamma stays",
                            "final_answer": "B",
                            "confidence": 0.5,
                        },
                    ],
                ],
                "agent_round_parsed_answers": [["A", "A"], ["B", "A"], ["B", "B"]],
                "convergence_per_round": [
                    {"round": 1, "distinct_answers": 2, "vote_counts": {"A": 1, "B": 2}, "unanimous": False},
                    {"round": 2, "distinct_answers": 2, "vote_counts": {"A": 2, "B": 1}, "unanimous": False},
                ],
                "answer_changes_per_agent": [
                    {"agent_index": 0, "answers_by_round": ["A", "A"], "changed_from_prior_round": [False, False], "first_change_round": None},
                    {"agent_index": 1, "answers_by_round": ["B", "A"], "changed_from_prior_round": [False, True], "first_change_round": 2},
                    {"agent_index": 2, "answers_by_round": ["B", "B"], "changed_from_prior_round": [False, False], "first_change_round": None},
                ],
                "round1_majority_answer": "B",
                "round1_majority_correct": 0,
                "final_round_majority_answer": "A",
                "final_round_majority_correct": 1,
                "final_judge_answer": "A",
                "final_judge_correct": 1,
                "judge_trace": {"judge_raw_response": "\\boxed{A}"},
                "agent_responses": [
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{A}"}, {"role": "user", "content": "d"}, {"role": "assistant", "content": "\\boxed{A}"}],
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{B}"}, {"role": "user", "content": "d"}, {"role": "assistant", "content": "\\boxed{A}"}],
                    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{B}"}, {"role": "user", "content": "d"}, {"role": "assistant", "content": "\\boxed{B}"}],
                ],
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "gemini-3-flash-preview"
    ar.analyze(results_dir, out_dir)

    with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    fidelity_rows = payload["debate_persona_fidelity_rows"]
    assert len(fidelity_rows) == 1
    row = fidelity_rows[0]
    assert row["item_uid"] == "gpqa:item-phase3"
    assert row["unique_round1_answers"] == 2
    assert row["persona_pair_disagreement_rate"] == 2 / 3
    assert row["round1_correct_minority_present"] is True
    assert row["correct_minority_amplified_by_final_majority"] is True
    assert row["correct_minority_amplified_by_judge"] is True
    assert row["correct_minority_suppressed_by_final_round"] is False
    assert len(row["revision_rate_by_persona"]) == 3
    assert row["revision_rate_by_persona"][1]["revision_rate"] == 1.0
    assert row["public_rationale_diversity"] is not None

    summary = payload["persona_fidelity_summary"]["('gpqa', '3a-2r')"]
    assert summary["n_questions"] == 1
    assert summary["unique_round1_answers_mean"] == 2.0
    assert summary["persona_pair_disagreement_rate"] == 2 / 3
    assert summary["correct_minority_amplified_by_final_majority_rate"] == 1.0
    assert summary["correct_minority_amplified_by_judge_rate"] == 1.0
    assert summary["judge_rescue_rate"] == 0.0
    assert summary["judge_harm_rate"] == 0.0

    convergence_summary = payload["convergence_summary"]["('gpqa', '3a-2r')"]["per_round"]
    assert convergence_summary[0]["round"] == 1
    assert convergence_summary[0]["mean_distinct_answers"] == 2.0
    assert convergence_summary[1]["mean_distinct_answers"] == 2.0

    answer_change_summary = payload["answer_change_summary"]["('gpqa', '3a-2r')"]
    assert answer_change_summary["agents_total"] == 3
    assert answer_change_summary["agents_changed_at_least_once"] == 1
    assert answer_change_summary["agents_never_changed"] == 2
    assert answer_change_summary["change_events"] == 1
    assert answer_change_summary["change_event_rate"] == 1 / 3
    assert answer_change_summary["first_change_round_counts"]["2"] == 1
    assert answer_change_summary["by_round"][0]["round"] == 2
    assert answer_change_summary["by_round"][0]["changed"] == 1

    judge_matrix = payload["judge_matrix"]["('gpqa', '3a-2r')"]
    assert judge_matrix["maj=1,judge=1"] == 1
    judge_override = payload["judge_override"]["('gpqa', '3a-2r')"]
    assert judge_override["judge_differs_majority"] == 0
    assert judge_override["judge_differs_plurality"] == 0

    tables_md = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "## Convergence Summary (Debate)" in tables_md
    assert "## Answer Change Summary (Debate)" in tables_md
    assert "## Answer Change By Round (Debate)" in tables_md
    assert "## First Change Distribution (Debate)" in tables_md
    assert "### Judge Behavior (Overrides, Rescues, Harms)" in tables_md


def test_analyze_results_supports_hle_runs_via_registry_adapter(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    runs_dir = results_dir / "hle_runs"

    _write_jsonl(
        runs_dir / "hle_majority.jsonl",
        [
            {
                "schema_version": "phase2.majority.v1",
                "dataset": "hle",
                "mode": "majority",
                "row_origin": "standalone_persona_majority",
                "majority_origin": "standalone_persona",
                "model_name": "gemini-3-flash-preview",
                "item_uid": "hle:item-1",
                "dataset_revision": "test",
                "orig_id": 1,
                "answer": "B",
                "sample_completions": ["Confidence: 0.8\n\\boxed{B}"],
                "sample_parsed_answers": ["B"],
                "majority_result": {
                    "result_kind": "standalone_majority",
                    "result_origin": "standalone_persona",
                    "vote_counts": {"B": 1},
                    "strict_majority_answer": "B",
                    "plurality_answer": "B",
                    "majority_answer": "B",
                    "majority_correct": 1,
                },
                "final_majority_answer": "B",
                "final_majority_correct": 1,
                "final_answer": "B",
                "final_correct": 1,
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "gemini-3-flash-preview"
    ar.analyze(results_dir, out_dir)

    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    summaries = payload["run_summaries"]
    assert len(summaries) == 1
    assert summaries[0]["meta"]["dataset"] == "hle"
    assert "('hle', 'standalone_persona_majority')" in payload["comparison_pooled"]


def test_analyze_results_normalizes_legacy_debate_rows_from_agent_responses(tmp_path: Path):
    results_dir = tmp_path / "results"
    out_dir = tmp_path / "out"
    runs_dir = results_dir / "legacy_debate"

    _write_jsonl(
        runs_dir / "legacy_debate_only_agent_responses.jsonl",
        [
            {
                "schema_version": "phase2.debate.v1",
                "dataset": "gpqa",
                "mode": "debate",
                "row_origin": "debate_judge",
                "debater_model": "gemini-3-flash-preview",
                "item_uid": "gpqa:item-legacy-agent-responses",
                "dataset_revision": "test",
                "orig_id": 314,
                "question": "Which option is correct?",
                "answer": "A",
                "n_agents": 3,
                "n_rounds": 2,
                "round1_majority_answer": None,
                "round1_majority_correct": 0,
                "final_round_majority_answer": "A",
                "final_round_majority_correct": 1,
                "final_judge_answer": "A",
                "final_judge_correct": 1,
                "judge_trace": {"judge_raw_response": "\\boxed{A}"},
                "agent_responses": [
                    [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "First take. \\boxed{A}"},
                        {"role": "user", "content": "debate"},
                        {"role": "assistant", "content": "Still A. \\boxed{A}"},
                    ],
                    [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "Initial view. \\boxed{B}"},
                        {"role": "user", "content": "debate"},
                        {"role": "assistant", "content": "I revise. \\boxed{A}"},
                    ],
                    [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "Alternative. \\boxed{C}"},
                        {"role": "user", "content": "debate"},
                        {"role": "assistant", "content": "I stay with C. \\boxed{C}"},
                    ],
                ],
            }
        ],
    )

    ar.TARGET_MODEL_TAG = "gemini-3-flash-preview"
    ar.analyze(results_dir, out_dir)

    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    fidelity_row = payload["debate_persona_fidelity_rows"][0]
    assert fidelity_row["item_uid"] == "gpqa:item-legacy-agent-responses"
    assert fidelity_row["unique_round1_answers"] == 3
    assert len(fidelity_row["revision_rate_by_persona"]) == 3
    assert fidelity_row["revision_rate_by_persona"][1]["answers_by_round"] == ["B", "A"]
    assert fidelity_row["revision_rate_by_persona"][1]["revision_rate"] == 1.0

    convergence = payload["convergence_summary"]["('gpqa', '3a-2r')"]["per_round"]
    assert convergence[0]["mean_distinct_answers"] == 3.0
    assert convergence[1]["mean_distinct_answers"] == 2.0

    answer_change = payload["answer_change_summary"]["('gpqa', '3a-2r')"]
    assert answer_change["agents_total"] == 3
    assert answer_change["agents_changed_at_least_once"] == 1
    assert answer_change["first_change_round_counts"]["2"] == 1


def test_parse_run_meta_supports_legacy_and_phase2_rows():
    legacy_meta = ar.parse_run_meta(
        Path("legacy_single.jsonl"),
        {
            "dataset": "gpqa",
            "mode": "single",
                "model_name": "gemini-3-flash-preview",
            "orig_id": 5,
        },
    )
    phase2_meta = ar.parse_run_meta(
        Path("phase2_single.jsonl"),
        {
            "schema_version": "phase2.single.v1",
            "dataset": "gpqa",
            "mode": "single",
            "row_origin": "single_independent",
            "model_name": "gemini-3-flash-preview",
            "item_uid": "gpqa:item-5",
            "orig_id": 50,
        },
    )
    phase7_meta = ar.parse_run_meta(
        Path("phase7_single.jsonl"),
        {
            "run_meta": {
                "dataset": "gpqa",
                "dataset_meta": {"subset_size": 1, "seed": 123},
                "output_schema_version": "phase7.logical.v1",
            },
            "strategy": {"mode": "single", "n_samples": 1},
            "task": {"dataset": "gpqa", "item_uid": "gpqa:item-7"},
            "model_name": "gemini-3-flash-preview",
        },
    )
    assert legacy_meta.mode == "single"
    assert legacy_meta.dataset == "gpqa"
    assert phase2_meta.schema_version == "phase2.single.v1"
    assert phase7_meta.dataset == "gpqa"
    assert phase7_meta.schema_version == "phase7.logical.v1"
    assert phase7_meta.n == 1
    assert phase7_meta.seed == 123
    include_legacy, legacy_model = ar.should_include_path(
        Path("legacy_single.jsonl"),
        target_model_tag="gemini-3-flash-preview",
        first_row={"model_name": "gemini-3-flash-preview"},
    )
    include_phase2, phase2_model = ar.should_include_path(
        Path("phase2_single.jsonl"),
        target_model_tag="gemini-3-flash-preview",
        first_row={"model_name": "gemini-3-flash-preview"},
    )
    assert include_legacy is True
    assert legacy_model == "gemini-3-flash-preview"
    assert include_phase2 is True
    assert phase2_model == "gemini-3-flash-preview"


def test_analyze_results_defaults_to_gemini_target_model(monkeypatch):
    monkeypatch.delenv("TARGET_MODEL_TAG", raising=False)
    reloaded = importlib.reload(ar)
    try:
        assert reloaded.TARGET_MODEL_TAG == "gemini-3-flash-preview"
        include, model = reloaded.should_include_path(
            Path("phase7_single.jsonl"),
            target_model_tag=reloaded.TARGET_MODEL_TAG,
            first_row={"model_name": "gemini-3-flash-preview"},
        )
        assert include is True
        assert model == "gemini-3-flash-preview"
    finally:
        importlib.reload(reloaded)
