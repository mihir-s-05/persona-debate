from __future__ import annotations

import json
from pathlib import Path

from debate_v_majority.cli.dataset_eval import _parse_question_answer
from debate_v_majority.cli.debate_runner import run_debate
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.tools import trace2txt
from debate_v_majority.tools.extract_transcripts import extract_from_jsonl_row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class _FakeDebateEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "fake-model") -> None:
        self.outputs_by_call = outputs_by_call
        self.model_name = model_name

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        call_idx = getattr(self, "_call_idx", 0)
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts)
        setattr(self, "_call_idx", call_idx + 1)
        if progress_callback is not None:
            progress_callback(len(outputs))
        return outputs


def test_render_trace2txt_for_debate_row(tmp_path: Path):
    input_path = tmp_path / "debate.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dataset": "gpqa",
                "mode": "debate",
                "item_uid": "gpqa:item-trace",
                "item_display_id": "item-trace",
                "orig_id": 7,
                "question": "Which option is correct?",
                "answer": "A",
                "persona_summaries": [
                    {"persona_id": "p1", "title": "Verifier", "short_rule": "Check details."},
                    {"persona_id": "p2", "title": "Challenger", "short_rule": "Probe weaknesses."},
                ],
                "judge_summary": {
                    "judge_id": "j1",
                    "title": "Balanced selector",
                    "decision_style": "Select from transcript evidence.",
                    "selection_principle": "Prefer the best-supported answer.",
                },
                "persona_meta": {
                    "public_rationale_max_tokens": 96,
                },
                "judge_meta": {
                    "judge_persona_mode": "task_family_generated",
                    "judge_trace_mode": "assistant_transcript",
                    "judge_family_assignment": {"judge_family": "physics", "source": "gpqa_family_llm"},
                },
                "agent_round_outputs": [
                    [
                        {
                            "private_raw_response": "Reason one.\n\\boxed{A}",
                            "public_rationale": "reason one",
                            "final_answer": "A",
                            "confidence": 0.8,
                        },
                        {
                            "private_raw_response": "Updated reason one.\n\\boxed{A}",
                            "public_rationale": "updated reason one",
                            "final_answer": "A",
                            "confidence": 0.9,
                        },
                    ],
                    [
                        {
                            "private_raw_response": "Reason two.\n\\boxed{B}",
                            "public_rationale": "reason two",
                            "final_answer": "B",
                            "confidence": 0.4,
                        },
                        {
                            "private_raw_response": "Updated reason two.\n\\boxed{A}",
                            "public_rationale": "updated reason two",
                            "final_answer": "A",
                            "confidence": 0.7,
                        },
                    ],
                ],
                "convergence_per_round": [
                    {"round": 1, "distinct_answers": 2, "unanimous": False, "vote_counts": {"A": 1, "B": 1}},
                    {"round": 2, "distinct_answers": 1, "unanimous": True, "vote_counts": {"A": 2}},
                ],
                "answer_changes_per_agent": [
                    {"agent_index": 0, "answers_by_round": ["A", "A"], "changed_from_prior_round": [False, False], "first_change_round": None},
                    {"agent_index": 1, "answers_by_round": ["B", "A"], "changed_from_prior_round": [False, True], "first_change_round": 2},
                ],
                "round1_majority_answer": None,
                "round1_majority_correct": 0,
                "final_round_majority_answer": "A",
                "final_round_majority_correct": 1,
                "final_judge_answer": "A",
                "final_judge_correct": 1,
                "judge_trace": {
                    "judge_context_is_full_transcript": True,
                    "judge_context_start_round": 1,
                    "judge_context_end_round": 2,
                    "judge_context": [
                        {"role": "system", "content": "Judge system prompt."},
                        {"role": "user", "content": "Question: Which option is correct?\n\nROUND 1\nAgent 1: \\boxed{A}"},
                    ],
                    "judge_raw_response": "Based on the evidence, the answer is \\boxed{A}",
                },
            }
        ],
    )

    output_path = tmp_path / "trace.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--item", "gpqa:item-trace", "--out", str(output_path)])
    assert exit_code == 0

    text = output_path.read_text(encoding="utf-8")
    assert "Mode: debate" in text
    assert "Item UID: gpqa:item-trace" in text
    assert "Personas" in text
    assert "Verifier" in text
    assert "Judge" in text
    assert "Round 1" in text
    assert "Round-1 majority" in text
    assert "Final-round majority: A (correct=1)" in text
    assert "Judge final: A (correct=1)" in text
    assert "Judge trace mode: assistant_transcript" in text
    assert "Judge Context" in text
    assert "Full transcript: True" in text
    assert "Based on the evidence" in text

    stdout_path = tmp_path / "trace-row-index.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(stdout_path)])
    assert exit_code == 0
    assert "Round 2" in stdout_path.read_text(encoding="utf-8")


def test_render_row_text_includes_thought_and_judge_thinking_metadata() -> None:
    row = {
        "dataset": "hle",
        "mode": "debate",
        "item_uid": "hle:x",
        "question": "Q?",
        "answer": "D",
        "agent_round_outputs": [
            [
                {
                    "thought_summary": None,
                    "call_metadata": {"thought_summary": "Hidden think from API."},
                    "private_raw_response": "visible only",
                    "visible_output": "visible only",
                    "final_answer": "D",
                    "parse_success": True,
                    "confidence": 0.9,
                    "extractor_trace": {},
                    "scoring_result": {"correct": 1},
                }
            ],
        ],
        "judge_trace": {
            "judge_raw_response": "\\boxed{D}",
            "judge_raw_call_metadata": {"thought_summary": "Judge thinks before answering."},
            "judge_retry_raw_response": "\\boxed{D}",
            "judge_retry_call_metadata": {"thought_summary": "Judge retries with a stricter answer."},
        },
    }
    text = trace2txt.render_row_text(row)
    assert "Thought summary (model thinking):" in text
    assert "Hidden think from API." in text
    assert "Response (visible output):" in text
    assert "Judge thought summary (raw call)" in text
    assert "Judge thinks before answering." in text
    assert "Judge thought summary (retry call)" in text
    assert "Judge retries with a stricter answer." in text
    assert "Judge Raw Response (visible output)" in text
    assert "Judge Retry Raw Response (visible output)" in text


def test_extract_transcripts_jsonl_row_includes_thought_summaries(tmp_path: Path) -> None:
    row = {
        "question": "Q?",
        "answer": "D",
        "final_answer": "D",
        "final_correct": 1,
        "agent_round_outputs": [
            [
                {
                    "thought_summary": None,
                    "call_metadata": {"thought_summary": "Agent hidden think."},
                }
            ]
        ],
        "judge_trace": {
            "judge_raw_response": "\\boxed{D}",
            "judge_raw_call_metadata": {"thought_summary": "Judge raw think."},
            "judge_retry_raw_response": "\\boxed{D}",
            "judge_retry_call_metadata": {"thought_summary": "Judge retry think."},
        },
    }
    outfile = extract_from_jsonl_row(row, {}, "case1", str(tmp_path))
    assert outfile is not None
    text = Path(outfile).read_text(encoding="utf-8")
    assert "THOUGHT SUMMARIES (DEBATERS)" in text
    assert "Agent hidden think." in text
    assert "THOUGHT SUMMARY — Judge (raw call)" in text
    assert "Judge raw think." in text
    assert "THOUGHT SUMMARY — Judge (retry call)" in text
    assert "Judge retry think." in text


def test_trace2txt_renders_real_run_debate_judge_summary(tmp_path: Path, monkeypatch):
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
                "id": "gpqa-trace-real",
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
    monkeypatch.setattr(
        "debate_v_majority.shared.token_counting.PromptTokenCounter.count_chat_tokens",
        lambda self, messages: 11,
    )
    _, gt_letter, _ = _parse_question_answer("gpqa", items[0].raw_task)
    wrong_letter = next(letter for letter in ("A", "B", "C", "D") if letter != gt_letter)
    class _FakePersonaEngine:
        model_name = "fake-persona-model"
        provider_name = "fake"

        def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None, model_role=None):
            del batch_size, sampling_kwargs, model_role
            outputs = []
            for ctx in contexts:
                all_content = " ".join(str(message.get("content", "")) for message in ctx)
                prompt = str(ctx[-1].get("content", ""))
                if "Classify this GPQA item into exactly one domain family" in prompt:
                    outputs.append('{"judge_family":"physics"}')
                elif "You generate reasoning-diversity axes" in all_content:
                    outputs.append(
                        json.dumps(
                            {
                                "axes": [
                                    {
                                        "axis_id": "verification_style",
                                        "name": "Verification Style",
                                        "low_desc": "Validate cautiously before committing.",
                                        "high_desc": "Advance quickly, then verify at the end.",
                                        "notes": "Keep the axis answer-agnostic.",
                                    }
                                ]
                            }
                        )
                    )
                elif "Generate the full persona population jointly" in all_content:
                    outputs.append(
                        json.dumps(
                            {
                                "descriptors": [
                                    {
                                        "persona_id": "persona_1",
                                        "name": "Verifier",
                                        "axis_interpretation": {"x": "steady"},
                                        "short_rule": "check details",
                                        "reasoning_summary": "verify claims carefully",
                                    },
                                    {
                                        "persona_id": "persona_2",
                                        "name": "Challenger",
                                        "axis_interpretation": {"x": "skeptical"},
                                        "short_rule": "probe weaknesses",
                                        "reasoning_summary": "stress test arguments",
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
                    )
                elif "Generate a constrained judge card" in prompt:
                    outputs.append(
                        json.dumps(
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
                    )
                else:
                    raise AssertionError(f"Unexpected persona prompt: {prompt}")
            if progress_callback is not None:
                progress_callback(len(outputs))
            return outputs

    debater_engine = _FakeDebateEngine(
        outputs_by_call=[
            [
                f"Reason one.\nConfidence: 0.80\n\\boxed{{{gt_letter}}}",
                f"Reason two.\nConfidence: 0.40\n\\boxed{{{wrong_letter}}}",
            ],
        ]
    )
    judge_engine = _FakeDebateEngine(outputs_by_call=[[f"Confidence: 0.75\n\\boxed{{{gt_letter}}}"]])
    persona_engine = _FakePersonaEngine()
    results = run_debate(
        dataset="gpqa",
        items=items,
        engine=debater_engine,
        judge_engine=judge_engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
        use_personas=True,
        artifacts_dir=tmp_path / "artifacts",
        persona_backend="llm",
        persona_save_artifacts=True,
        persona_generator_engine=persona_engine,
        persona_judge_engine=persona_engine,
        gpqa_family_cache_path=tmp_path / "gpqa_family_cache.json",
    )

    input_path = tmp_path / "debate_real.jsonl"
    _write_jsonl(input_path, [results[0][0]])
    output_path = tmp_path / "trace-real.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(output_path)])
    assert exit_code == 0

    text = output_path.read_text(encoding="utf-8")
    assert "Judge family:" in text
    assert "Domain scope:" in text
    assert "Resolve policy:" in text
    assert "Answer format policy:" in text
    assert "Judge Context" in text


def test_trace2txt_renders_hle_metadata_and_extractor_sections(tmp_path: Path):
    input_path = tmp_path / "hle_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dataset": "hle",
                "mode": "debate",
                "item_uid": "hle:hle-item-1",
                "item_display_id": "hle-item-1",
                "orig_id": 0,
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "B",
                "source_variant": "verified",
                "source_subset_label": "Gold subset",
                "canonical_item_id": "hle-item-1",
                "answer_format_type": "multiple_choice",
                "domain_family": "physical_sciences",
                "source_dataset_revision": "0bc83643672d4f68a5f89998617a639d85e7318b",
                "source_paper_version": "arXiv:2602.13964v3",
                "agent_round_outputs": [
                    [
                        {
                            "private_raw_response": "Reason one.\nConfidence: 0.80\n\\boxed{B}",
                            "public_rationale": "reason one",
                            "final_answer": "B",
                            "confidence": 0.8,
                            "confidence_raw_text": "Confidence: 0.80",
                            "confidence_parse_failed": False,
                            "parse_success": True,
                            "extractor_trace": {
                                "extractor_provenance": "hle_verified.extractor.v1",
                                "candidate_answer": "B",
                                "normalized_answer": "B",
                                "answer_format_type": "multiple_choice",
                            },
                            "scoring_result": {
                                "match_type": "multiple_choice_label",
                                "expected_answer": "B",
                                "predicted_answer": "B",
                                "correct": 1,
                            },
                        }
                    ]
                ],
                "convergence_per_round": [{"round": 1, "distinct_answers": 1, "unanimous": True, "vote_counts": {"B": 1}}],
                "answer_changes_per_agent": [{"agent_index": 0, "answers_by_round": ["B"], "changed_from_prior_round": [False], "first_change_round": None}],
                "round1_majority_answer": "B",
                "round1_majority_correct": 1,
                "final_round_majority_answer": "B",
                "final_round_majority_correct": 1,
                "final_judge_answer": "B",
                "final_judge_correct": 1,
                "judge_trace": {
                    "judge_raw_response": "Confidence: 0.88\n\\boxed{B}",
                    "judge_extractor_trace": {
                        "extractor_provenance": "hle_verified.extractor.v1",
                        "normalized_answer": "B",
                        "parse_success": True,
                        "normalized_confidence": 0.88,
                    },
                    "judge_scoring_result": {
                        "match_type": "multiple_choice_label",
                        "expected_answer": "B",
                        "predicted_answer": "B",
                        "correct": 1,
                    },
                },
            }
        ],
    )

    output_path = tmp_path / "hle_trace.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(output_path)])
    assert exit_code == 0

    text = output_path.read_text(encoding="utf-8")
    assert "HLE Metadata" in text
    assert "Variant: verified" in text
    assert "Answer format: multiple_choice" in text
    assert "Extractor: hle_verified.extractor.v1" in text
    assert "Scoring match type: multiple_choice_label" in text
    assert "Judge Extraction" in text


def test_trace2txt_renders_hle_majority_extraction_details(tmp_path: Path):
    input_path = tmp_path / "hle_majority_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dataset": "hle",
                "mode": "majority",
                "item_uid": "hle:majority-item",
                "item_display_id": "majority-item",
                "question": "Name the country.",
                "answer": "united states of america",
                "source_variant": "verified",
                "source_subset_label": "Gold subset",
                "canonical_item_id": "majority-item",
                "answer_format_type": "freeform_exact",
                "domain_family": "humanities_social_science",
                "sample_completions": ["Answer: the united states"],
                "sample_parsed_answers": ["united states"],
                "sample_extractions": [
                    {
                        "extractor_provenance": "hle_verified.extractor.v1",
                        "parse_success": True,
                        "candidate_answer": "the united states",
                        "normalized_confidence": None,
                        "answer_format_type": "freeform_exact",
                    }
                ],
                "sample_scoring_results": [
                    {
                        "match_type": "freeform_verified_rule_set",
                        "expected_answer": "united states of america",
                        "predicted_answer": "united states",
                        "correct": 1,
                        "accepted_answers": ["united states of america", "united states"],
                    }
                ],
                "vote_counts": {"united states": 1},
                "strict_majority_answer": "united states",
                "plurality_answer": "united states",
                "final_majority_answer": "united states",
                "final_majority_correct": 1,
            }
        ],
    )
    output_path = tmp_path / "hle_majority_trace.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(output_path)])
    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "Majority Trace" in text
    assert "Extractor: hle_verified.extractor.v1" in text
    assert "Scoring match type: freeform_verified_rule_set" in text
    assert "Accepted answers: ['united states of america', 'united states']" in text


def test_trace2txt_renders_single_completion_and_outcome(tmp_path: Path):
    input_path = tmp_path / "single_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dataset": "hle",
                "mode": "single",
                "item_uid": "hle:single-item",
                "item_display_id": "single-item",
                "question": "What is 2 + 2?",
                "answer": "4",
                "source_variant": "verified",
                "answer_format_type": "freeform_exact",
                "sample_completions": ["I compute 2 + 2 = 4.\nFinal answer: 4"],
                "sample_parsed_answers": ["4"],
                "sample_extractions": [
                    {
                        "extractor_provenance": "hle_verified.extractor.v1",
                        "parse_success": True,
                        "candidate_answer": "4",
                        "normalized_confidence": 0.91,
                        "answer_format_type": "freeform_exact",
                    }
                ],
                "sample_scoring_results": [
                    {
                        "match_type": "freeform_verified_rule_set",
                        "expected_answer": "4",
                        "predicted_answer": "4",
                        "correct": 1,
                    }
                ],
                "sample_call_metadata": [
                    {
                        "thought_summary": "Adds the two integers directly before stating the answer."
                    }
                ],
                "final_answer": "4",
                "final_correct": 1,
            }
        ],
    )
    output_path = tmp_path / "single_trace.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(output_path)])
    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "Question" in text
    assert "What is 2 + 2?" in text
    assert "Single Trace" in text
    assert "Thought summary:" in text
    assert "Adds the two integers directly before stating the answer." in text
    assert "Completion:" in text
    assert "Final answer: 4" in text
    assert "Parsed answer: 4" in text
    assert "Correct: 1" in text


def test_trace2txt_prefers_phase7_nested_display_and_trace_blocks(tmp_path: Path):
    input_path = tmp_path / "phase7_trace.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "dataset": "gpqa",
                "mode": "debate",
                "item_uid": "gpqa:item-phase7",
                "question": "Which option is correct?",
                "display": {
                    "persona_summaries": [{"persona_id": "p1", "title": "Verifier"}],
                    "judge_summary": {"judge_id": "j1", "judge_family": "science_selector"},
                    "round1_majority_answer": "B",
                    "final_round_majority_answer": "A",
                    "judge_final_answer": "A",
                },
                "trace": {
                    "per_round_agent_outputs": [
                        [
                            {
                                "private_raw_response": "Reason one.\\n\\boxed{A}",
                                "public_rationale": "reason one",
                                "final_answer": "A",
                                "parse_success": True,
                            }
                        ]
                    ],
                    "judge_trace": {"judge_raw_response": "\\boxed{A}"},
                },
                "final_round_majority_correct": 1,
                "final_judge_correct": 1,
            }
        ],
    )
    output_path = tmp_path / "phase7_trace.txt"
    exit_code = trace2txt.main(["--input", str(input_path), "--row-index", "0", "--out", str(output_path)])
    assert exit_code == 0
    text = output_path.read_text(encoding="utf-8")
    assert "Verifier" in text
    assert "Judge family: science_selector" in text
    assert "Round-1 majority: B" in text
    assert "Final-round majority: A (correct=1)" in text
    assert "Judge final: A (correct=1)" in text


