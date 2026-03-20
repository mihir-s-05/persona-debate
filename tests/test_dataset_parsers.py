from debate_v_majority.datasets import aime25, gpqa, hle
from debate_v_majority.cli.dataset_eval import _parse_answer as parse_answer_for_eval
from debate_v_majority.cli.judge import _strict_parse_answer as strict_judge_parse_answer


def test_aime25_parse_answer_uses_tail_fallback_for_out_of_range_boxed_value():
    text = "candidate \\boxed{1524}. final answer is 524"
    assert aime25.parse_answer(text, {}) == "524"


def test_gpqa_parse_answer_extracts_boxed_choice():
    text = "reasoning...\n\\boxed{c}"
    assert gpqa.parse_answer(text, {}) == "C"


def test_gpqa_parse_question_answer_supports_numbered_incorrect_keys():
    sample = {
        "Question": "What is correct?",
        "Correct Answer": "blue",
        "Incorrect Answer 1": "red",
        "Incorrect Answer 2": "green",
        "Incorrect Answer 3": "yellow",
    }
    prompt, gt, raw = gpqa.parse_question_answer(sample)
    assert "Choices:" in prompt
    assert gt in {"A", "B", "C", "D"}
    assert raw is sample


def test_gpqa_parse_question_answer_rejects_missing_distractors():
    sample = {
        "Question": "What is correct?",
        "Correct Answer": "blue",
        "Incorrect Answer 1": "red",
        "Incorrect Answer 2": "green",
    }
    try:
        gpqa.parse_question_answer(sample)
    except ValueError as exc:
        assert "missing distractor fields" in str(exc)
    else:
        raise AssertionError("Expected malformed GPQA row to fail fast")


def test_hle_parse_question_answer_multiple_choice_normalizes_answer_text():
    sample = {
        "id": "hle-mcq-1",
        "question": "Which option is correct?\nA) red\nB) blue\nC) green",
        "answer": "blue",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
    }
    prompt, gt, raw = hle.parse_question_answer(sample)
    assert "\\boxed{A}" in prompt
    assert gt == "B"
    assert raw["answer_format_type"] == "multiple_choice"
    assert raw["domain_family"] == "physical_sciences"


def test_hle_parse_question_answer_supports_latex_itemized_choices():
    sample = {
        "id": "hle-mcq-itemized-1",
        "question": (
            "Consider the options.\n\n"
            "\\begin{itemize}\n"
            "  \\item A. red\n"
            "  \\item B. blue\n"
            "  \\item C. green\n"
            "\\end{itemize}"
        ),
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
    }
    prompt, gt, raw = hle.parse_question_answer(sample)
    assert "\\boxed{A}" in prompt
    assert gt == "B"
    assert raw["choice_map"] == {"A": "red", "B": "blue", "C": "green"}


def test_hle_extract_response_parses_confidence_and_choice_label():
    task = {
        "id": "hle-mcq-2",
        "question": "Which option is correct?\nA) red\nB) blue\nC) green",
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Reasoning.\nConfidence: 73%\n\\boxed{b}", task)
    assert extraction["normalized_answer"] == "B"
    assert extraction["parse_success"] is True
    assert extraction["normalized_confidence"] == 0.73
    assert extraction["confidence_parse_failed"] is False


def test_hle_extract_response_strips_markdown_bold_cues_for_multiple_choice_text_answers():
    task = {
        "id": "hle-mcq-3",
        "question": "Which option is correct?\nA) red\nB) blue\nC) green",
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Reasoning.\n**Final Answer**: blue", task)
    assert extraction["candidate_answer"] == "B"
    assert extraction["normalized_answer"] == "B"
    assert extraction["parse_success"] is True


def test_hle_extract_response_accepts_qed_arrow_for_multiple_choice():
    task = {
        "id": "hle-mcq-4",
        "question": "Which option is correct?\nA) red\nB) blue\nC) green",
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "physics",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Reasoning.\nQED ==> B", task)
    assert extraction["candidate_answer"] == "B"
    assert extraction["normalized_answer"] == "B"
    assert extraction["parse_success"] is True


def test_hle_infer_domain_family_uses_judge_bank_family_names():
    assert hle.infer_domain_family({"category": "math"}) == "math"
    assert hle.infer_domain_family({"category": "biology/medicine"}) == "medicine"
    assert hle.infer_domain_family({"category": "computer science/ai"}) == "computer_science"
    assert hle.infer_domain_family({"category": "humanities/social science"}) == "humanities"


def test_hle_numeric_exact_scoring_normalizes_integer_strings():
    task = {
        "id": "hle-num-1",
        "question": "Compute the value.",
        "answer": "0042",
        "answer_type": "exactMatch",
        "category": "math",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("We compute it.\nFinal answer: 42", task)
    scoring = hle.score_answer(extraction["normalized_answer"], task)
    assert extraction["normalized_answer"] == "42"
    assert scoring["match_type"] == "numeric_exact"
    assert scoring["correct"] == 1


def test_hle_freeform_exact_normalizes_text_commands():
    task = {
        "id": "hle-free-1",
        "question": "Name the concept.",
        "answer": "\\text{Bayes theorem}",
        "answer_type": "exactMatch",
        "category": "humanities/social science",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Answer: Bayes theorem", task)
    scoring = hle.score_answer(extraction["normalized_answer"], task)
    assert extraction["normalized_answer"] == "bayes theorem"
    assert scoring["match_type"] == "freeform_verified_rule_set"
    assert scoring["correct"] == 1


def test_hle_numeric_exact_strips_markdown_bold_cues_before_fallback_capture():
    task = {
        "id": "hle-num-2",
        "question": "Compute the value.",
        "answer": "42",
        "answer_type": "exactMatch",
        "category": "math",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Reasoning.\n**Final Answer**: 42", task)
    assert extraction["candidate_answer"] == "42"
    assert extraction["normalized_answer"] == "42"
    assert extraction["parse_success"] is True


def test_hle_numeric_exact_accepts_qed_arrow():
    task = {
        "id": "hle-num-3",
        "question": "Compute the value.",
        "answer": "42",
        "answer_type": "exactMatch",
        "category": "math",
        "Verified_Classes": "Gold subset",
    }
    extraction = hle.extract_response("Reasoning.\nQED ==> 42", task)
    scoring = hle.score_answer(extraction["normalized_answer"], task)
    assert extraction["candidate_answer"] == "42"
    assert extraction["normalized_answer"] == "42"
    assert scoring["correct"] == 1


def test_hle_freeform_exact_normalizes_spacing_before_latex_commands():
    task = {
        "id": "hle-free-3",
        "question": "State the exact expression.",
        "answer": "\\frac{2 \\sqrt{3}}{\\pi}",
        "answer_type": "exactMatch",
        "category": "math",
        "Verified_Classes": "Gold subset",
    }
    scoring = hle.score_answer("\\frac{2\\sqrt{3}}{\\pi}", task)
    assert scoring["predicted_answer"] == "\\frac{2\\sqrt{3}}{\\pi}"
    assert scoring["expected_answer"] == "\\frac{2\\sqrt{3}}{\\pi}"
    assert scoring["correct"] == 1


def test_hle_freeform_scoring_uses_verified_answer_aliases():
    task = {
        "id": "hle-free-2",
        "question": "Name the concept.",
        "answer": "United States of America",
        "answer_type": "exactMatch",
        "category": "humanities/social science",
        "Verified_Classes": "Gold subset",
        "answer_aliases": ["USA", "The United States"],
        "verify_meta_info": {
            "answer_verify": {
                "accepted_answers": ["U.S.A.", "United States"],
            }
        },
    }
    scoring = hle.score_answer("the united states", task)
    assert scoring["match_type"] == "freeform_verified_rule_set"
    assert scoring["correct"] == 1
    assert "united states" in scoring["accepted_answers"]


def test_cli_parse_answer_returns_none_when_no_boxed_for_aime():
    raw_task = {"problem": "What is 1+1?", "answer": "2"}
    text = "I need to think about this more carefully before answering."
    assert parse_answer_for_eval("aime25", text, raw_task) is None


def test_aime_strict_judge_parse_rejects_out_of_range_boxed_answer():
    raw_task = {"problem": "What is 1+1?", "answer": "2"}
    assert strict_judge_parse_answer("aime25", "\\boxed{1002}", raw_task) is None


def test_gpqa_strict_judge_parse_requires_boxed_choice():
    raw_task = {
        "Question": "Which option is correct?",
        "Correct Answer": "blue",
        "Incorrect Answer 1": "red",
        "Incorrect Answer 2": "green",
        "Incorrect Answer 3": "yellow",
    }
    assert strict_judge_parse_answer("gpqa", "Final answer: B", raw_task) is None
    assert strict_judge_parse_answer("gpqa", "\\boxed{B}", raw_task) == "B"

