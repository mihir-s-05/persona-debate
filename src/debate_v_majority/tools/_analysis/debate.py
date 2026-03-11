from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .common import entropy_from_counts, mean


def assistant_messages(agent_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(agent_messages, list):
        return []
    assistant_only = [msg for msg in agent_messages if isinstance(msg, dict) and msg.get("role") == "assistant"]
    if assistant_only:
        return assistant_only
    return [msg for msg in agent_messages if isinstance(msg, dict)]


def coerce_round_output(
    output: Any,
    *,
    assistant_content: str | None,
    parsed_answer: str | None,
) -> dict[str, Any]:
    row = dict(output) if isinstance(output, dict) else {}
    text = None
    for key in ("assistant_content", "private_raw_response", "raw_response", "public_rationale"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None and isinstance(assistant_content, str) and assistant_content.strip():
        text = assistant_content
    if text is not None:
        row.setdefault("assistant_content", text)
        row.setdefault("private_raw_response", text)
    if row.get("final_answer") is None:
        row["final_answer"] = parsed_answer
    return row


def round_output_text(output: Any) -> str | None:
    row = dict(output) if isinstance(output, dict) else {}
    for key in ("assistant_content", "private_raw_response", "raw_response", "public_rationale"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def resolve_debate_round_outputs(
    rec: dict[str, Any],
    *,
    dataset: str,
    adapters: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if dataset not in adapters:
        raise ValueError(f"unknown dataset {dataset}")
    adapter = adapters[dataset]
    outputs = rec.get("agent_round_outputs") or []
    parsed = rec.get("agent_round_parsed_answers") or []
    agent_responses = rec.get("agent_responses") or []
    raw_task = rec.get("raw_task") or rec

    n_agents = max(
        int(rec.get("n_agents") or 0),
        len(outputs) if isinstance(outputs, list) else 0,
        len(parsed) if isinstance(parsed, list) else 0,
        len(agent_responses) if isinstance(agent_responses, list) else 0,
    )

    rounds_from_outputs = max((len(agent_outputs) for agent_outputs in outputs), default=0) if isinstance(outputs, list) else 0
    rounds_from_parsed = max((len(agent_answers) for agent_answers in parsed), default=0) if isinstance(parsed, list) else 0
    rounds_from_msgs = max(
        (len(assistant_messages(agent_messages)) for agent_messages in agent_responses),
        default=0,
    ) if isinstance(agent_responses, list) else 0
    n_rounds = max(int(rec.get("n_rounds") or 0), rounds_from_outputs, rounds_from_parsed, rounds_from_msgs)

    normalized: list[list[dict[str, Any]]] = []
    for agent_idx in range(n_agents):
        output_rows = outputs[agent_idx] if agent_idx < len(outputs) and isinstance(outputs, list) else []
        parsed_answers = parsed[agent_idx] if agent_idx < len(parsed) and isinstance(parsed, list) else []
        assistant_rows = assistant_messages(agent_responses[agent_idx]) if agent_idx < len(agent_responses) and isinstance(agent_responses, list) else []
        allow_legacy_transcript_parse = not output_rows and not parsed_answers
        agent_normalized: list[dict[str, Any]] = []
        for round_idx in range(n_rounds):
            assistant_content = None
            if round_idx < len(assistant_rows):
                content = assistant_rows[round_idx].get("content")
                assistant_content = str(content) if content is not None else ""
            parsed_answer = parsed_answers[round_idx] if round_idx < len(parsed_answers) else None
            if parsed_answer is None and allow_legacy_transcript_parse and assistant_content is not None:
                parsed_answer = adapter.parse_answer(assistant_content, raw_task)
            output = output_rows[round_idx] if round_idx < len(output_rows) else {}
            normalized_output = coerce_round_output(
                output,
                assistant_content=assistant_content,
                parsed_answer=parsed_answer,
            )
            normalized_output.setdefault(
                "analysis_missing_final_answer",
                normalized_output.get("final_answer") is None,
            )
            agent_normalized.append(normalized_output)
        normalized.append(agent_normalized)
    return normalized


@dataclass(frozen=True)
class DebateAnalysisRow:
    outputs: list[list[dict[str, Any]]]
    answers: list[list[str | None]]
    n_agents: int
    n_rounds: int
    round1_answers: list[str | None]
    final_round_answers: list[str | None]
    round1_majority_answer: str | None
    round1_majority_correct: int
    final_round_majority_answer: str | None
    final_round_majority_correct: int
    judge_answer: str | None
    judge_correct: int
    convergence_rows: list[dict[str, Any]]
    answer_change_rows: list[dict[str, Any]]


def build_convergence_rows_from_answers(answers: list[list[str | None]], *, vote_details: Any) -> list[dict[str, Any]]:
    n_rounds = max((len(agent_answers) for agent_answers in answers), default=0)
    rows: list[dict[str, Any]] = []
    for round_idx in range(n_rounds):
        round_answers = [
            agent_answers[round_idx] if round_idx < len(agent_answers) else None
            for agent_answers in answers
        ]
        counts = vote_details(round_answers)["vote_counts"]
        non_none = [answer for answer in round_answers if answer is not None]
        distinct_answers = len(set(non_none))
        unanimous = bool(non_none) and distinct_answers == 1 and len(non_none) == len(round_answers)
        rows.append(
            {
                "round": round_idx + 1,
                "distinct_answers": distinct_answers,
                "unanimous": unanimous,
                "vote_counts": counts,
            }
        )
    return rows


def build_answer_change_rows_from_answers(answers: list[list[str | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_idx, agent_answers in enumerate(answers):
        changed_flags: list[bool] = []
        first_change_round: int | None = None
        prev = None
        for round_idx, answer in enumerate(agent_answers, start=1):
            changed = round_idx > 1 and answer != prev
            changed_flags.append(changed)
            if changed and first_change_round is None:
                first_change_round = round_idx
            prev = answer
        rows.append(
            {
                "agent_index": agent_idx,
                "answers_by_round": list(agent_answers),
                "changed_from_prior_round": changed_flags,
                "first_change_round": first_change_round,
            }
        )
    return rows


def normalize_debate_analysis_row(
    rec: dict[str, Any],
    *,
    dataset: str,
    gt: Any,
    check_correct: Any,
    adapters: dict[str, Any],
    vote_details: Any,
) -> DebateAnalysisRow:
    outputs = resolve_debate_round_outputs(rec, dataset=dataset, adapters=adapters)
    answers = [[output.get("final_answer") for output in agent_outputs] for agent_outputs in outputs]
    n_agents = len(answers)
    n_rounds = max((len(agent_answers) for agent_answers in answers), default=0)

    round1_answers = [agent_answers[0] if agent_answers else None for agent_answers in answers]
    final_round_answers = [agent_answers[-1] if agent_answers else None for agent_answers in answers]

    raw_task = rec.get("raw_task") or {}
    round1_majority_result = rec.get("round1_majority_result") or {}
    round1_majority_answer = round1_majority_result.get("majority_answer", rec.get("round1_majority_answer"))
    if round1_majority_answer is None and round1_answers:
        round1_majority_answer = vote_details(round1_answers)["majority_answer"]
    round1_majority_correct = round1_majority_result.get("majority_correct", rec.get("round1_majority_correct"))
    if round1_majority_correct is None:
        round1_majority_correct = int(
            check_correct(round1_majority_answer, dataset=dataset, gt=gt, raw_task=raw_task) == 1
        )

    final_round_majority_result = rec.get("final_round_majority_result") or {}
    final_round_majority_answer = final_round_majority_result.get(
        "majority_answer",
        rec.get("final_round_majority_answer", rec.get("final_majority_answer")),
    )
    if final_round_majority_answer is None and final_round_answers:
        final_round_majority_answer = vote_details(final_round_answers)["majority_answer"]
    final_round_majority_correct = final_round_majority_result.get(
        "majority_correct",
        rec.get("final_round_majority_correct", rec.get("final_majority_correct")),
    )
    if final_round_majority_correct is None:
        final_round_majority_correct = int(
            check_correct(final_round_majority_answer, dataset=dataset, gt=gt, raw_task=raw_task) == 1
        )

    judge_answer = rec.get("final_judge_answer", rec.get("judge_final_answer"))
    judge_correct = rec.get("final_judge_correct", rec.get("judge_final_correct"))
    if judge_correct is None:
        judge_correct = int(check_correct(judge_answer, dataset=dataset, gt=gt, raw_task=raw_task) == 1)

    stored_convergence = rec.get("convergence_per_round") or []
    convergence_rows = (
        [row for row in stored_convergence if isinstance(row, dict)]
        if stored_convergence
        else build_convergence_rows_from_answers(answers, vote_details=vote_details)
    )
    stored_answer_changes = rec.get("answer_changes_per_agent") or []
    answer_change_rows = (
        [row for row in stored_answer_changes if isinstance(row, dict)]
        if stored_answer_changes
        else build_answer_change_rows_from_answers(answers)
    )

    return DebateAnalysisRow(
        outputs=outputs,
        answers=answers,
        n_agents=n_agents,
        n_rounds=n_rounds,
        round1_answers=round1_answers,
        final_round_answers=final_round_answers,
        round1_majority_answer=round1_majority_answer,
        round1_majority_correct=int(round1_majority_correct or 0),
        final_round_majority_answer=final_round_majority_answer,
        final_round_majority_correct=int(final_round_majority_correct or 0),
        judge_answer=None if judge_answer is None else str(judge_answer),
        judge_correct=int(judge_correct or 0),
        convergence_rows=convergence_rows,
        answer_change_rows=answer_change_rows,
    )


def tokenize_text(text: Any) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text or "").lower()))


def mean_pairwise_jaccard_distance(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    distances: list[float] = []
    for left, right in combinations(texts, 2):
        left_tokens = tokenize_text(left)
        right_tokens = tokenize_text(right)
        union = left_tokens | right_tokens
        if not union:
            distances.append(0.0)
            continue
        overlap = left_tokens & right_tokens
        distances.append(1.0 - (len(overlap) / len(union)))
    return mean(distances)


def compute_debate_row_metrics(
    *,
    rec: dict[str, Any],
    dataset: str,
    gt: Any,
    check_correct: Any,
    adapters: dict[str, Any],
    vote_details: Any,
) -> dict[str, Any]:
    raw_task = rec.get("raw_task") or {}
    normalized = normalize_debate_analysis_row(
        rec,
        dataset=dataset,
        gt=gt,
        check_correct=check_correct,
        adapters=adapters,
        vote_details=vote_details,
    )
    outputs = normalized.outputs
    n_agents = normalized.n_agents
    round1_answers = normalized.round1_answers
    round1_counts: Counter[str | None] = Counter(round1_answers)
    pair_total = 0
    pair_disagree = 0
    for left_idx in range(len(round1_answers)):
        for right_idx in range(left_idx + 1, len(round1_answers)):
            pair_total += 1
            if round1_answers[left_idx] != round1_answers[right_idx]:
                pair_disagree += 1
    round1_unique_non_none = len({answer for answer in round1_answers if answer is not None})

    round_distinct = [
        int(row.get("distinct_answers", 0) or 0)
        for row in normalized.convergence_rows
        if isinstance(row, dict)
    ]
    round1_distinct = round_distinct[0] if round_distinct else round1_unique_non_none
    final_distinct = round_distinct[-1] if round_distinct else round1_unique_non_none
    if round1_distinct > 0:
        convergence_rate = max(0.0, min(1.0, 1.0 - (final_distinct / round1_distinct)))
    else:
        convergence_rate = 0.0

    persona_summaries = rec.get("persona_summaries") or rec.get("sample_persona_summaries") or []
    persona_ids = rec.get("persona_ids") or rec.get("sample_persona_ids") or []
    answer_changes = normalized.answer_change_rows
    revision_rate_by_persona: list[dict[str, Any]] = []
    for agent_idx in range(n_agents):
        outputs_for_agent = outputs[agent_idx] if agent_idx < len(outputs) else []
        persona_summary = persona_summaries[agent_idx] if agent_idx < len(persona_summaries) else {}
        change_row = answer_changes[agent_idx] if agent_idx < len(answer_changes) else {}
        changed_flags = list(change_row.get("changed_from_prior_round") or [])
        change_count = sum(1 for flag in changed_flags if flag)
        revision_opportunities = max(0, len(outputs_for_agent) - 1)
        revision_rate = (change_count / revision_opportunities) if revision_opportunities else 0.0
        revision_rate_by_persona.append(
            {
                "agent_index": agent_idx,
                "persona_id": persona_summary.get("persona_id") or (persona_ids[agent_idx] if agent_idx < len(persona_ids) else None),
                "title": persona_summary.get("title"),
                "short_rule": persona_summary.get("short_rule"),
                "answers_by_round": [output.get("final_answer") for output in outputs_for_agent],
                "change_count": change_count,
                "revision_opportunities": revision_opportunities,
                "revision_rate": revision_rate,
                "first_change_round": change_row.get("first_change_round"),
            }
        )

    round1_rationales = [
        str(agent_outputs[0].get("public_rationale") or "").strip()
        for agent_outputs in outputs
        if agent_outputs and str(agent_outputs[0].get("public_rationale") or "").strip()
    ]
    public_rationale_diversity = mean_pairwise_jaccard_distance(round1_rationales)

    round1_majority_correct = normalized.round1_majority_correct
    final_majority_correct = normalized.final_round_majority_correct
    judge_correct = normalized.judge_correct
    round1_has_correct = any(
        check_correct(answer, dataset=dataset, gt=gt, raw_task=raw_task) == 1
        for answer in round1_answers
        if answer is not None
    )
    round1_correct_minority_present = bool(round1_has_correct and round1_majority_correct == 0)
    final_round_answers = normalized.final_round_answers
    final_round_has_correct = any(
        check_correct(answer, dataset=dataset, gt=gt, raw_task=raw_task) == 1
        for answer in final_round_answers
        if answer is not None
    )

    return {
        "item_uid": rec.get("item_uid"),
        "orig_id": rec.get("orig_id"),
        "round1_answer_entropy": entropy_from_counts(round1_counts),
        "unique_round1_answers": round1_unique_non_none,
        "persona_pair_disagreement_rate": (pair_disagree / pair_total) if pair_total else 0.0,
        "revision_rate_by_persona": revision_rate_by_persona,
        "convergence_rate": convergence_rate,
        "public_rationale_diversity": public_rationale_diversity,
        "judge_majority_disagreed": normalized.judge_answer != normalized.final_round_majority_answer,
        "judge_rescue": bool(judge_correct == 1 and final_majority_correct == 0),
        "judge_harm": bool(judge_correct == 0 and final_majority_correct == 1),
        "round1_correct_minority_present": round1_correct_minority_present,
        "correct_minority_amplified_by_final_majority": bool(round1_correct_minority_present and final_majority_correct == 1),
        "correct_minority_amplified_by_judge": bool(round1_correct_minority_present and judge_correct == 1),
        "correct_minority_suppressed_by_final_round": bool(round1_correct_minority_present and not final_round_has_correct),
    }


def extract_round_answers_debate(
    rec: dict[str, Any],
    *,
    dataset: str,
    adapters: dict[str, Any],
) -> list[list[str | None]]:
    outputs = resolve_debate_round_outputs(rec, dataset=dataset, adapters=adapters)
    return [[round_output.get("final_answer") for round_output in agent_outputs] for agent_outputs in outputs]

