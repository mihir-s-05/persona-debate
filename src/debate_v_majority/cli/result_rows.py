from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .. import DatasetName, Mode
from ..engines import InferenceResult
from ..personas import PersonaArtifact
from ..shared import majority_vote_details
from .dataset_eval import _check_answer_correctness
from .engine_runtime import _merge_token_counts


def _compute_round_convergence(
    agent_round_outputs: list[list[dict[str, Any]]],
    *,
    n_rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_idx in range(n_rounds):
        answers = [
            agent_outputs[round_idx].get("final_answer") if round_idx < len(agent_outputs) else None
            for agent_outputs in agent_round_outputs
        ]
        vote_counts: dict[str, int] = {}
        for answer in answers:
            key = "<unparsed>" if answer is None else str(answer)
            vote_counts[key] = vote_counts.get(key, 0) + 1
        parsed_answers = [str(answer) for answer in answers if answer is not None]
        distinct_parsed = sorted(set(parsed_answers))
        rows.append(
            {
                "round": round_idx + 1,
                "distinct_answers": len(distinct_parsed),
                "vote_counts": vote_counts,
                "unanimous": len(vote_counts) == 1 and "<unparsed>" not in vote_counts,
                "parsed_answers": [answer if answer is not None else None for answer in answers],
            }
        )
    return rows


def _compute_answer_changes(agent_round_outputs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_idx, outputs in enumerate(agent_round_outputs):
        answers = [output.get("final_answer") for output in outputs]
        changed_flags: list[bool] = []
        first_change_round: int | None = None
        prev = None
        for round_idx, answer in enumerate(answers, start=1):
            changed = round_idx > 1 and answer != prev
            changed_flags.append(changed)
            if changed and first_change_round is None:
                first_change_round = round_idx
            prev = answer
        rows.append(
            {
                "agent_index": agent_idx,
                "answers_by_round": answers,
                "changed_from_prior_round": changed_flags,
                "first_change_round": first_change_round,
            }
        )
    return rows


def _compute_persona_fidelity_metrics(
    agent_round_outputs: list[list[dict[str, Any]]],
    *,
    answer_changes: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
) -> dict[str, Any]:
    round1_answers = [
        agent_outputs[0].get("final_answer") if agent_outputs else None
        for agent_outputs in agent_round_outputs
    ]
    round1_counts: Counter[str | None] = Counter(round1_answers)
    total_answers = sum(round1_counts.values())
    round1_entropy = 0.0
    if total_answers > 0:
        round1_entropy = -sum(
            (count / total_answers) * math.log2(count / total_answers)
            for count in round1_counts.values()
            if count > 0
        )
    unique_round1_answers = len({answer for answer in round1_answers if answer is not None})

    pair_total = 0
    pair_disagree = 0
    for left_idx in range(len(round1_answers)):
        for right_idx in range(left_idx + 1, len(round1_answers)):
            pair_total += 1
            if round1_answers[left_idx] != round1_answers[right_idx]:
                pair_disagree += 1
    disagreement_rate = (pair_disagree / pair_total) if pair_total else 0.0

    revision_rate_by_persona: list[dict[str, Any]] = []
    for agent_idx, agent_outputs in enumerate(agent_round_outputs):
        change_row = answer_changes[agent_idx] if agent_idx < len(answer_changes) else {}
        changed_flags = list(change_row.get("changed_from_prior_round") or [])
        change_count = sum(1 for flag in changed_flags if flag)
        revision_opportunities = max(0, len(agent_outputs) - 1)
        revision_rate = (change_count / revision_opportunities) if revision_opportunities else 0.0
        revision_rate_by_persona.append(
            {
                "agent_index": agent_idx,
                "answers_by_round": [output.get("final_answer") for output in agent_outputs],
                "change_count": change_count,
                "revision_opportunities": revision_opportunities,
                "revision_rate": round(revision_rate, 4),
                "first_change_round": change_row.get("first_change_round"),
            }
        )

    round_distinct = [
        int(row.get("distinct_answers", 0) or 0)
        for row in convergence
        if isinstance(row, dict)
    ]
    round1_distinct = round_distinct[0] if round_distinct else unique_round1_answers
    final_distinct = round_distinct[-1] if round_distinct else unique_round1_answers
    convergence_rate = 0.0
    if round1_distinct > 0:
        convergence_rate = max(0.0, min(1.0, 1.0 - (final_distinct / round1_distinct)))

    round1_rationales = [
        str(agent_outputs[0].get("public_rationale") or "").strip()
        for agent_outputs in agent_round_outputs
        if agent_outputs and str(agent_outputs[0].get("public_rationale") or "").strip()
    ]
    public_rationale_diversity = None
    if len(round1_rationales) >= 2:
        distances: list[float] = []
        for left_idx in range(len(round1_rationales)):
            for right_idx in range(left_idx + 1, len(round1_rationales)):
                left_tokens = set(re.findall(r"[A-Za-z0-9]+", round1_rationales[left_idx].lower()))
                right_tokens = set(re.findall(r"[A-Za-z0-9]+", round1_rationales[right_idx].lower()))
                union = left_tokens | right_tokens
                if not union:
                    distances.append(0.0)
                else:
                    distances.append(1.0 - (len(left_tokens & right_tokens) / len(union)))
        if distances:
            public_rationale_diversity = round(sum(distances) / len(distances), 4)

    return {
        "round1_answer_entropy": round(round1_entropy, 4),
        "unique_round1_answers": unique_round1_answers,
        "persona_pair_disagreement_rate": round(disagreement_rate, 4),
        "revision_rate_by_persona": revision_rate_by_persona,
        "convergence_rate": round(convergence_rate, 4),
        "public_rationale_diversity": public_rationale_diversity,
    }


def _compute_round_token_usage(agent_round_outputs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not agent_round_outputs:
        return []
    n_rounds = max((len(agent_outputs) for agent_outputs in agent_round_outputs), default=0)
    rows: list[dict[str, Any]] = []
    for round_idx in range(n_rounds):
        token_entries = []
        for agent_outputs in agent_round_outputs:
            if round_idx >= len(agent_outputs):
                continue
            call_meta = agent_outputs[round_idx].get("call_metadata") or {}
            token_entries.append(call_meta.get("token_counts"))
        rows.append({"round": round_idx + 1, **_merge_token_counts(token_entries)})
    return rows


def _vote_details(answers: list[str | None]) -> dict[str, Any]:
    details = majority_vote_details(answers)
    return {
        "vote_counts": details["vote_counts"],
        "strict_majority_answer": details["strict_majority_answer"],
        "plurality_answer": details["plurality_answer"],
        "final_majority_answer": details["majority_answer"],
    }


def _vote_result_payload(
    *,
    answers: list[str | None],
    dataset: DatasetName,
    gt_answer: Any,
    raw_task: dict[str, Any],
    result_kind: str,
    result_origin: str,
) -> dict[str, Any]:
    details = majority_vote_details(answers)
    majority_answer = details["majority_answer"]
    return {
        "result_kind": result_kind,
        "result_origin": result_origin,
        "vote_counts": details["vote_counts"],
        "strict_majority_answer": details["strict_majority_answer"],
        "plurality_answer": details["plurality_answer"],
        "majority_answer": majority_answer,
        "majority_correct": _check_answer_correctness(dataset, majority_answer, gt_answer, raw_task),
    }


def _artifact_path_str(
    artifact_path: Path | None,
    *,
    allow_missing: bool = False,
) -> str | None:
    if artifact_path is None:
        return None
    if artifact_path.exists() or allow_missing:
        return str(artifact_path)
    return None


def _persona_runtime_meta(
    artifact: PersonaArtifact | None,
    *,
    artifact_path: Path | None,
    allow_missing_artifact_path: bool,
    persona_sampling_method: str,
    persona_backend: str,
    public_rationale_max_tokens: int | None = None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    meta: dict[str, Any] = {
        "artifact_version": artifact.artifact_version,
        "artifact_path": _artifact_path_str(artifact_path, allow_missing=allow_missing_artifact_path),
        "persona_seed": artifact.persona_seed,
        "generator_model": artifact.generator_model,
        "judge_generator_model": artifact.judge_generator_model,
        "axes_mode": artifact.axes.mode,
        "sampling_method": persona_sampling_method,
        "backend": persona_backend,
        "public_rationale_max_tokens": public_rationale_max_tokens,
    }
    if artifact.slot_layout is not None:
        meta["slot_layout"] = list(artifact.slot_layout)
        meta["n_plain_agents"] = artifact.n_plain_agents
    return meta


def _base_row_fields(
    *,
    dataset: DatasetName,
    item: Any,
    question: str,
    gt_answer: Any,
    raw_task: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "subset_id": item.subset_id,
        "orig_id": item.orig_id,
        "item_uid": item.item_uid,
        "dataset_revision": item.dataset_revision,
        "item_display_id": item.item_display_id,
        "dataset_meta": dict(item.dataset_meta),
        "family": item.family,
        "question": question,
        "answer": gt_answer,
        "raw_task": raw_task,
    }
    for key in ("source_dataset_id", "source_dataset_config", "source_dataset_split"):
        value = raw_task.get(key)
        if value is None:
            value = item.dataset_meta.get(key)
        if value is not None:
            row[key] = value
    if dataset == "hle":
        row.update(
            {
                "source_variant": raw_task.get("source_variant") or item.dataset_meta.get("dataset_variant"),
                "source_subset_label": raw_task.get("source_subset_label") or raw_task.get("Verified_Classes"),
                "canonical_item_id": raw_task.get("id"),
                "answer_format_type": raw_task.get("answer_format_type"),
                "domain_family": raw_task.get("domain_family"),
                "source_dataset_id": raw_task.get("source_dataset_id"),
                "source_dataset_revision": raw_task.get("source_dataset_revision"),
                "source_paper_version": raw_task.get("source_paper_version"),
            }
        )
    return row


def _sample_row_common(
    *,
    mode_label: Mode,
    model_name: str | None,
    model_backend: str,
    n_samples: int,
    sample_completions: list[str],
    sample_call_metadata: list[dict[str, Any] | None],
    sample_parsed: list[str | None],
    sample_extractions: list[dict[str, Any]],
    final_answer: str | None,
    final_correct: int,
) -> dict[str, Any]:
    sample_token_usage = [meta["token_counts"] if meta is not None else None for meta in sample_call_metadata]
    return {
        "mode": mode_label,
        "model_name": model_name,
        "model_backend": model_backend,
        "n_samples": n_samples,
        "sample_completions": sample_completions,
        "sample_call_metadata": sample_call_metadata,
        "sample_parsed_answers": sample_parsed,
        "sample_extractions": [extraction["extractor_trace"] for extraction in sample_extractions],
        "sample_scoring_results": [extraction["scoring_result"] for extraction in sample_extractions],
        "sample_token_usage": sample_token_usage,
        "token_usage_summary": _merge_token_counts(sample_token_usage),
        "final_answer": final_answer,
        "final_correct": final_correct,
    }


__all__ = [
    "_artifact_path_str",
    "_base_row_fields",
    "_compute_answer_changes",
    "_compute_persona_fidelity_metrics",
    "_compute_round_convergence",
    "_compute_round_token_usage",
    "_persona_runtime_meta",
    "_sample_row_common",
    "_vote_details",
    "_vote_result_payload",
]
