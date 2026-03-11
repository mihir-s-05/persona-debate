#!/usr/bin/env python3
"""
Analyze debate / majority / single-run outputs across supported datasets.

Writes:
  - <out-dir>/summary.json
  - <out-dir>/tables.md
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..shared import majority_vote_details
from ._analysis.common import (
    FINDINGS_MD_SECTIONS,
    append_findings_md,
    count_none,
    entropy_from_counts,
    fmt_ci,
    fmt_pct,
    md_table,
    mean,
    median,
    now_iso,
    plurality_vote,
    plurality_vote_ignore_none,
    read_jsonl,
    strict_majority_vote,
    wilson_ci,
)
from ._analysis.debate import (
    DebateAnalysisRow,
    compute_debate_row_metrics,
    extract_round_answers_debate,
    normalize_debate_analysis_row,
    round_output_text,
)
from ._analysis.runmeta import RunMeta, RunSummary, load_adapters, parse_run_meta, should_include_path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_DIR = REPO_ROOT / "_autogen"
TARGET_MODEL_TAG = os.environ.get("TARGET_MODEL_TAG", "gemini-3-flash")
_FINDINGS_MD_SECTIONS = FINDINGS_MD_SECTIONS
_now_iso = now_iso
_mean = mean
_median = median
_md_table = md_table
_round_output_text = round_output_text


def _normalize_debate_analysis_row(
    rec: dict[str, Any],
    *,
    dataset: str,
    gt: Any,
    check_correct: Any,
    adapters: dict[str, Any],
) -> DebateAnalysisRow:
    return normalize_debate_analysis_row(
        rec,
        dataset=dataset,
        gt=gt,
        check_correct=check_correct,
        adapters=adapters,
        vote_details=majority_vote_details,
    )


def _compute_debate_row_metrics(
    *,
    rec: dict[str, Any],
    dataset: str,
    gt: Any,
    check_correct: Any,
    adapters: dict[str, Any],
) -> dict[str, Any]:
    return compute_debate_row_metrics(
        rec=rec,
        dataset=dataset,
        gt=gt,
        check_correct=check_correct,
        adapters=adapters,
        vote_details=majority_vote_details,
    )


def analyze(results_dir: Path, out_dir: Path) -> None:
    adapters = load_adapters()
    _FINDINGS_MD_SECTIONS.clear()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_paths = sorted(results_dir.rglob("*.jsonl"))

    candidate_runs: list[tuple[Path, list[dict[str, Any]], dict[str, Any], RunMeta]] = []
    skipped_paths = 0
    for path in all_paths:
        rows = list(read_jsonl(path))
        if not rows:
            skipped_paths += 1
            continue
        first_row = rows[0]
        meta = parse_run_meta(path, first_row)
        if meta.dataset not in adapters:
            skipped_paths += 1
            continue
        candidate_runs.append((path, rows, first_row, meta))

    # Model filtering: keep only runs attributable to TARGET_MODEL_TAG.
    included_runs: list[tuple[Path, list[dict[str, Any]], RunMeta]] = []
    excluded: Counter[str] = Counter()
    for path, rows, first_row, meta in candidate_runs:
        include, eff = should_include_path(path, target_model_tag=TARGET_MODEL_TAG, first_row=first_row)
        if include:
            eff_meta = meta
            if eff_meta.model_tag is None and eff is not None:
                eff_meta = RunMeta(**{**asdict(meta), "model_tag": eff})
            included_runs.append((path, rows, eff_meta))
        else:
            excluded[eff or "unknown_model"] += 1

    # Inventory: count files by dataset/method.
    inv = defaultdict(int)
    for _, _, meta in included_runs:
        inv[(meta.dataset, meta.method_label)] += 1

    inv_rows = [
        [ds, method, str(cnt)]
        for (ds, method), cnt in sorted(inv.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    append_findings_md(
        "### Inventory\n\n"
        + f"- Target model tag: `{TARGET_MODEL_TAG}`\n"
        + f"- Included files: {len(included_runs)} / {len(all_paths)}\n"
        + f"- Skipped non-run/unsupported files: {skipped_paths}\n"
        + (
            "- Excluded by inferred model tag:\n"
            + _md_table(["Inferred model tag", "Count"], [[k, str(v)] for k, v in excluded.items()])
            + "\n\n"
            if excluded
            else "\n\n"
        )
        + _md_table(["Dataset", "Method", "Runs"], inv_rows)
        + "\n\nNotes:\n"
        + "- This analysis prefers row-schema metadata over filenames and only falls back to filenames for legacy runs.\n"
        + "- Custom run tags are supported as long as the rows carry `mode`, `dataset`, and model metadata.\n"
    )

    # Aggregators
    run_summaries: list[RunSummary] = []

    # Overall pooled final outcomes (dataset, method, cfg_key) -> counts
    pooled = defaultdict(lambda: Counter())
    comparison_pooled = defaultdict(lambda: Counter())

    # Majority voting internals (dataset -> counts)
    maj_patterns = defaultdict(lambda: Counter())
    # Formatting / truncation indicators for single+majority baselines
    baseline_format = defaultdict(lambda: Counter())  # dataset -> counts

    # Debate dynamics
    debate_dyn = defaultdict(lambda: Counter())  # (dataset, cfg) -> counters
    debate_round_agree = defaultdict(lambda: defaultdict(int))  # (dataset,cfg) -> round->unanimous_count
    debate_round_total = defaultdict(lambda: defaultdict(int))  # (dataset,cfg) -> round->total
    debate_round_entropy_sum = defaultdict(lambda: defaultdict(float))
    debate_round_entropy_n = defaultdict(lambda: defaultdict(int))

    # Belief transitions
    trans_correctness = defaultdict(lambda: Counter())  # (dataset,cfg) -> (prev_state->next_state)
    trans_choice_gpqa = defaultdict(lambda: Counter())  # (cfg) -> (A->B etc)
    change_toward_prev_other_majority = defaultdict(lambda: Counter())  # (dataset,cfg) -> counts

    # Judge vs majority within debate
    judge_matrix = defaultdict(lambda: Counter())  # (dataset,cfg) -> (maj_correct/judge_correct combos)
    judge_override = defaultdict(lambda: Counter())  # (dataset,cfg) -> stats
    persona_fidelity = defaultdict(
        lambda: {
            "n_questions": 0,
            "round1_answer_entropy_sum": 0.0,
            "unique_round1_answers_sum": 0.0,
            "pair_disagree": 0.0,
            "pair_total": 0.0,
            "convergence_rate_sum": 0.0,
            "public_rationale_diversity_sum": 0.0,
            "public_rationale_diversity_n": 0,
            "judge_majority_disagreed": 0,
            "judge_rescue": 0,
            "judge_harm": 0,
            "round1_correct_minority_present": 0,
            "correct_minority_amplified_by_final_majority": 0,
            "correct_minority_amplified_by_judge": 0,
            "correct_minority_suppressed_by_final_round": 0,
            "revision_profiles": [],
        }
    )
    debate_persona_fidelity_rows: list[dict[str, Any]] = []
    convergence_summary_raw = defaultdict(lambda: defaultdict(list))  # (dataset,cfg) -> round -> [distinct answers]
    answer_change_summary_raw = defaultdict(
        lambda: {
            "by_round": defaultdict(lambda: {"changed": 0, "opportunities": 0}),
            "first_change_round_counts": Counter(),
            "agents_total": 0,
            "agents_changed": 0,
            "agents_never_changed": 0,
            "change_events": 0,
            "revision_opportunities": 0,
        }
    )

    # Per-question pooled by method (dataset, item_uid, method_key) -> correct/total
    per_question = defaultdict(lambda: Counter())

    # Qualitative example selection (keep small excerpts)
    examples: dict[str, list[dict[str, Any]]] = {}
    for dataset_name in adapters:
        examples[dataset_name] = []
        examples[f"{dataset_name}_judge_harm"] = []
        examples[f"{dataset_name}_unanimous_wrong"] = []
        examples[f"{dataset_name}_lost_correct"] = []

    def cfg_key(meta: RunMeta) -> str:
        if meta.mode != "debate":
            return "-"
        return f"{meta.n_agents}a-{meta.n_rounds}r"

    def item_key(rec: dict[str, Any], *, dataset: str) -> str:
        item_uid = rec.get("item_uid")
        if item_uid:
            return str(item_uid)
        orig_id = rec.get("orig_id")
        return f"{dataset}:legacy_orig_id:{orig_id}"

    def completion_has_boxed_or_choice(text: Any, *, dataset: str) -> bool:
        if not isinstance(text, str):
            return False
        if "\\boxed" in text or "\\fbox" in text:
            return True
        if dataset == "gpqa":
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            tail_lines = lines[-8:] if len(lines) > 8 else lines
            tail = "\n".join(tail_lines)
            if re.search(r"(?i)\b(?:final\s+answer|answer)\b[^A-D]*\(?\s*([ABCD])\s*\)?\b", tail):
                return True
            last = tail_lines[-1] if tail_lines else ""
            if re.fullmatch(r"(?i)\(?\s*([ABCD])\s*\)?", last):
                return True
        return False

    def short_tail(text: Any, n: int = 360) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if len(s) <= n:
            return s
        return s[-n:]

    def check_correct(
        answer: Any,
        *,
        dataset: str,
        gt: Any,
        raw_task: dict[str, Any] | None = None,
    ) -> int:
        adapter = adapters.get(dataset)
        if adapter is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        return int(adapter.check_answer_correctness(answer, gt, raw_task))

    def add_method_result(
        *,
        dataset: str,
        item_uid: str,
        method: str,
        answer: Any,
        correct: int,
        is_none: bool,
    ) -> None:
        comparison_pooled[(dataset, method)]["total"] += 1
        comparison_pooled[(dataset, method)]["correct"] += int(correct)
        comparison_pooled[(dataset, method)]["none"] += int(is_none)
        per_question[(dataset, item_uid, method)]["total"] += 1
        per_question[(dataset, item_uid, method)]["correct"] += int(correct)

    for path, rows, meta in included_runs:

        n_q = 0
        final_correct = 0
        final_none = 0

        judge_correct = 0
        judge_none = 0
        maj_correct_in_debate = 0
        maj_none_in_debate = 0
        round1_maj_correct_in_debate = 0
        round1_maj_none_in_debate = 0

        # Per-file n_samples may appear in records; capture mode-wise max
        observed_n_samples: set[int] = set()

        for rec in rows:
            n_q += 1
            gt = rec.get("answer")
            orig_id = rec.get("orig_id")
            raw_task = rec.get("raw_task") or {}
            q_key = item_key(rec, dataset=meta.dataset)

            if meta.mode == "majority":
                observed_n_samples.add(int(rec.get("n_samples") or 0))
                majority_result = rec.get("majority_result") or {}
                final_ans = majority_result.get("majority_answer", rec.get("final_majority_answer", rec.get("final_answer")))
                final_is_none = final_ans is None
                final_is_correct = int(majority_result.get("majority_correct", rec.get("final_majority_correct", rec.get("final_correct") or 0)))

                final_correct += final_is_correct
                final_none += int(final_is_none)
                add_method_result(
                    dataset=meta.dataset,
                    item_uid=q_key,
                    method=meta.method_label,
                    answer=final_ans,
                    correct=final_is_correct,
                    is_none=final_is_none,
                )

                # majority pattern analysis
                samples = rec.get("sample_parsed_answers") or []
                completions = rec.get("sample_completions") or []
                sample_vote = majority_vote_details(list(samples))
                uniq = len(set(samples))
                maj_patterns[meta.dataset][f"n_samples={len(samples)}"] += 1
                maj_patterns[meta.dataset][f"final_none={int(final_is_none)}"] += 1
                maj_patterns[meta.dataset][f"strict_majority_exists={int(sample_vote['strict_majority_answer'] is not None)}"] += 1
                maj_patterns[meta.dataset][f"unique_answers={uniq}"] += 1
                maj_patterns[meta.dataset][f"any_none_in_samples={int(any(x is None for x in samples))}"] += 1
                maj_patterns[meta.dataset][
                    f"all_none_in_samples={int(all(x is None for x in samples) if samples else 0)}"
                ] += 1
                maj_patterns[meta.dataset][f"n_non_none={sum(x is not None for x in samples)}"] += 1

                # "oracle" baselines: was the correct answer present among parsed samples?
                any_sample_correct = any(
                    check_correct(x, dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1
                    for x in samples
                )
                first_non_none = next((x for x in samples if x is not None), None)
                maj_patterns[meta.dataset]["any_sample_correct"] += int(any_sample_correct)
                maj_patterns[meta.dataset]["first_non_none_correct"] += int(
                    check_correct(first_non_none, dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1
                )

                # Formatting/truncation indicators
                baseline_format[meta.dataset]["total_records"] += 1
                baseline_format[meta.dataset]["total_completions"] += len(completions)
                for comp in completions:
                    if completion_has_boxed_or_choice(comp, dataset=meta.dataset):
                        baseline_format[meta.dataset]["completions_with_boxed_or_choice"] += 1
                    if isinstance(comp, str):
                        baseline_format[meta.dataset]["completion_chars_sum"] += len(comp)
                        baseline_format[meta.dataset]["completion_chars_ge_2000"] += int(len(comp) >= 2000)

            elif meta.mode == "single":
                final_ans = rec.get("final_answer")
                final_is_none = final_ans is None
                final_is_correct = int(rec.get("final_correct") or 0)

                final_correct += final_is_correct
                final_none += int(final_is_none)
                add_method_result(
                    dataset=meta.dataset,
                    item_uid=q_key,
                    method=meta.method_label,
                    answer=final_ans,
                    correct=final_is_correct,
                    is_none=final_is_none,
                )

                completions = rec.get("sample_completions") or []

                baseline_format[meta.dataset]["total_records"] += 1
                baseline_format[meta.dataset]["total_completions"] += len(completions)
                for comp in completions:
                    if completion_has_boxed_or_choice(comp, dataset=meta.dataset):
                        baseline_format[meta.dataset]["completions_with_boxed_or_choice"] += 1
                    if isinstance(comp, str):
                        baseline_format[meta.dataset]["completion_chars_sum"] += len(comp)
                        baseline_format[meta.dataset]["completion_chars_ge_2000"] += int(len(comp) >= 2000)

            elif meta.mode == "debate":
                cfg = cfg_key(meta)
                normalized = _normalize_debate_analysis_row(
                    rec,
                    dataset=meta.dataset,
                    gt=gt,
                    check_correct=check_correct,
                    adapters=adapters,
                )
                outputs = normalized.outputs
                answers = normalized.answers
                n_agents = normalized.n_agents
                n_rounds = normalized.n_rounds
                round1_maj_ans = normalized.round1_majority_answer
                round1_maj_is_correct = normalized.round1_majority_correct
                maj_ans = normalized.final_round_majority_answer
                maj_is_correct = normalized.final_round_majority_correct
                judge_ans = normalized.judge_answer
                judge_is_correct = normalized.judge_correct
                maj_correct_in_debate += maj_is_correct
                maj_none_in_debate += int(maj_ans is None)
                judge_correct += judge_is_correct
                judge_none += int(judge_ans is None)

                # For debate outcomes, treat the run's judge decision as authoritative.
                final_correct += judge_is_correct
                final_none += int(judge_ans is None)
                round1_maj_correct_in_debate += round1_maj_is_correct
                round1_maj_none_in_debate += int(round1_maj_ans is None)
                final_round_answers = normalized.final_round_answers
                add_method_result(
                    dataset=meta.dataset,
                    item_uid=q_key,
                    method=f"debate_judge:{cfg}",
                    answer=judge_ans,
                    correct=judge_is_correct,
                    is_none=judge_ans is None,
                )
                add_method_result(
                    dataset=meta.dataset,
                    item_uid=q_key,
                    method=f"debate_round1_majority:{cfg}",
                    answer=round1_maj_ans,
                    correct=round1_maj_is_correct,
                    is_none=round1_maj_ans is None,
                )
                add_method_result(
                    dataset=meta.dataset,
                    item_uid=q_key,
                    method=f"debate_final_round_majority:{cfg}",
                    answer=maj_ans,
                    correct=maj_is_correct,
                    is_none=maj_ans is None,
                )

                # Judge conditioning context.
                # - "repeat-winner": repo_vote result (unique max with count>1, ignoring None)
                # - "plurality": unique max among non-None, even if count==1
                final_round_non_none = [a for a in final_round_answers if a is not None]
                final_counts_non_none = Counter(final_round_non_none)
                final_max_count = max(final_counts_non_none.values()) if final_counts_non_none else 0
                final_plurality = plurality_vote_ignore_none(final_round_answers)
                repeat_winner_exists = maj_ans is not None

                # Judge-vs-majority matrix
                judge_matrix[(meta.dataset, cfg)][f"maj={maj_is_correct},judge={judge_is_correct}"] += 1
                jo = judge_override[(meta.dataset, cfg)]
                jo["n_questions"] += 1
                jo[f"final_max_count={final_max_count}"] += 1
                jo["final_repeat_winner_exists"] += int(repeat_winner_exists)
                jo["final_repeat_winner_absent"] += int(not repeat_winner_exists)

                jo["judge_equals_majority"] += int(judge_ans == maj_ans)
                jo["judge_differs_majority"] += int(judge_ans != maj_ans)
                jo["judge_equals_majority_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_ans == maj_ans)
                jo["judge_differs_majority_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_ans != maj_ans)
                jo["judge_equals_majority_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_ans == maj_ans)
                jo["judge_differs_majority_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_ans != maj_ans)

                jo["final_plurality_exists"] += int(final_plurality is not None)
                jo["judge_equals_plurality"] += int(judge_ans == final_plurality)
                jo["judge_differs_plurality"] += int(judge_ans != final_plurality)
                jo["judge_equals_plurality_when_repeat_winner_absent"] += int(
                    (not repeat_winner_exists) and (judge_ans == final_plurality)
                )

                jo["judge_correct_when_repeat_winner_exists"] += int(repeat_winner_exists and judge_is_correct == 1)
                jo["judge_total_when_repeat_winner_exists"] += int(repeat_winner_exists)
                jo["judge_correct_when_repeat_winner_absent"] += int((not repeat_winner_exists) and judge_is_correct == 1)
                jo["judge_total_when_repeat_winner_absent"] += int(not repeat_winner_exists)

                jo["judge_rescue_from_repeat_winner_wrong"] += int(
                    repeat_winner_exists and (maj_is_correct == 0) and (judge_is_correct == 1)
                )
                jo["judge_rescue_from_repeat_winner_absent"] += int(
                    (not repeat_winner_exists) and (judge_is_correct == 1)
                )
                jo["judge_harm_over_repeat_winner_correct"] += int(
                    repeat_winner_exists and (maj_is_correct == 1) and (judge_is_correct == 0)
                )

                # Round-level agreement/entropy
                for convergence_row in normalized.convergence_rows:
                    round_no = int(convergence_row.get("round", 0) or 0)
                    if round_no <= 0:
                        continue
                    round_answers = [
                        agent_answers[round_no - 1] if round_no - 1 < len(agent_answers) else None
                        for agent_answers in answers
                    ]
                    counts = Counter(round_answers)
                    unanimous = int(bool(convergence_row.get("unanimous")))
                    round_idx = round_no - 1
                    debate_round_agree[(meta.dataset, cfg)][round_idx] += unanimous
                    debate_round_total[(meta.dataset, cfg)][round_idx] += 1
                    debate_round_entropy_sum[(meta.dataset, cfg)][round_idx] += entropy_from_counts(counts)
                    debate_round_entropy_n[(meta.dataset, cfg)][round_idx] += 1

                # Judge answer provenance: was it in any final-round agent answer?
                if judge_ans not in set(final_round_answers):
                    jo["judge_not_in_final_round_answers"] += 1
                    jo["judge_correct_when_not_in_final_round_answers"] += int(judge_is_correct == 1)
                    jo["judge_total_when_not_in_final_round_answers"] += 1
                else:
                    jo["judge_correct_when_in_final_round_answers"] += int(judge_is_correct == 1)
                    jo["judge_total_when_in_final_round_answers"] += 1
                if judge_ans not in {x for row in answers for x in row}:
                    jo["judge_not_in_any_round_answers"] += 1

                # Ever-correct / lost-correct
                ever_correct = False
                final_round_has_correct = False
                for a in range(n_agents):
                    for r in range(n_rounds):
                        ans = answers[a][r]
                        if ans is None:
                            continue
                        if check_correct(ans, dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1:
                            ever_correct = True
                            if r == n_rounds - 1:
                                final_round_has_correct = True
                debate_dyn[(meta.dataset, cfg)]["ever_correct"] += int(ever_correct)
                debate_dyn[(meta.dataset, cfg)]["final_round_has_correct"] += int(final_round_has_correct)
                debate_dyn[(meta.dataset, cfg)]["lost_correct"] += int(ever_correct and not final_round_has_correct)
                debate_dyn[(meta.dataset, cfg)]["judge_wrong_with_final_correct_present"] += int(
                    (judge_is_correct == 0) and final_round_has_correct
                )
                debate_dyn[(meta.dataset, cfg)]["judge_wrong_with_ever_correct_present"] += int(
                    (judge_is_correct == 0) and ever_correct
                )
                debate_dyn[(meta.dataset, cfg)]["judge_correct"] += judge_is_correct
                debate_dyn[(meta.dataset, cfg)]["majority_correct"] += maj_is_correct
                debate_dyn[(meta.dataset, cfg)]["n_questions"] += 1

                # Mirror some correctness-availability context into judge_override for easier reporting.
                jo["final_round_has_correct"] += int(final_round_has_correct)
                jo["ever_correct"] += int(ever_correct)
                jo["judge_wrong_with_final_correct_present"] += int((judge_is_correct == 0) and final_round_has_correct)
                jo["judge_wrong_with_ever_correct_present"] += int((judge_is_correct == 0) and ever_correct)

                row_metrics = _compute_debate_row_metrics(
                    rec=rec,
                    dataset=meta.dataset,
                    gt=gt,
                    check_correct=check_correct,
                    adapters=adapters,
                )
                debate_persona_fidelity_rows.append(
                    {
                        "dataset": meta.dataset,
                        "cfg": cfg,
                        "run": path.name,
                        **row_metrics,
                    }
                )
                pf = persona_fidelity[(meta.dataset, cfg)]
                pf["n_questions"] += 1
                pf["round1_answer_entropy_sum"] += float(row_metrics["round1_answer_entropy"])
                pf["unique_round1_answers_sum"] += float(row_metrics["unique_round1_answers"])
                disagreement_rate = float(row_metrics["persona_pair_disagreement_rate"])
                pair_total = (n_agents * (n_agents - 1)) / 2 if n_agents >= 2 else 0.0
                pf["pair_disagree"] += disagreement_rate * pair_total
                pf["pair_total"] += pair_total
                pf["convergence_rate_sum"] += float(row_metrics["convergence_rate"])
                if row_metrics["public_rationale_diversity"] is not None:
                    pf["public_rationale_diversity_sum"] += float(row_metrics["public_rationale_diversity"])
                    pf["public_rationale_diversity_n"] += 1
                pf["judge_majority_disagreed"] += int(row_metrics["judge_majority_disagreed"])
                pf["judge_rescue"] += int(row_metrics["judge_rescue"])
                pf["judge_harm"] += int(row_metrics["judge_harm"])
                pf["round1_correct_minority_present"] += int(row_metrics["round1_correct_minority_present"])
                pf["correct_minority_amplified_by_final_majority"] += int(
                    row_metrics["correct_minority_amplified_by_final_majority"]
                )
                pf["correct_minority_amplified_by_judge"] += int(row_metrics["correct_minority_amplified_by_judge"])
                pf["correct_minority_suppressed_by_final_round"] += int(
                    row_metrics["correct_minority_suppressed_by_final_round"]
                )
                pf["revision_profiles"].extend(row_metrics["revision_rate_by_persona"])
                debate_dyn[(meta.dataset, cfg)]["round1_correct_minority_present"] += int(
                    row_metrics["round1_correct_minority_present"]
                )
                debate_dyn[(meta.dataset, cfg)]["correct_minority_amplified_by_final_majority"] += int(
                    row_metrics["correct_minority_amplified_by_final_majority"]
                )
                debate_dyn[(meta.dataset, cfg)]["correct_minority_amplified_by_judge"] += int(
                    row_metrics["correct_minority_amplified_by_judge"]
                )
                debate_dyn[(meta.dataset, cfg)]["correct_minority_suppressed_by_final_round"] += int(
                    row_metrics["correct_minority_suppressed_by_final_round"]
                )

                for convergence_row in normalized.convergence_rows:
                    round_no = int(convergence_row.get("round", 0) or 0)
                    if round_no <= 0:
                        continue
                    convergence_summary_raw[(meta.dataset, cfg)][round_no].append(
                        float(convergence_row.get("distinct_answers", 0) or 0)
                    )

                ac = answer_change_summary_raw[(meta.dataset, cfg)]
                for change_row in normalized.answer_change_rows:
                    changed_flags = list(change_row.get("changed_from_prior_round") or [])
                    changed_any = any(bool(flag) for flag in changed_flags)
                    ac["agents_total"] += 1
                    ac["agents_changed"] += int(changed_any)
                    ac["agents_never_changed"] += int(not changed_any)
                    first_change_round = change_row.get("first_change_round")
                    if first_change_round is not None:
                        ac["first_change_round_counts"][int(first_change_round)] += 1
                    for round_idx, changed in enumerate(changed_flags, start=1):
                        if round_idx == 1:
                            continue
                        ac["by_round"][round_idx]["opportunities"] += 1
                        ac["by_round"][round_idx]["changed"] += int(bool(changed))
                        ac["revision_opportunities"] += 1
                        ac["change_events"] += int(bool(changed))

                # Belief changes: correctness-state transitions + conformity proxy
                for a in range(n_agents):
                    seq = answers[a]
                    for r in range(1, n_rounds):
                        prev = seq[r - 1]
                        cur = seq[r]
                        prev_state = (
                            "none"
                            if prev is None
                            else (
                                "correct"
                                if check_correct(prev, dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1
                                else "wrong"
                            )
                        )
                        cur_state = (
                            "none"
                            if cur is None
                            else (
                                "correct"
                                if check_correct(cur, dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1
                                else "wrong"
                            )
                        )
                        trans_correctness[(meta.dataset, cfg)][f"{prev_state}->{cur_state}"] += 1

                        if prev != cur:
                            debate_dyn[(meta.dataset, cfg)]["answer_changes"] += 1
                        debate_dyn[(meta.dataset, cfg)]["answer_steps"] += 1

                        # GPQA letter-to-letter transitions (excluding None)
                        if meta.dataset == "gpqa" and prev is not None and cur is not None:
                            trans_choice_gpqa[cfg][f"{prev}->{cur}"] += 1

                        # Conformity proxy: did the agent move *toward* prev-round plurality of other agents?
                        others_prev = [answers[aa][r - 1] for aa in range(n_agents) if aa != a]
                        other_plural = plurality_vote(others_prev)
                        if prev != cur:
                            change_toward_prev_other_majority[(meta.dataset, cfg)]["changed"] += 1
                            if cur == other_plural:
                                change_toward_prev_other_majority[(meta.dataset, cfg)]["changed_to_other_plurality"] += 1
                            if prev == other_plural:
                                change_toward_prev_other_majority[(meta.dataset, cfg)]["changed_away_from_other_plurality"] += 1

                # Keep a few examples (short tails only to avoid bloat)
                if judge_is_correct == 1 and maj_is_correct == 0:
                    # Judge rescue: judge correct, majority wrong/None.
                    ex = {
                        "run": path.name,
                        "cfg": cfg,
                        "item_uid": rec.get("item_uid"),
                        "orig_id": orig_id,
                        "gt": gt,
                        "final_majority": maj_ans,
                        "final_judge": judge_ans,
                        "agent_tail": "",
                        "judge_tail": "",
                    }
                    for a in range(n_agents):
                        for r in range(n_rounds):
                            if check_correct(answers[a][r], dataset=meta.dataset, gt=gt, raw_task=raw_task) == 1:
                                agent_output = outputs[a][r] if a < len(outputs) and r < len(outputs[a]) else {}
                                ex["agent_tail"] = short_tail(_round_output_text(agent_output))
                                break
                        if ex["agent_tail"]:
                            break
                    ex["judge_tail"] = short_tail(str((rec.get("judge_trace") or {}).get("judge_raw_response", "")))
                    examples[meta.dataset].append(ex)

                if maj_is_correct == 1 and judge_is_correct == 0:
                    # Judge harm: judge flips a correct majority to wrong.
                    examples[f"{meta.dataset}_judge_harm"].append(
                        {
                            "run": path.name,
                            "cfg": cfg,
                            "item_uid": rec.get("item_uid"),
                            "orig_id": orig_id,
                            "gt": gt,
                            "final_majority": maj_ans,
                            "final_judge": judge_ans,
                            "judge_tail": short_tail(
                                str((rec.get("judge_trace") or {}).get("judge_raw_response", ""))
                            ),
                        }
                    )

                if n_rounds and final_round_answers:
                    final_counts = Counter(final_round_answers)
                    if len(final_counts) == 1:
                        unanimous_ans = next(iter(final_counts.keys()))
                        if unanimous_ans is not None and str(unanimous_ans) != str(gt):
                            examples[f"{meta.dataset}_unanimous_wrong"].append(
                                {
                                    "run": path.name,
                                    "cfg": cfg,
                                    "item_uid": rec.get("item_uid"),
                                    "orig_id": orig_id,
                                    "gt": gt,
                                    "unanimous_final": unanimous_ans,
                                    "judge": judge_ans,
                                    "agent_tail": short_tail(_round_output_text(outputs[0][n_rounds - 1])),
                                }
                            )

                if ever_correct and not final_round_has_correct:
                    examples[f"{meta.dataset}_lost_correct"].append(
                        {
                            "run": path.name,
                            "cfg": cfg,
                            "item_uid": rec.get("item_uid"),
                            "orig_id": orig_id,
                            "gt": gt,
                            "final_majority": maj_ans,
                            "final_judge": judge_ans,
                        }
                    )

            else:
                raise ValueError(f"Unknown mode: {meta.mode}")

        final_incorrect = n_q - final_correct - final_none
        meta_final = meta
        if observed_n_samples:
            meta_final = RunMeta(**{**asdict(meta), "n_samples": max(observed_n_samples)})

        rs = RunSummary(
            meta=meta_final,
            n_questions=n_q,
            final_correct=final_correct,
            final_incorrect=final_incorrect,
            final_none=final_none,
        )
        if meta.mode == "debate":
            rs.judge_correct = judge_correct
            rs.judge_none = judge_none
            rs.judge_incorrect = n_q - judge_correct - judge_none
            rs.majority_correct = maj_correct_in_debate
            rs.majority_none = maj_none_in_debate
            rs.majority_incorrect = n_q - maj_correct_in_debate - maj_none_in_debate
            rs.round1_majority_correct = round1_maj_correct_in_debate
            rs.round1_majority_none = round1_maj_none_in_debate
            rs.round1_majority_incorrect = n_q - round1_maj_correct_in_debate - round1_maj_none_in_debate
        run_summaries.append(rs)

        # pooled final-outcome stats
        cfg = cfg_key(meta_final)
        pooled[(meta_final.dataset, meta_final.method_label, cfg)]["total"] += n_q
        pooled[(meta_final.dataset, meta_final.method_label, cfg)]["correct"] += final_correct
        pooled[(meta_final.dataset, meta_final.method_label, cfg)]["none"] += final_none

    persona_fidelity_summary: dict[str, dict[str, Any]] = {}
    for key, stats in persona_fidelity.items():
        n_questions = int(stats["n_questions"])
        revision_profiles = list(stats["revision_profiles"])
        revision_rate_mean = _mean(
            profile.get("revision_rate", 0.0)
            for profile in revision_profiles
            if isinstance(profile, dict)
        )
        persona_fidelity_summary[str(key)] = {
            "n_questions": n_questions,
            "round1_answer_entropy_mean": (
                float(stats["round1_answer_entropy_sum"]) / n_questions if n_questions else None
            ),
            "unique_round1_answers_mean": (
                float(stats["unique_round1_answers_sum"]) / n_questions if n_questions else None
            ),
            "persona_pair_disagreement_rate": (
                float(stats["pair_disagree"]) / float(stats["pair_total"]) if stats["pair_total"] else None
            ),
            "convergence_rate_mean": (float(stats["convergence_rate_sum"]) / n_questions if n_questions else None),
            "public_rationale_diversity_mean": (
                float(stats["public_rationale_diversity_sum"]) / int(stats["public_rationale_diversity_n"])
                if int(stats["public_rationale_diversity_n"])
                else None
            ),
            "judge_majority_disagreement_rate": (
                int(stats["judge_majority_disagreed"]) / n_questions if n_questions else None
            ),
            "judge_rescue_rate": (int(stats["judge_rescue"]) / n_questions if n_questions else None),
            "judge_harm_rate": (int(stats["judge_harm"]) / n_questions if n_questions else None),
            "round1_correct_minority_present_rate": (
                int(stats["round1_correct_minority_present"]) / n_questions if n_questions else None
            ),
            "correct_minority_amplified_by_final_majority_rate": (
                int(stats["correct_minority_amplified_by_final_majority"]) / n_questions if n_questions else None
            ),
            "correct_minority_amplified_by_judge_rate": (
                int(stats["correct_minority_amplified_by_judge"]) / n_questions if n_questions else None
            ),
            "correct_minority_suppressed_by_final_round_rate": (
                int(stats["correct_minority_suppressed_by_final_round"]) / n_questions if n_questions else None
            ),
            "revision_rate_mean": revision_rate_mean,
        }

    convergence_summary: dict[str, dict[str, Any]] = {}
    for key, per_round in convergence_summary_raw.items():
        round_rows: list[dict[str, Any]] = []
        for round_no in sorted(per_round):
            distinct_values = list(per_round[round_no])
            round_rows.append(
                {
                    "round": round_no,
                    "questions": len(distinct_values),
                    "mean_distinct_answers": _mean(distinct_values),
                    "median_distinct_answers": _median(distinct_values),
                }
            )
        convergence_summary[str(key)] = {"per_round": round_rows}

    answer_change_summary: dict[str, dict[str, Any]] = {}
    for key, stats in answer_change_summary_raw.items():
        by_round_rows: list[dict[str, Any]] = []
        by_round = stats["by_round"]
        for round_no in sorted(by_round):
            changed = int(by_round[round_no]["changed"])
            opportunities = int(by_round[round_no]["opportunities"])
            by_round_rows.append(
                {
                    "round": round_no,
                    "changed": changed,
                    "opportunities": opportunities,
                    "change_rate": (changed / opportunities) if opportunities else None,
                }
            )
        answer_change_summary[str(key)] = {
            "agents_total": int(stats["agents_total"]),
            "agents_changed_at_least_once": int(stats["agents_changed"]),
            "agents_never_changed": int(stats["agents_never_changed"]),
            "change_events": int(stats["change_events"]),
            "revision_opportunities": int(stats["revision_opportunities"]),
            "change_event_rate": (
                int(stats["change_events"]) / int(stats["revision_opportunities"])
                if int(stats["revision_opportunities"])
                else None
            ),
            "first_change_round_counts": dict(stats["first_change_round_counts"]),
            "by_round": by_round_rows,
        }

    # Write summary.json for downstream report generation.
    out_summary = {
        "generated_at": _now_iso(),
        "results_dir": str(results_dir),
        "target_model_tag": TARGET_MODEL_TAG,
        "run_summaries": [
            {
                "meta": asdict(r.meta),
                "n_questions": r.n_questions,
                "final_correct": r.final_correct,
                "final_incorrect": r.final_incorrect,
                "final_none": r.final_none,
                "judge_correct": r.judge_correct,
                "judge_incorrect": r.judge_incorrect,
                "judge_none": r.judge_none,
                "majority_correct": r.majority_correct,
                "majority_incorrect": r.majority_incorrect,
                "majority_none": r.majority_none,
                "round1_majority_correct": r.round1_majority_correct,
                "round1_majority_incorrect": r.round1_majority_incorrect,
                "round1_majority_none": r.round1_majority_none,
            }
            for r in run_summaries
        ],
        "pooled": {str(k): dict(v) for k, v in pooled.items()},
        "comparison_pooled": {str(k): dict(v) for k, v in comparison_pooled.items()},
        "majority_patterns": {ds: dict(c) for ds, c in maj_patterns.items()},
        "baseline_format": {ds: dict(c) for ds, c in baseline_format.items()},
        "debate_dynamics": {str(k): dict(v) for k, v in debate_dyn.items()},
        "judge_matrix": {str(k): dict(v) for k, v in judge_matrix.items()},
        "judge_override": {str(k): dict(v) for k, v in judge_override.items()},
        "trans_correctness": {str(k): dict(v) for k, v in trans_correctness.items()},
        "change_toward_prev_other_plurality": {
            str(k): dict(v) for k, v in change_toward_prev_other_majority.items()
        },
        "trans_choice_gpqa": {str(k): dict(v) for k, v in trans_choice_gpqa.items()},
        "persona_fidelity_summary": persona_fidelity_summary,
        "debate_persona_fidelity_rows": debate_persona_fidelity_rows,
        "convergence_summary": convergence_summary,
        "answer_change_summary": answer_change_summary,
        "per_question": {str(k): dict(v) for k, v in per_question.items()},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, sort_keys=True)

    # Build human-readable tables.md
    lines: list[str] = []
    lines.append("# Auto-Generated Tables\n")
    lines.append(f"- Generated at: `{out_summary['generated_at']}`")
    lines.append(f"- Results dir: `{results_dir}`\n")

    # Run summary table
    rows: list[list[str]] = []
    for r in sorted(run_summaries, key=lambda x: (x.meta.dataset, x.meta.method_label, x.meta.path)):
        m = r.meta
        cfg = cfg_key(m)
        acc = fmt_ci(r.final_correct, r.n_questions)
        none_rate = fmt_pct(r.final_none / r.n_questions) if r.n_questions else "0.0%"
        extra = ""
        if m.mode == "debate":
            extra = (
                f"judge={fmt_pct((r.judge_correct or 0)/r.n_questions)} "
                f"r1maj={fmt_pct((r.round1_majority_correct or 0)/r.n_questions)} "
                f"maj={fmt_pct((r.majority_correct or 0)/r.n_questions)} "
            )
        rows.append(
            [
                m.dataset,
                m.method_label,
                cfg,
                str(m.n or ""),
                str(m.seed or ""),
                acc,
                none_rate,
                extra,
                Path(m.path).name,
            ]
        )
    lines.append("## Per-Run Summary\n")
    lines.append(
        _md_table(
            [
                "Dataset",
                "Method",
                "Cfg",
                "n",
                "Seed",
                "Final Acc",
                "Final None",
                "Debate extras",
                "File",
            ],
            rows,
        )
    )
    lines.append("")

    # Pooled stats table
    pooled_rows: list[list[str]] = []
    for (ds, method, cfg), c in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k_new = int(c["correct"])
        n = int(c["total"])
        none_new = int(c["none"])
        pooled_rows.append(
            [
                ds,
                method,
                cfg,
                str(n),
                fmt_ci(k_new, n),
                fmt_pct(none_new / n if n else 0.0),
            ]
        )
    lines.append("## Pooled Final Accuracy by Dataset/Method/Cfg\n")
    lines.append(
        _md_table(
            ["Dataset", "Method", "Cfg", "Total Q", "Acc", "None"],
            pooled_rows,
        )
    )
    lines.append("")

    comparison_rows: list[list[str]] = []
    for (ds, method), c in sorted(comparison_pooled.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        k_cmp = int(c["correct"])
        n_cmp = int(c["total"])
        none_cmp = int(c["none"])
        comparison_rows.append([ds, method, str(n_cmp), fmt_ci(k_cmp, n_cmp), fmt_pct(none_cmp / n_cmp if n_cmp else 0.0)])
    if comparison_rows:
        lines.append("## Comparison Methods\n")
        lines.append(_md_table(["Dataset", "Method", "Total Q", "Acc", "None"], comparison_rows))
        lines.append("")

    # Debate round dynamics summary
    lines.append("## Debate Round Dynamics\n")
    for (ds, cfg), totals in sorted(debate_round_total.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows2: list[list[str]] = []
        for r in sorted(totals.keys()):
            tot = debate_round_total[(ds, cfg)][r]
            un = debate_round_agree[(ds, cfg)][r]
            ent_sum = debate_round_entropy_sum[(ds, cfg)][r]
            ent_n = debate_round_entropy_n[(ds, cfg)][r]
            ent = ent_sum / ent_n if ent_n else 0.0
            rows2.append([str(r + 1), str(tot), fmt_pct(un / tot if tot else 0.0), f"{ent:.3f}"])
        lines.append(f"### {ds} {cfg}\n")
        lines.append(_md_table(["Round", "Questions", "Unanimous", "Mean entropy (bits)"], rows2))
        lines.append("")

    convergence_rows_md: list[list[str]] = []
    for (ds, cfg), _stats in sorted(convergence_summary_raw.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for row in convergence_summary[str((ds, cfg))]["per_round"]:
            convergence_rows_md.append(
                [
                    ds,
                    cfg,
                    str(row["round"]),
                    str(row["questions"]),
                    f"{row['mean_distinct_answers']:.2f}" if row["mean_distinct_answers"] is not None else "",
                    f"{row['median_distinct_answers']:.2f}" if row["median_distinct_answers"] is not None else "",
                ]
            )
    if convergence_rows_md:
        lines.append("## Convergence Summary (Debate)\n")
        lines.append(
            _md_table(
                ["Dataset", "Cfg", "Round", "Questions", "Mean distinct answers", "Median distinct answers"],
                convergence_rows_md,
            )
        )
        lines.append("")

    answer_change_rows_md: list[list[str]] = []
    for (ds, cfg), _stats in sorted(answer_change_summary_raw.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        stats = answer_change_summary[str((ds, cfg))]
        answer_change_rows_md.append(
            [
                ds,
                cfg,
                str(stats["agents_total"]),
                str(stats["agents_changed_at_least_once"]),
                str(stats["agents_never_changed"]),
                str(stats["change_events"]),
                fmt_pct(stats["change_event_rate"]) if stats["change_event_rate"] is not None else "",
            ]
        )
    if answer_change_rows_md:
        lines.append("## Answer Change Summary (Debate)\n")
        lines.append(
            _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Agents",
                    "Changed at least once",
                    "Never changed",
                    "Change events",
                    "Change-event rate",
                ],
                answer_change_rows_md,
            )
        )
        lines.append("")

    answer_change_by_round_rows: list[list[str]] = []
    for (ds, cfg), _stats in sorted(answer_change_summary_raw.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        stats = answer_change_summary[str((ds, cfg))]
        for row in stats["by_round"]:
            answer_change_by_round_rows.append(
                [
                    ds,
                    cfg,
                    str(row["round"]),
                    str(row["changed"]),
                    str(row["opportunities"]),
                    fmt_pct(row["change_rate"]) if row["change_rate"] is not None else "",
                ]
            )
    if answer_change_by_round_rows:
        lines.append("## Answer Change By Round (Debate)\n")
        lines.append(
            _md_table(
                ["Dataset", "Cfg", "Round", "Changed", "Opportunities", "Change rate"],
                answer_change_by_round_rows,
            )
        )
        lines.append("")

    first_change_rows: list[list[str]] = []
    for (ds, cfg), _stats in sorted(answer_change_summary_raw.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        stats = answer_change_summary[str((ds, cfg))]
        first_change_counts = stats["first_change_round_counts"]
        for round_no_key in sorted(first_change_counts, key=lambda key: int(key)):
            first_change_rows.append([ds, cfg, str(round_no_key), str(first_change_counts[round_no_key])])
    if first_change_rows:
        lines.append("## First Change Distribution (Debate)\n")
        lines.append(_md_table(["Dataset", "Cfg", "First change round", "Agents"], first_change_rows))
        lines.append("")

    persona_rows: list[list[str]] = []
    for (ds, cfg), _stats in sorted(persona_fidelity.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        stats = persona_fidelity_summary[str((ds, cfg))]
        persona_rows.append(
            [
                ds,
                cfg,
                str(stats["n_questions"]),
                f"{stats['round1_answer_entropy_mean']:.3f}" if stats["round1_answer_entropy_mean"] is not None else "",
                f"{stats['unique_round1_answers_mean']:.2f}" if stats["unique_round1_answers_mean"] is not None else "",
                fmt_pct(stats["persona_pair_disagreement_rate"]) if stats["persona_pair_disagreement_rate"] is not None else "",
                fmt_pct(stats["revision_rate_mean"]) if stats["revision_rate_mean"] is not None else "",
                fmt_pct(stats["convergence_rate_mean"]) if stats["convergence_rate_mean"] is not None else "",
                (
                    f"{stats['public_rationale_diversity_mean']:.3f}"
                    if stats["public_rationale_diversity_mean"] is not None
                    else ""
                ),
            ]
        )
    if persona_rows:
        lines.append("## Persona Fidelity Summary (Debate)\n")
        lines.append(
            _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Round-1 entropy",
                    "Unique round-1 answers",
                    "Pair disagreement",
                    "Revision rate",
                    "Convergence rate",
                    "Public rationale diversity",
                ],
                persona_rows,
            )
        )
        lines.append("")

    # Judge conditioning (debate only)
    judge_cond_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = int(c.get("n_questions", 0))
        if n <= 0:
            continue
        rep = int(c.get("final_repeat_winner_exists", 0))
        rep_abs = int(c.get("final_repeat_winner_absent", 0))
        judge_rep_k = int(c.get("judge_correct_when_repeat_winner_exists", 0))
        judge_abs_k = int(c.get("judge_correct_when_repeat_winner_absent", 0))
        judge_in_final_n = int(c.get("judge_total_when_in_final_round_answers", 0))
        judge_in_final_k = int(c.get("judge_correct_when_in_final_round_answers", 0))
        judge_not_in_final_n = int(c.get("judge_total_when_not_in_final_round_answers", 0))
        judge_not_in_final_k = int(c.get("judge_correct_when_not_in_final_round_answers", 0))

        judge_cond_rows.append(
            [
                ds,
                cfg,
                str(n),
                fmt_pct(rep / n),
                fmt_pct(judge_rep_k / rep if rep else 0.0),
                fmt_pct(judge_abs_k / rep_abs if rep_abs else 0.0),
                fmt_pct(judge_in_final_k / judge_in_final_n if judge_in_final_n else 0.0),
                fmt_pct(judge_not_in_final_k / judge_not_in_final_n if judge_not_in_final_n else 0.0),
            ]
        )
    if judge_cond_rows:
        lines.append("## Judge Conditional Breakdown (Debate)\n")
        lines.append(
            _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Repeat-winner exists",
                    "Judge acc (repeat-winner exists)",
                    "Judge acc (repeat-winner absent)",
                    "Judge acc (judge in final-round answers)",
                    "Judge acc (judge not in final-round answers)",
                ],
                judge_cond_rows,
            )
        )
        lines.append("")

    # Append key high-level findings from pooled stats
    pooled_md_rows: list[list[str]] = []
    for (ds, method, cfg), c in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k = int(c["correct"])
        n = int(c["total"])
        pooled_md_rows.append([ds, method, cfg, f"{k}/{n}", fmt_ci(k, n)])
    append_findings_md(
        "### Pooled Accuracy (All Included Runs)\n\n"
        + _md_table(
            ["Dataset", "Method", "Cfg", "k/n", "Acc (Wilson 95%)"],
            pooled_md_rows,
        )
        + "\n"
    )

    comparison_md_rows: list[list[str]] = []
    for (ds, method), c in sorted(comparison_pooled.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        k = int(c["correct"])
        n = int(c["total"])
        comparison_md_rows.append([ds, method, f"{k}/{n}", fmt_ci(k, n)])
    append_findings_md(
        "### Comparison Methods (Shared Round-1 vs Standalone Majority)\n\n"
        + _md_table(
            ["Dataset", "Method", "k/n", "Acc (Wilson 95%)"],
            comparison_md_rows,
        )
        + "\n"
    )

    # Majority failure patterns (per dataset)
    for ds, c in maj_patterns.items():
        total_runs = sum(v for k, v in c.items() if k.startswith("n_samples="))
        # Extract unique-answers distribution
        uniq = sorted(
            [(k, v) for k, v in c.items() if k.startswith("unique_answers=")],
            key=lambda kv: int(kv[0].split("=")[1]),
        )
        uniq_table = _md_table(
            ["Unique answers among samples", "Count"],
            [[k.split("=")[1], str(v)] for k, v in uniq],
        )

        none_ct = int(c.get("final_none=1", 0))
        strict = int(c.get("strict_majority_exists=1", 0))
        append_findings_md(
            f"### Majority Voting Disagreement Patterns ({ds})\n\n"
            + f"- Records analyzed: {total_runs}\n"
            + f"- `final_answer=None`: {none_ct}/{total_runs} ({fmt_pct(none_ct/total_runs if total_runs else 0.0)})\n"
            + f"- Strict-majority exists (parsed samples): {strict}/{total_runs} ({fmt_pct(strict/total_runs if total_runs else 0.0)})\n\n"
            + uniq_table
            + "\n"
        )

        # Oracle-style upper bounds using the existing parsed samples:
        # - any_sample_correct: at least one parsed sample equals GT
        # - first_non_none_correct: always pick first parseable sample
        any_correct = int(c.get("any_sample_correct", 0))
        first_non_none_correct = int(c.get("first_non_none_correct", 0))
        append_findings_md(
            f"### Majority: Oracle-Style Bounds ({ds})\n\n"
            + _md_table(
                ["Metric", "Value"],
                [
                    ["Records", str(total_runs)],
                    ["Any sample correct", f"{any_correct}/{total_runs} ({fmt_pct(any_correct/total_runs if total_runs else 0.0)})"],
                    ["First non-None correct", f"{first_non_none_correct}/{total_runs} ({fmt_pct(first_non_none_correct/total_runs if total_runs else 0.0)})"],
                ],
            )
            + "\n"
        )

    # Baseline formatting/truncation signals
    fmt_rows: list[list[str]] = []
    for ds, c in sorted(baseline_format.items(), key=lambda kv: kv[0]):
        total_comp = int(c.get("total_completions", 0))
        total_rec = int(c.get("total_records", 0))
        with_box = int(c.get("completions_with_boxed_or_choice", 0))
        chars_sum = int(c.get("completion_chars_sum", 0))
        avg_len = (chars_sum / total_comp) if total_comp else 0.0
        fmt_rows.append(
            [
                ds,
                str(total_rec),
                str(total_comp),
                fmt_pct(with_box / total_comp if total_comp else 0.0),
                f"{avg_len:.0f}",
            ]
        )
    if fmt_rows:
        append_findings_md(
            "### Baseline Formatting/Truncation Signals (Single+Majority)\n\n"
            + _md_table(
                ["Dataset", "Records", "Completions", "Has \\boxed/explicit choice", "Avg chars/completion"],
                fmt_rows,
            )
            + "\n"
        )

    # Debate dynamics + judge behavior
    dyn_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(debate_dyn.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_q = int(c.get("n_questions", 0))
        if n_q <= 0:
            continue
        ever = int(c.get("ever_correct", 0))
        lost = int(c.get("lost_correct", 0))
        final_has = int(c.get("final_round_has_correct", 0))
        jc = int(c.get("judge_correct", 0))
        mc = int(c.get("majority_correct", 0))
        changes = int(c.get("answer_changes", 0))
        steps = int(c.get("answer_steps", 0))
        change_rate = changes / steps if steps else 0.0
        dyn_rows.append(
            [
                ds,
                cfg,
                str(n_q),
                fmt_pct(ever / n_q),
                fmt_pct(final_has / n_q),
                fmt_pct(lost / n_q),
                fmt_pct(jc / n_q),
                fmt_pct(mc / n_q),
                f"{100.0*change_rate:.1f}%",
            ]
        )
    append_findings_md(
        "### Debate Dynamics (Belief Change + GT Retention)\n\n"
        + _md_table(
            [
                "Dataset",
                "Cfg",
                "Q",
                "Ever correct (any agent/round)",
                "Correct in final round",
                "Correct lost (ever but not final)",
                "Judge acc",
                "Final-round majority acc",
                "Per-step answer change rate",
            ],
            dyn_rows,
        )
        + "\n"
    )

    # Judge behavior summary: overrides + rescue/harm counts.
    judge_rows: list[list[str]] = []
    for (ds, cfg), ov in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        total = int(debate_dyn.get((ds, cfg), {}).get("n_questions", 0))
        if total <= 0:
            continue
        eq = int(ov.get("judge_equals_majority", 0))
        diff = int(ov.get("judge_differs_majority", 0))
        not_final = int(ov.get("judge_not_in_final_round_answers", 0))
        not_any = int(ov.get("judge_not_in_any_round_answers", 0))
        jm = judge_matrix.get((ds, cfg), {})
        rescue = int(jm.get("maj=0,judge=1", 0))
        harm = int(jm.get("maj=1,judge=0", 0))
        wrong_with_final_correct = int(debate_dyn[(ds, cfg)].get("judge_wrong_with_final_correct_present", 0))
        wrong_with_ever_correct = int(debate_dyn[(ds, cfg)].get("judge_wrong_with_ever_correct_present", 0))
        judge_rows.append(
            [
                ds,
                cfg,
                str(total),
                fmt_pct(diff / total),
                str(rescue),
                str(harm),
                fmt_pct(not_final / total),
                fmt_pct(not_any / total),
                fmt_pct(wrong_with_final_correct / total),
                fmt_pct(wrong_with_ever_correct / total),
            ]
        )
    if judge_rows:
        append_findings_md(
            "### Judge Behavior (Overrides, Rescues, Harms)\n\n"
            + _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Override rate (judge != maj)",
                    "Rescues (maj wrong -> judge right)",
                    "Harms (maj right -> judge wrong)",
                    "Judge not in final-round answers",
                    "Judge not in any-round answers",
                    "Judge wrong despite final correct present",
                    "Judge wrong despite ever-correct present",
                ],
                judge_rows,
            )
            + "\n"
        )

    # Judge conditional breakdown: how the judge behaves when agents do/don't produce a repeat-winner.
    judge_cond_find_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(judge_override.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = int(c.get("n_questions", 0))
        if n <= 0:
            continue
        rep = int(c.get("final_repeat_winner_exists", 0))
        rep_abs = int(c.get("final_repeat_winner_absent", 0))
        judge_rep_k = int(c.get("judge_correct_when_repeat_winner_exists", 0))
        judge_abs_k = int(c.get("judge_correct_when_repeat_winner_absent", 0))
        harm = int(c.get("judge_harm_over_repeat_winner_correct", 0))
        rescue_none = int(c.get("judge_rescue_from_repeat_winner_absent", 0))
        rescue_wrong = int(c.get("judge_rescue_from_repeat_winner_wrong", 0))
        judge_not_in_final = int(c.get("judge_not_in_final_round_answers", 0))
        judge_cond_find_rows.append(
            [
                ds,
                cfg,
                str(n),
                f"{rep}/{n}",
                f"{judge_rep_k}/{rep}" if rep else "0/0",
                f"{judge_abs_k}/{rep_abs}" if rep_abs else "0/0",
                str(harm),
                str(rescue_none),
                str(rescue_wrong),
                fmt_pct(judge_not_in_final / n),
            ]
        )
    if judge_cond_find_rows:
        append_findings_md(
            "### Judge Conditional Breakdown (Repeat-Winner vs No Repeat-Winner)\n\n"
            + _md_table(
                [
                    "Dataset",
                    "Cfg",
                    "Q",
                    "Repeat-winner exists",
                    "Judge correct (repeat-winner exists)",
                    "Judge correct (repeat-winner absent)",
                    "Harms over correct repeat-winner",
                    "Rescues from no repeat-winner",
                    "Rescues from wrong repeat-winner",
                    "Judge not in final-round answers",
                ],
                judge_cond_find_rows,
            )
            + "\n"
        )

    # Conformity proxy results
    conf_rows: list[list[str]] = []
    for (ds, cfg), c in sorted(change_toward_prev_other_majority.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        changed = int(c.get("changed", 0))
        to_other = int(c.get("changed_to_other_plurality", 0))
        away_other = int(c.get("changed_away_from_other_plurality", 0))
        conf_rows.append(
            [
                ds,
                cfg,
                str(changed),
                fmt_pct(to_other / changed if changed else 0.0),
                fmt_pct(away_other / changed if changed else 0.0),
            ]
        )
    append_findings_md(
        "### Conformity Proxy: Changes Toward Other-Agent Plurality\n\n"
        + _md_table(
            ["Dataset", "Cfg", "Total changes", "Changed to other plurality", "Changed away from other plurality"],
            conf_rows,
        )
        + "\n"
    )

    if _FINDINGS_MD_SECTIONS:
        lines.append("## Findings")
        lines.append("")
        lines.extend(_FINDINGS_MD_SECTIONS)
        lines.append("")

    with open(out_dir / "tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    # Narrative case-study output was intentionally removed.


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    analyze(Path(args.results_dir), out_dir)
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
