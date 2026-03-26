#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _select_row(rows: list[dict[str, Any]], *, item: str | None, row_index: int | None) -> dict[str, Any]:
    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"row index {row_index} out of range for {len(rows)} rows")
        return rows[row_index]
    if item is None:
        if not rows:
            raise ValueError("no rows available")
        return rows[0]
    wanted = str(item)
    for row in rows:
        candidates = (
            row.get("item_uid"),
            row.get("item_display_id"),
            row.get("orig_id"),
            row.get("subset_id"),
        )
        if any(str(candidate) == wanted for candidate in candidates if candidate is not None):
            return row
    raise KeyError(f"no row matched item identifier {wanted!r}")


def _indent(text: str, *, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in str(text).splitlines())


def _display_block(row: dict[str, Any]) -> dict[str, Any]:
    display = row.get("display")
    return dict(display) if isinstance(display, dict) else {}


def _trace_block(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    return dict(trace) if isinstance(trace, dict) else {}


def _persona_summaries(row: dict[str, Any]) -> list[dict[str, Any]]:
    display = _display_block(row)
    persona_meta = row.get("persona_meta")
    if isinstance(display.get("persona_summaries"), list):
        return list(display["persona_summaries"])
    if isinstance(persona_meta, dict) and isinstance(persona_meta.get("persona_summaries"), list):
        return list(persona_meta["persona_summaries"])
    return list(row.get("persona_summaries") or row.get("sample_persona_summaries") or [])


def _judge_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    display = _display_block(row)
    judge_meta = row.get("judge_meta")
    summary = display.get("judge_summary")
    if isinstance(summary, dict):
        return summary
    if isinstance(judge_meta, dict) and isinstance(judge_meta.get("judge_summary"), dict):
        return dict(judge_meta["judge_summary"])
    summary = row.get("judge_summary")
    return dict(summary) if isinstance(summary, dict) else None


def _agent_round_outputs(row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    trace = _trace_block(row)
    outputs = trace.get("per_round_agent_outputs")
    if isinstance(outputs, list):
        return list(outputs)
    return list(row.get("agent_round_outputs") or [])


def _judge_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = _trace_block(row)
    judge_trace = trace.get("judge_trace")
    if isinstance(judge_trace, dict):
        return dict(judge_trace)
    judge_trace = row.get("judge_trace")
    return dict(judge_trace) if isinstance(judge_trace, dict) else {}


def _persona_label(row: dict[str, Any], agent_idx: int) -> str:
    summaries = _persona_summaries(row)
    if agent_idx < len(summaries):
        summary = summaries[agent_idx]
        if summary is None:
            return f"Agent {agent_idx + 1} (Plain)"
        title = summary.get("title")
        persona_id = summary.get("persona_id")
        if title and persona_id:
            return f"Agent {agent_idx + 1}: {title} ({persona_id})"
        if title:
            return f"Agent {agent_idx + 1}: {title}"
    return f"Agent {agent_idx + 1}"


def _render_persona_section(row: dict[str, Any]) -> list[str]:
    summaries = _persona_summaries(row)
    if not summaries:
        return []
    lines = ["", "Personas"]
    for idx, summary in enumerate(summaries, start=1):
        if summary is None:
            lines.append(f"- Agent {idx}: (Plain – no persona)")
            continue
        title = summary.get("title") or "Untitled persona"
        persona_id = summary.get("persona_id") or f"persona-{idx}"
        short_rule = summary.get("short_rule")
        strategy = summary.get("core_reasoning_strategy")
        lines.append(f"- Agent {idx}: {title} [{persona_id}]")
        if short_rule:
            lines.append(f"  Rule: {short_rule}")
        if strategy:
            lines.append(f"  Strategy: {strategy}")
    return lines


def _render_judge_section(row: dict[str, Any]) -> list[str]:
    summary = _judge_summary(row)
    if not summary:
        return []
    lines = ["", "Judge"]
    judge_id = summary.get("judge_id") or "judge"
    title = summary.get("title") or summary.get("judge_family") or "Untitled judge"
    lines.append(f"- {title} [{judge_id}]")
    if summary.get("decision_style"):
        lines.append(f"  Decision style: {summary['decision_style']}")
    if summary.get("selection_principle"):
        lines.append(f"  Selection principle: {summary['selection_principle']}")
    if summary.get("judge_family"):
        lines.append(f"  Judge family: {summary['judge_family']}")
    if summary.get("domain_scope"):
        lines.append(f"  Domain scope: {summary['domain_scope']}")
    if summary.get("independent_resolve_policy"):
        lines.append(f"  Resolve policy: {summary['independent_resolve_policy']}")
    if summary.get("answer_format_policy"):
        lines.append(f"  Answer format policy: {summary['answer_format_policy']}")
    if summary.get("confidence_policy") is not None:
        lines.append(f"  Confidence policy: {summary['confidence_policy']}")
    return lines


def _render_hle_metadata(row: dict[str, Any]) -> list[str]:
    if str(row.get("dataset") or "").strip().lower() != "hle":
        return []
    lines = ["", "HLE Metadata"]
    if row.get("source_variant") is not None:
        lines.append(f"- Variant: {row.get('source_variant')}")
    if row.get("source_subset_label") is not None:
        lines.append(f"- Verified subset: {row.get('source_subset_label')}")
    if row.get("canonical_item_id") is not None:
        lines.append(f"- Canonical item ID: {row.get('canonical_item_id')}")
    if row.get("answer_format_type") is not None:
        lines.append(f"- Answer format: {row.get('answer_format_type')}")
    if row.get("domain_family") is not None:
        lines.append(f"- Domain family: {row.get('domain_family')}")
    if row.get("source_dataset_revision") is not None:
        lines.append(f"- Dataset revision: {row.get('source_dataset_revision')}")
    if row.get("source_paper_version") is not None:
        lines.append(f"- Source paper: {row.get('source_paper_version')}")
    return lines if len(lines) > 1 else []


def _render_scoring_result(scoring: dict[str, Any] | None, *, prefix: str = "  ") -> list[str]:
    if not scoring:
        return []
    lines: list[str] = []
    if scoring.get("match_type") is not None:
        lines.append(f"{prefix}Scoring match type: {scoring.get('match_type')}")
    if scoring.get("expected_answer") is not None:
        lines.append(f"{prefix}Expected answer: {scoring.get('expected_answer')}")
    if scoring.get("predicted_answer") is not None:
        lines.append(f"{prefix}Predicted answer: {scoring.get('predicted_answer')}")
    if scoring.get("correct") is not None:
        lines.append(f"{prefix}Correct: {scoring.get('correct')}")
    accepted_answers = scoring.get("accepted_answers")
    if accepted_answers:
        lines.append(f"{prefix}Accepted answers: {accepted_answers}")
    return lines


def _render_extraction_block(
    *,
    parsed_answer: Any,
    parse_success: Any,
    confidence: Any,
    confidence_raw_text: Any,
    confidence_parse_failed: Any,
    extractor: dict[str, Any] | None,
    scoring: dict[str, Any] | None,
    prefix: str = "  ",
) -> list[str]:
    lines = [f"{prefix}Parsed answer: {parsed_answer if parsed_answer is not None else '<unparsed>'}"]
    lines.append(f"{prefix}Parse success: {bool(parse_success)}")
    if confidence is not None:
        lines.append(f"{prefix}Confidence: {float(confidence):.2f}")
    elif confidence_raw_text:
        lines.append(f"{prefix}Confidence raw: {confidence_raw_text}")
    if confidence_parse_failed:
        lines.append(f"{prefix}Confidence parse failed: True")
    extractor = extractor or {}
    if extractor.get("extractor_provenance"):
        lines.append(f"{prefix}Extractor: {extractor.get('extractor_provenance')}")
    if extractor.get("candidate_answer") is not None:
        lines.append(f"{prefix}Candidate answer: {extractor.get('candidate_answer')}")
    if extractor.get("answer_format_type") is not None:
        lines.append(f"{prefix}Answer format: {extractor.get('answer_format_type')}")
    lines.extend(_render_scoring_result(scoring, prefix=prefix))
    return lines


def _render_response_block(text: str, *, prefix: str = "  ", heading: str = "Response") -> list[str]:
    rendered = str(text or "").strip()
    if not rendered:
        return []
    return [f"{prefix}{heading}:", _indent(rendered, prefix=prefix + "  ")]


def _thought_summary_text(round_output: dict[str, Any]) -> str:
    """Prefer top-level thought_summary; fall back to Gemini metadata on call_metadata."""
    ts = str(round_output.get("thought_summary") or "").strip()
    if ts:
        return ts
    meta = round_output.get("call_metadata")
    if isinstance(meta, dict):
        ts2 = str(meta.get("thought_summary") or "").strip()
        if ts2:
            return ts2
    return ""


def _judge_thought_sections(judge_trace: dict[str, Any]) -> list[tuple[str, str]]:
    """Return judge thought summaries without conflating raw and retry calls."""
    sections: list[tuple[str, str]] = []
    for key, title in (
        ("judge_raw_call_metadata", "Judge thought summary (raw call)"),
        ("judge_retry_call_metadata", "Judge thought summary (retry call)"),
    ):
        meta = judge_trace.get(key)
        if not isinstance(meta, dict):
            continue
        ts = str(meta.get("thought_summary") or "").strip()
        if ts:
            sections.append((title, ts))
    return sections


def _render_judge_extraction_section(
    title: str,
    *,
    extractor: dict[str, Any] | None,
    scoring: dict[str, Any] | None,
) -> list[str]:
    if not extractor:
        return []
    lines = ["", title]
    lines.extend(
        _render_extraction_block(
            parsed_answer=extractor.get("normalized_answer"),
            parse_success=extractor.get("parse_success"),
            confidence=extractor.get("normalized_confidence"),
            confidence_raw_text=extractor.get("confidence_raw_text"),
            confidence_parse_failed=extractor.get("confidence_parse_failed"),
            extractor=extractor,
            scoring=scoring,
            prefix="- ",
        )
    )
    return lines


def _render_debate_trace(row: dict[str, Any]) -> list[str]:
    lines = ["", "Debate Trace"]
    display = _display_block(row)
    persona_meta = row.get("persona_meta") if isinstance(row.get("persona_meta"), dict) else {}
    judge_meta = row.get("judge_meta") if isinstance(row.get("judge_meta"), dict) else {}
    agent_round_outputs = _agent_round_outputs(row)
    n_rounds = max((len(agent_outputs) for agent_outputs in agent_round_outputs), default=0)
    for round_idx in range(n_rounds):
        lines.append("")
        lines.append(f"Round {round_idx + 1}")
        for agent_idx, agent_outputs in enumerate(agent_round_outputs):
            label = _persona_label(row, agent_idx)
            round_output = agent_outputs[round_idx] if round_idx < len(agent_outputs) else {}
            lines.append(f"- {label}")
            lines.extend(
                _render_extraction_block(
                    parsed_answer=round_output.get("final_answer"),
                    parse_success=round_output.get("parse_success"),
                    confidence=round_output.get("confidence"),
                    confidence_raw_text=round_output.get("confidence_raw_text"),
                    confidence_parse_failed=round_output.get("confidence_parse_failed"),
                    extractor=round_output.get("extractor_trace"),
                    scoring=round_output.get("scoring_result"),
                    prefix="  ",
                )
            )
            rationale = str(round_output.get("public_rationale") or "").strip()
            if rationale:
                lines.append(f"  Public rationale: {rationale}")
            thought_summary = _thought_summary_text(round_output)
            if thought_summary:
                lines.append("  Thought summary (model thinking):")
                lines.append(_indent(thought_summary, prefix="    "))
            response_heading = "Response (visible output)" if thought_summary else "Response"
            lines.extend(
                _render_response_block(
                    str(round_output.get("visible_output") or round_output.get("private_raw_response") or ""),
                    prefix="  ",
                    heading=response_heading,
                )
            )

    lines.append("")
    lines.append("Outcomes")
    lines.append(
        f"- Round-1 majority: {display.get('round1_majority_answer', row.get('round1_majority_answer'))} "
        f"(correct={row.get('round1_majority_correct')})"
    )
    lines.append(
        f"- Final-round majority: {display.get('final_round_majority_answer', row.get('final_round_majority_answer'))} "
        f"(correct={row.get('final_round_majority_correct')})"
    )
    lines.append(
        f"- Judge final: {display.get('judge_final_answer', row.get('final_judge_answer'))} "
        f"(correct={row.get('final_judge_correct')})"
    )

    execution_meta_lines: list[str] = []
    if persona_meta.get("public_rationale_max_tokens") is not None:
        execution_meta_lines.append(
            f"- Public rationale budget: {persona_meta.get('public_rationale_max_tokens')}"
        )
    if judge_meta.get("judge_persona_mode") is not None:
        execution_meta_lines.append(f"- Judge persona mode: {judge_meta.get('judge_persona_mode')}")
    if judge_meta.get("judge_trace_mode") is not None:
        execution_meta_lines.append(f"- Judge trace mode: {judge_meta.get('judge_trace_mode')}")
    family_assignment = judge_meta.get("judge_family_assignment")
    if isinstance(family_assignment, dict) and family_assignment.get("judge_family") is not None:
        execution_meta_lines.append(
            f"- Judge family assignment: {family_assignment.get('judge_family')} "
            f"(source={family_assignment.get('source')})"
        )
    if execution_meta_lines:
        lines.append("")
        lines.append("Execution Metadata")
        lines.extend(execution_meta_lines)

    convergence = row.get("convergence_per_round") or []
    if convergence:
        lines.append("")
        lines.append("Convergence")
        for item in convergence:
            round_no = item.get("round")
            distinct = item.get("distinct_answers")
            unanimous = item.get("unanimous")
            votes = item.get("vote_counts")
            lines.append(f"- Round {round_no}: distinct={distinct}, unanimous={unanimous}, votes={votes}")

    changes = row.get("answer_changes_per_agent") or []
    if changes:
        lines.append("")
        lines.append("Answer Changes")
        for item in changes:
            label = _persona_label(row, int(item.get("agent_index", 0)))
            lines.append(
                f"- {label}: first_change_round={item.get('first_change_round')}, "
                f"changed={item.get('changed_from_prior_round')}, answers={item.get('answers_by_round')}"
            )

    judge_trace = _judge_trace(row)
    judge_context = judge_trace.get("judge_context") or []
    if judge_context:
        lines.append("")
        lines.append("Judge Context")
        if judge_trace.get("judge_context_is_full_transcript") is not None:
            lines.append(
                f"- Full transcript: {judge_trace.get('judge_context_is_full_transcript')}"
            )
        start_round = judge_trace.get("judge_context_start_round")
        end_round = judge_trace.get("judge_context_end_round")
        if start_round is not None or end_round is not None:
            lines.append(f"- Rounds covered: {start_round} -> {end_round}")
        previous_summary = judge_trace.get("judge_previous_summary")
        if previous_summary:
            lines.append("- Previous judge summary included:")
            lines.append(_indent(str(previous_summary)))
        for msg in judge_context:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip() or "unknown"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"- {role}:")
            lines.append(_indent(content))
    for title, judge_thought in _judge_thought_sections(judge_trace):
        lines.append("")
        lines.append(title)
        lines.append(_indent(judge_thought))
    judge_raw = str(judge_trace.get("judge_raw_response") or "").strip()
    if judge_raw:
        lines.append("")
        lines.append("Judge Raw Response (visible output)")
        lines.append(_indent(judge_raw))
    judge_retry_raw = str(judge_trace.get("judge_retry_raw_response") or "").strip()
    if judge_retry_raw:
        lines.append("")
        lines.append("Judge Retry Raw Response (visible output)")
        lines.append(_indent(judge_retry_raw))
    lines.extend(
        _render_judge_extraction_section(
            "Judge Extraction",
            extractor=judge_trace.get("judge_extractor_trace") or {},
            scoring=judge_trace.get("judge_scoring_result"),
        )
    )
    lines.extend(
        _render_judge_extraction_section(
            "Judge Retry Extraction",
            extractor=judge_trace.get("judge_retry_extractor_trace") or {},
            scoring=judge_trace.get("judge_retry_scoring_result"),
        )
    )
    return lines


def _render_majority_trace(row: dict[str, Any]) -> list[str]:
    lines = ["", "Majority Trace"]
    completions = row.get("sample_completions") or []
    parsed = row.get("sample_parsed_answers") or []
    extractions = row.get("sample_extractions") or []
    scoring_results = row.get("sample_scoring_results") or []
    persona_summaries = row.get("sample_persona_summaries") or row.get("persona_summaries") or []
    for idx, completion in enumerate(completions):
        label = _persona_label({"persona_summaries": persona_summaries}, idx)
        parsed_answer = parsed[idx] if idx < len(parsed) else None
        lines.append(f"- {label}")
        extraction = extractions[idx] if idx < len(extractions) else {}
        score = scoring_results[idx] if idx < len(scoring_results) else None
        lines.extend(
            _render_extraction_block(
                parsed_answer=parsed_answer,
                parse_success=extraction.get("parse_success"),
                confidence=extraction.get("normalized_confidence"),
                confidence_raw_text=extraction.get("confidence_raw_text"),
                confidence_parse_failed=extraction.get("confidence_parse_failed"),
                extractor=extraction,
                scoring=score,
                prefix="  ",
            )
        )
        lines.extend(_render_response_block(str(completion or ""), prefix="  ", heading="Completion"))
    lines.append("")
    lines.append("Outcome")
    lines.append(f"- Vote counts: {row.get('vote_counts')}")
    lines.append(f"- Strict majority: {row.get('strict_majority_answer')}")
    lines.append(f"- Plurality: {row.get('plurality_answer')}")
    lines.append(f"- Final majority: {row.get('final_majority_answer')} (correct={row.get('final_majority_correct')})")
    return lines


def _render_single_trace(row: dict[str, Any]) -> list[str]:
    lines = ["", "Single Trace"]
    completions = row.get("sample_completions") or []
    parsed = row.get("sample_parsed_answers") or []
    extractions = row.get("sample_extractions") or []
    scoring_results = row.get("sample_scoring_results") or []
    call_metadata = row.get("sample_call_metadata") or []

    for idx, completion in enumerate(completions):
        lines.append(f"- Sample {idx + 1}")
        metadata = call_metadata[idx] if idx < len(call_metadata) else {}
        thought_summary = str(metadata.get("thought_summary") or "").strip() if isinstance(metadata, dict) else ""
        if thought_summary:
            lines.append("  Thought summary:")
            lines.append(_indent(thought_summary, prefix="    "))
        parsed_answer = parsed[idx] if idx < len(parsed) else None
        extraction = extractions[idx] if idx < len(extractions) else {}
        score = scoring_results[idx] if idx < len(scoring_results) else None
        lines.extend(
            _render_extraction_block(
                parsed_answer=parsed_answer,
                parse_success=extraction.get("parse_success"),
                confidence=extraction.get("normalized_confidence"),
                confidence_raw_text=extraction.get("confidence_raw_text"),
                confidence_parse_failed=extraction.get("confidence_parse_failed"),
                extractor=extraction,
                scoring=score,
                prefix="  ",
            )
        )
        lines.extend(_render_response_block(str(completion or ""), prefix="  ", heading="Completion"))

    lines.append("")
    lines.append("Outcome")
    lines.append(f"- Final answer: {row.get('final_answer')}")
    lines.append(f"- Correct: {row.get('final_correct')}")
    return lines


def render_row_text(row: dict[str, Any]) -> str:
    lines = [
        f"Mode: {row.get('mode')}",
        f"Dataset: {row.get('dataset')}",
    ]
    if row.get("item_uid") is not None:
        lines.append(f"Item UID: {row.get('item_uid')}")
    if row.get("item_display_id") is not None:
        lines.append(f"Display ID: {row.get('item_display_id')}")
    if row.get("orig_id") is not None:
        lines.append(f"Original ID: {row.get('orig_id')}")
    if row.get("question"):
        lines.extend(["", "Question", str(row["question"])])
    if row.get("answer") is not None:
        lines.extend(["", f"Ground Truth: {row.get('answer')}"])
    lines.extend(_render_hle_metadata(row))
    lines.extend(_render_persona_section(row))
    lines.extend(_render_judge_section(row))
    mode = str(row.get("mode") or "").strip().lower()
    if mode == "debate":
        lines.extend(_render_debate_trace(row))
    elif mode == "single":
        lines.extend(_render_single_trace(row))
    elif mode == "majority":
        lines.extend(_render_majority_trace(row))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a JSONL run file.")
    parser.add_argument("--item", help="Item identifier to match against item_uid, item_display_id, orig_id, or subset_id.")
    parser.add_argument("--row-index", type=int, help="0-based row index within the JSONL file.")
    parser.add_argument("--out", help="Optional text output path. Defaults to stdout.")
    args = parser.parse_args(argv)

    rows = list(read_jsonl(Path(args.input)))
    row = _select_row(rows, item=args.item, row_index=args.row_index)
    text = render_row_text(row)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
