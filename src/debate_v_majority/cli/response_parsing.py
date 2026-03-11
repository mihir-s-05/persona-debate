from __future__ import annotations

import re
from typing import Any, Literal

from .. import DatasetName
from ..engines import InferenceResult
from ..shared import PromptTokenCounter
from ..shared import assistant_message_indexes
from .dataset_eval import _check_answer_correctness, _get_dataset_module, _parse_answer
from .engine_runtime import _inference_result_meta


def _parse_agent_round_answers(
    *,
    dataset: DatasetName,
    agent_contexts: list[list[dict[str, str]]],
    n_rounds: int,
    raw_task: dict[str, Any],
) -> list[list[str | None]]:
    out: list[list[str | None]] = []
    for ctx in agent_contexts:
        idxs = assistant_message_indexes(ctx)
        seq: list[str | None] = []
        for r in range(n_rounds):
            if r >= len(idxs):
                seq.append(None)
                continue
            text = ctx[idxs[r]].get("content", "")
            seq.append(_parse_answer(dataset, text, raw_task))
        out.append(seq)
    return out


def _word_bounded_text(text: str, *, max_tokens: int) -> tuple[str, bool]:
    words = re.findall(r"\S+", str(text or ""))
    if len(words) <= max_tokens:
        return " ".join(words), False
    return " ".join(words[:max_tokens]), True


def _derive_public_rationale(
    *,
    raw_response: str,
    parsed_answer: str | None,
    max_tokens: int,
) -> tuple[str | None, bool]:
    stripped = str(raw_response or "").strip()
    if not stripped:
        return None, False
    cleaned = re.sub(r"(?is)\\(?:boxed|fbox)\s*{.*?}", " ", stripped)
    if parsed_answer is not None:
        answer_pat = re.escape(str(parsed_answer))
        cleaned = re.sub(rf"(?i)\b(?:final\s+answer|answer|final\s+choice|choice)\b[^.\n]{{0,80}}{answer_pat}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:\n\t")
    if not cleaned:
        cleaned = stripped
    bounded, truncated = _word_bounded_text(cleaned, max_tokens=max_tokens)
    return bounded or None, truncated


def _extract_generic_confidence(text: str) -> dict[str, Any]:
    stripped = str(text or "")
    patterns = [
        re.compile(r"(?i)\bconfidence\b\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)\b"),
        re.compile(r"(?i)\bconfidence\b\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%"),
    ]
    for pattern in patterns:
        m = pattern.search(stripped)
        if not m:
            continue
        raw_text = m.group(0).strip()
        try:
            value = float(m.group(1))
        except (TypeError, ValueError):
            return {"raw_text": raw_text, "value": None, "parse_failed": True}
        if "%" in raw_text:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return {"raw_text": raw_text, "value": value, "parse_failed": False}
        return {"raw_text": raw_text, "value": None, "parse_failed": True}
    if re.search(r"(?i)\bconfidence\b", stripped):
        line = next((ln.strip() for ln in stripped.splitlines() if "confidence" in ln.lower()), "")
        return {"raw_text": line or "confidence_mentioned", "value": None, "parse_failed": True}
    return {"raw_text": None, "value": None, "parse_failed": False}


def _extract_output_details(
    *,
    dataset: DatasetName,
    raw_response: str,
    raw_task: dict[str, Any],
    gt_answer: Any | None = None,
    parse_mode: Literal["default", "strict", "recover"] = "default",
) -> dict[str, Any]:
    if dataset == "hle":
        mod = _get_dataset_module(dataset)
        extraction = mod.extract_response(raw_response, raw_task, parse_mode=parse_mode)
        if parse_mode == "default" and not extraction.get("parse_success"):
            recovered_extraction = mod.extract_response(raw_response, raw_task, parse_mode="recover")
            if recovered_extraction.get("parse_success"):
                diagnostics = dict(recovered_extraction.get("parse_diagnostics") or {})
                diagnostics["default_recovery_used"] = True
                recovered_extraction["parse_diagnostics"] = diagnostics
                extraction = recovered_extraction
        score = mod.score_answer(extraction.get("normalized_answer"), raw_task)
        return {
            "final_answer": extraction.get("normalized_answer"),
            "confidence": extraction.get("normalized_confidence"),
            "parse_success": bool(extraction.get("parse_success")),
            "confidence_raw_text": extraction.get("confidence_raw_text"),
            "confidence_parse_failed": bool(extraction.get("confidence_parse_failed")),
            "extractor_trace": extraction,
            "scoring_result": score,
        }

    parsed_answer = _parse_answer(dataset, raw_response, raw_task)
    confidence_info = _extract_generic_confidence(raw_response)
    scoring_result = None
    if gt_answer is not None:
        scoring_result = {
            "correct": _check_answer_correctness(dataset, parsed_answer, gt_answer, raw_task),
            "expected_answer": gt_answer,
            "predicted_answer": parsed_answer,
            "answer_format_type": raw_task.get("answer_format_type"),
            "match_type": "dataset_native",
            "scorer_provenance": f"{dataset}.native_scorer.v1",
        }
    return {
        "final_answer": parsed_answer,
        "confidence": confidence_info["value"],
        "parse_success": parsed_answer is not None,
        "confidence_raw_text": confidence_info["raw_text"],
        "confidence_parse_failed": bool(confidence_info["parse_failed"]),
        "extractor_trace": {
            "extractor_provenance": f"{dataset}.native_extractor.v1",
            "raw_output": str(raw_response),
            "candidate_answer": parsed_answer,
            "normalized_answer": parsed_answer,
            "normalized_confidence": confidence_info["value"],
            "confidence_raw_text": confidence_info["raw_text"],
            "confidence_parse_failed": bool(confidence_info["parse_failed"]),
            "parse_success": parsed_answer is not None,
            "parse_diagnostics": {"parse_mode": parse_mode, "dataset": dataset},
        },
        "scoring_result": scoring_result,
    }


def _build_round_output(
    *,
    dataset: DatasetName,
    raw_response: str,
    raw_task: dict[str, Any],
    gt_answer: Any,
    public_rationale_max_tokens: int,
    inference_result: InferenceResult | None = None,
    request_messages: list[dict[str, Any]] | None = None,
    request_engine: Any | None = None,
    prompt_token_counter: PromptTokenCounter | None = None,
) -> dict[str, Any]:
    details = _extract_output_details(
        dataset=dataset,
        raw_response=raw_response,
        raw_task=raw_task,
        gt_answer=gt_answer,
        parse_mode="default",
    )
    parsed_answer = details["final_answer"]
    public_rationale, truncated = _derive_public_rationale(
        raw_response=raw_response,
        parsed_answer=parsed_answer,
        max_tokens=public_rationale_max_tokens,
    )
    return {
        "private_raw_response": str(raw_response),
        "visible_output": str(raw_response),
        "thought_summary": None if inference_result is None else inference_result.thought_summary,
        "thought_summary_available": False if inference_result is None else bool(inference_result.thought_summary_available),
        "public_rationale": public_rationale,
        "public_rationale_truncated": bool(truncated),
        "final_answer": parsed_answer,
        "confidence": details["confidence"],
        "confidence_raw_text": details["confidence_raw_text"],
        "confidence_parse_failed": details["confidence_parse_failed"],
        "parse_success": bool(details["parse_success"]),
        "extractor_trace": details["extractor_trace"],
        "scoring_result": details["scoring_result"],
        "call_metadata": _inference_result_meta(
            inference_result,
            request_messages=request_messages,
            engine=request_engine,
            prompt_token_counter=prompt_token_counter,
        ),
    }


def _format_debate_share_entry(round_output: dict[str, Any]) -> str:
    """Extract visible output (thinking already stripped) for sharing with other debaters."""
    return str(round_output.get("visible_output") or round_output.get("private_raw_response") or "")


def _judge_trace_mode_enabled(judge_trace_mode: str | None) -> str:
    mode = str(judge_trace_mode or "assistant_transcript").strip().lower()
    if mode in {"visible_plus_thought_summary", "assistant_transcript"}:
        return mode
    raise ValueError(f"Unsupported judge_trace_mode: {judge_trace_mode}")


def _render_agent_round_outputs_for_judge(
    *,
    agent_round_outputs: list[list[dict[str, Any]]],
    start_round: int,
    end_round: int,
    judge_trace_mode: str,
) -> list[str]:
    mode = _judge_trace_mode_enabled(judge_trace_mode)
    transcripts: list[str] = []
    for agent_idx, outputs in enumerate(agent_round_outputs):
        parts = [f"AGENT {agent_idx + 1}"]
        upper_bound = min(end_round, len(outputs))
        for round_num in range(max(1, start_round), upper_bound + 1):
            output = outputs[round_num - 1]
            visible_output = str(
                output.get("visible_output") or output.get("private_raw_response") or ""
            ).strip()
            thought_summary = str(output.get("thought_summary") or "").strip()
            round_lines = [f"ROUND {round_num}"]
            if visible_output:
                round_lines.append("Visible output:")
                round_lines.append(visible_output)
            if mode == "visible_plus_thought_summary":
                if thought_summary:
                    round_lines.append("Thought summary:")
                    round_lines.append(thought_summary)
                else:
                    round_lines.append("Thought summary: <unavailable>")
            parts.append("\n".join(round_lines))
        transcripts.append("\n\n".join(parts))
    return transcripts


__all__ = [
    "_build_round_output",
    "_derive_public_rationale",
    "_extract_generic_confidence",
    "_extract_output_details",
    "_format_debate_share_entry",
    "_judge_trace_mode_enabled",
    "_parse_agent_round_answers",
    "_render_agent_round_outputs_for_judge",
    "_word_bounded_text",
]
