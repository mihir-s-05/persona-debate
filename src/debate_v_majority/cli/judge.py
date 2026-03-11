from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .. import DatasetName
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from ..shared import (
    PrevJudgeInfo,
    PromptTokenCounter,
    format_prev_judge_full,
    format_prev_judge_short,
    normalize_freeform_string,
    normalize_numeric_string,
    parse_math,
    round_block_start,
)
from .judge_common import (
    _count_prompt_tokens,
    _parse_csv_ints,
    _parse_judge_rounds,
)


JUDGE_SYSTEM_PROMPT = (
    "You are a transcript-grounded judge agent. Evaluate the answers and arguments already present in the "
    "debate transcript, select the answer best supported by that transcript, and avoid solving the problem "
    "from scratch unless a minimal consistency check is needed. Provide the final answer exactly in the "
    "format requested by the question.\n\n"
)

JUDGE_RETRY_NUDGE = (
    "Your previous output was unparsable.\n"
    "Reply again and output ONLY the final answer in the required format (e.g., \\boxed{...}).\n"
    "Do not include any other text."
)


@dataclass(frozen=True)
class JudgeParseResult:
    """Judge parse outcome with provenance."""

    answer: str | None
    mode: str
    source: str
    strict_success: bool


def _get_dataset_adapter(dataset: DatasetName):
    return resolve_dataset_adapter(dataset)


def _parse_answer(dataset: DatasetName, text: str, raw_task: dict[str, Any]) -> str | None:
    return _get_dataset_adapter(dataset).parse_answer(text, raw_task)


def _get_judge_prompt(dataset: DatasetName) -> dict[str, str]:
    return _get_dataset_adapter(dataset).judge_prompt


def _strict_parse_answer(dataset: DatasetName, text: str, raw_task: dict[str, Any]) -> str | None:
    t = str(text or "")
    if not t.strip():
        return None

    if dataset == "hle":
        adapter = _get_dataset_adapter(dataset)
        return adapter.strict_parse_answer(t, raw_task)

    if dataset == "aime25":
        boxed = parse_math(t)
        if boxed is None:
            return None
        norm = normalize_numeric_string(boxed)
        if norm is None:
            return None
        try:
            v = int(norm)
        except ValueError:
            return None
        if 0 <= v <= 999:
            return normalize_numeric_string(str(v))
        return None

    if dataset == "gpqa":
        boxed = parse_math(t)
        if boxed is None:
            return None
        parsed = _parse_answer(dataset, f"\\boxed{{{boxed}}}", raw_task)
        if parsed is None:
            return None
        _question, gt, _prepared = _get_dataset_adapter(dataset).parse_question_answer(raw_task)
        parsed_norm = normalize_freeform_string(parsed)
        if parsed_norm is None:
            return None
        if str(gt).strip().upper() in {"A", "B", "C", "D"}:
            return parsed_norm.upper() if parsed_norm in {"a", "b", "c", "d"} else None
        return parsed_norm

    return None


def _recover_parse_answer(dataset: DatasetName, text: str, raw_task: dict[str, Any]) -> str | None:
    """
    Conservative recovery parser used before rerunning judge generation.
    Unlike dataset-level parsing, this avoids brittle "last token" fallbacks.
    """
    t = str(text or "")
    if not t.strip():
        return None

    if dataset == "hle":
        adapter = _get_dataset_adapter(dataset)
        return adapter.recover_parse_answer(t, raw_task)

    if dataset == "aime25":
        t = re.sub(r"(\d),(\d)", r"\1\2", t)
        tail = t[-4000:] if len(t) > 4000 else t
        int_token = r"(?<![\d.])-?\d{1,3}(?!\.\d)\b"
        m = None
        for m in re.finditer(rf"(?i)\b(?:final\s+answer|answer|final)\b[^0-9]{{0,40}}({int_token})", tail):
            pass
        if not m:
            return None
        try:
            v = int(m.group(1))
        except ValueError:
            return None
        if 0 <= v <= 999:
            return normalize_numeric_string(str(v))
        return None

    if dataset == "gpqa":
        tail = t[-2400:] if len(t) > 2400 else t
        m = None
        cue_pat = re.compile(
            r"(?i)\b(?:final\s+answer|answer|final\s+choice|choice)\b[^A-D]{0,40}\(?\s*([ABCD])\s*\)?\b"
        )
        for m in cue_pat.finditer(t):
            pass
        if m:
            return m.group(1).upper()

        m = None
        opt_pat = re.compile(r"(?i)\boption\s*([ABCD])\b")
        for m in opt_pat.finditer(tail):
            pass
        if m:
            return m.group(1).upper()

        m = None
        final_line_pat = re.compile(r"(?i)\bfinal\b[^A-D]{0,40}\(?\s*([ABCD])\s*\)?")
        for m in final_line_pat.finditer(t):
            pass
        if m:
            return m.group(1).upper()

        return None

    return None


def _parse_judge_output(
    *,
    dataset: DatasetName,
    text: Any,
    raw_task: dict[str, Any],
    source_prefix: str,
    strict_enabled: bool = True,
    recovery_enabled: bool = True,
) -> JudgeParseResult:
    """
    Parse judge output with strict-first, recovery-second policy.
    """
    if text is None:
        return JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)

    raw_text = str(text)
    variants = [("raw", raw_text)]

    if strict_enabled:
        for suffix, candidate in variants:
            parsed = _strict_parse_answer(dataset, candidate, raw_task)
            if parsed is not None:
                src = f"{source_prefix}_strict" if suffix == "raw" else f"{source_prefix}_strict_variant"
                return JudgeParseResult(answer=parsed, mode="strict", source=src, strict_success=True)

    if recovery_enabled:
        for suffix, candidate in variants:
            parsed = _recover_parse_answer(dataset, candidate, raw_task)
            if parsed is not None:
                src = f"{source_prefix}_recovery" if suffix == "raw" else f"{source_prefix}_recovery_variant"
                return JudgeParseResult(answer=parsed, mode="recover", source=src, strict_success=False)

    return JudgeParseResult(answer=None, mode="none", source="none", strict_success=False)


def _build_judge_context(
    *,
    dataset: DatasetName,
    question: str,
    raw_task: dict[str, Any],
    responses: list[str],
    previous_judge: str | None = None,
    judge_system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build judge context messages."""
    parts = [f"Question: {question}"]
    if previous_judge:
        parts.append(f"Previous judge output (from earlier rounds):\n{previous_judge}")

    for agent_idx, response in enumerate(responses, start=1):
        parts.append(f"=== Agent {agent_idx} debate transcript ===\n{response}")

    judge_prompt = _get_judge_prompt(dataset)
    user_prompt = "\n\n".join(parts) + judge_prompt["user_prompt_suffix"]
    user_content: Any = user_prompt
    if dataset == "hle":
        from ..datasets import hle as hle_dataset

        user_content = hle_dataset.build_prompt_content(
            user_prompt,
            raw_task,
            attach_images=True,
            attachment_notice="Relevant task images are attached with this judge prompt.",
        )

    return [
        {"role": "system", "content": judge_system_prompt or JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _select_adaptive_judge_window(
    *,
    dataset: DatasetName,
    question: str,
    raw_task: dict[str, Any],
    agent_round_outputs: list[list[dict[str, Any]]],
    end_round: int,
    prev: PrevJudgeInfo | None,
    engine: Any,
    counter: PromptTokenCounter | None,
    context_len_tokens: int,
    max_new_tokens: int,
    judge_system_prompt: str | None = None,
    judge_trace_mode: str = "visible_plus_thought_summary",
) -> tuple[int, str | None]:
    from .response_parsing import _render_agent_round_outputs_for_judge

    if end_round <= 0 or context_len_tokens <= 0:
        return 1, None

    budget = max(1, int(context_len_tokens) - max(0, int(max_new_tokens)) - 256)

    prev_options: list[str | None] = [None]
    if prev is not None:
        prev_options = [format_prev_judge_full(prev), format_prev_judge_short(prev), None]

    for prev_text in prev_options:
        start_round = 1
        while start_round <= end_round:
            responses = _render_agent_round_outputs_for_judge(
                agent_round_outputs=agent_round_outputs,
                start_round=start_round,
                end_round=end_round,
                judge_trace_mode=judge_trace_mode,
            )
            msgs = _build_judge_context(
                dataset=dataset,
                question=question,
                raw_task=raw_task,
                responses=responses,
                previous_judge=prev_text,
                judge_system_prompt=judge_system_prompt,
            )
            n_tokens = _count_prompt_tokens(engine=engine, counter=counter, messages=msgs)
            if n_tokens is None:
                return start_round, prev_text
            if n_tokens <= budget:
                return start_round, prev_text
            start_round += 1

    raise RuntimeError(f"Judge prompt does not fit within context_len_tokens={context_len_tokens}")


def _build_judge_round_context(
    *,
    dataset: DatasetName,
    question: str,
    raw_task: dict[str, Any],
    agent_round_outputs: list[list[dict[str, Any]]],
    block_end: int,
    judge_block_size: int | None,
    prev_judge: PrevJudgeInfo | None,
    judge_system_prompt: str | None,
    judge_trace_mode: str,
    judge_engine: Any,
    judge_token_counter: PromptTokenCounter | None,
    context_len_tokens: int,
    judge_max_new_tokens: int,
) -> tuple[int, str | None, list[dict[str, Any]]]:
    from .response_parsing import _render_agent_round_outputs_for_judge

    if judge_block_size is not None and int(judge_block_size) > 0:
        block_start = round_block_start(block_end, int(judge_block_size))
        prev_text = format_prev_judge_full(prev_judge) if prev_judge else None
    else:
        block_start, prev_text = _select_adaptive_judge_window(
            dataset=dataset,
            question=question,
            raw_task=raw_task,
            agent_round_outputs=agent_round_outputs,
            end_round=block_end,
            prev=prev_judge,
            judge_system_prompt=judge_system_prompt,
            judge_trace_mode=judge_trace_mode,
            engine=judge_engine,
            counter=judge_token_counter,
            context_len_tokens=context_len_tokens,
            max_new_tokens=judge_max_new_tokens,
        )

    transcripts = _render_agent_round_outputs_for_judge(
        agent_round_outputs=agent_round_outputs,
        start_round=block_start,
        end_round=block_end,
        judge_trace_mode=judge_trace_mode,
    )
    return (
        int(block_start),
        prev_text,
        _build_judge_context(
            dataset=dataset,
            question=question,
            raw_task=raw_task,
            responses=transcripts,
            previous_judge=prev_text,
            judge_system_prompt=judge_system_prompt,
        ),
    )


__all__ = [
    "JUDGE_RETRY_NUDGE",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeParseResult",
    "_build_judge_context",
    "_build_judge_round_context",
    "_parse_judge_output",
    "_recover_parse_answer",
    "_select_adaptive_judge_window",
    "_strict_parse_answer",
]
