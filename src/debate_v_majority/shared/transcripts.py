from __future__ import annotations

from dataclasses import dataclass


def render_agent_transcript(agent_conv: list[dict[str, str]], include_system: bool = False) -> str:
    """Render a single agent's chat transcript into a string."""
    parts = []
    for msg in agent_conv:
        role = (msg.get("role") or "").strip()
        if role == "system" and not include_system:
            continue
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def assistant_message_indexes(agent_conv: list[dict[str, str]]) -> list[int]:
    """Find all assistant message indexes in a conversation."""
    idxs: list[int] = []
    for i, msg in enumerate(agent_conv):
        role = (msg.get("role") or "").strip()
        if role == "assistant":
            idxs.append(i)
    return idxs


def slice_agent_conv_round_range(
    agent_conv: list[dict[str, str]],
    *,
    start_round: int,
    end_round: int,
) -> list[dict[str, str]]:
    """
    Slice an agent conversation to include only assistant rounds [start_round, end_round] (1-indexed).
    Retains initial pre-round prefix (system/user question).
    """
    if not agent_conv:
        return []
    if start_round <= 0 or end_round <= 0:
        return agent_conv[:]
    if start_round > end_round:
        return agent_conv[:]

    assistant_idxs = assistant_message_indexes(agent_conv)
    if not assistant_idxs:
        return agent_conv[:]

    n_rounds = len(assistant_idxs)
    start_round = max(1, min(int(start_round), n_rounds))
    end_round = max(1, min(int(end_round), n_rounds))
    if start_round > end_round:
        return agent_conv[:]

    prefix_end = assistant_idxs[0]
    start_assistant = assistant_idxs[start_round - 1]
    end_assistant = assistant_idxs[end_round - 1]

    start = max(prefix_end, start_assistant - 1)
    end = end_assistant + 1
    return agent_conv[:prefix_end] + agent_conv[start:end]


def render_agent_assistant_rounds(
    agent_conv: list[dict[str, str]],
    *,
    start_round: int,
    end_round: int,
) -> str:
    """Render only assistant messages for rounds [start_round, end_round] (1-indexed)."""
    assistant_idxs = assistant_message_indexes(agent_conv)
    if not assistant_idxs:
        return ""
    n_rounds = len(assistant_idxs)
    start_round = max(1, min(int(start_round), n_rounds))
    end_round = max(1, min(int(end_round), n_rounds))
    if start_round > end_round:
        return ""

    parts: list[str] = []
    for round_num in range(start_round, end_round + 1):
        msg_idx = assistant_idxs[round_num - 1]
        content = agent_conv[msg_idx].get("content", "")
        parts.append(f"ROUND {round_num}:\n{content}")
    return "\n\n".join(parts)


def round_block_start(round_num: int, block_size: int) -> int:
    """Calculate the start round for a block-based judge window."""
    if block_size <= 0:
        return 1
    round_num = max(1, int(round_num))
    block_size = int(block_size)
    return ((round_num - 1) // block_size) * block_size + 1


@dataclass
class PrevJudgeInfo:
    """Information about a previous judge decision."""
    start_round: int
    end_round: int
    parsed_answer: str
    raw_output: str


def format_prev_judge_full(prev: PrevJudgeInfo) -> str:
    """Format previous judge output (full version)."""
    return (
        f"Rounds {prev.start_round}-{prev.end_round} judge answer: {prev.parsed_answer}\n"
        f"Judge transcript:\n{prev.raw_output}"
    )


def format_prev_judge_short(prev: PrevJudgeInfo) -> str:
    """Format previous judge output (short version)."""
    return f"Rounds {prev.start_round}-{prev.end_round} judge answer: {prev.parsed_answer}"


__all__ = [
    "PrevJudgeInfo",
    "assistant_message_indexes",
    "format_prev_judge_full",
    "format_prev_judge_short",
    "render_agent_assistant_rounds",
    "render_agent_transcript",
    "round_block_start",
    "slice_agent_conv_round_range",
]
