"""Staged execution state persistence.

Provides a JSONL-based state file where each pipeline stage appends one
self-contained entry.  The file acts as append-only history: every prior
snapshot is preserved, and the latest line is used by default when resuming.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ENTRY_SCHEMA_VERSION = "stage_entry.v1"
STAGE_TRACE_SCHEMA_VERSION = "stage_trace.v2"


def path_setting(path: Path | None) -> str | None:
    return None if path is None else str(path)


def subset_item_resume_signature(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        data = asdict(item)
    elif isinstance(item, dict):
        data = dict(item)
    elif hasattr(item, "__dict__"):
        data = dict(vars(item))
    else:
        data = dict(item)
    return {
        "subset_id": data.get("subset_id"),
        "orig_id": data.get("orig_id"),
        "item_uid": data.get("item_uid"),
        "dataset_revision": data.get("dataset_revision"),
        "item_display_id": data.get("item_display_id"),
        "raw_task": data.get("raw_task"),
        "family": data.get("family"),
    }


@dataclass
class StageEntry:
    schema_version: str
    stage_type: str
    completed_stage: str
    dataset: str
    items: list[dict[str, Any]]
    persona_data: dict[str, Any] = field(default_factory=dict)
    debate_data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageEntry":
        return cls(
            schema_version=str(data.get("schema_version", STAGE_ENTRY_SCHEMA_VERSION)),
            stage_type=str(data["stage_type"]),
            completed_stage=str(data["completed_stage"]),
            dataset=str(data["dataset"]),
            items=list(data.get("items", [])),
            persona_data=dict(data.get("persona_data") or {}),
            debate_data=dict(data.get("debate_data") or {}),
            meta=dict(data.get("meta") or {}),
            created_at=str(data.get("created_at", "")),
        )


def make_stage_entry(
    *,
    stage_type: str,
    completed_stage: str,
    dataset: str,
    items: list[dict[str, Any]],
    persona_data: dict[str, Any] | None = None,
    debate_data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> StageEntry:
    return StageEntry(
        schema_version=STAGE_ENTRY_SCHEMA_VERSION,
        stage_type=stage_type,
        completed_stage=completed_stage,
        dataset=dataset,
        items=items,
        persona_data=persona_data or {},
        debate_data=debate_data or {},
        meta=meta or {},
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def append_stage_entry(path: Path, entry: StageEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = False
    if path.exists():
        size = path.stat().st_size
        if size > 0:
            with path.open("rb") as existing:
                existing.seek(-1, 2)
                needs_leading_newline = existing.read(1) not in (b"\n", b"\r")
    with path.open("a", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    _write_stage_trace(path=path, entry=entry)


def _sanitize_filename_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or ""))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def _entry_timestamp_fragment(entry: StageEntry) -> str:
    try:
        parsed = datetime.fromisoformat(str(entry.created_at))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _trace_file_path(path: Path, entry: StageEntry) -> Path:
    stage_type = _sanitize_filename_part(entry.stage_type)
    completed_stage = _sanitize_filename_part(entry.completed_stage)
    timestamp = _entry_timestamp_fragment(entry)
    return path.parent / (
        f"{path.stem}.trace.{stage_type}.{completed_stage}.v2.{timestamp}.md"
    )


def _trace_file_glob(path: Path, entry: StageEntry) -> str:
    stage_type = _sanitize_filename_part(entry.stage_type)
    completed_stage = _sanitize_filename_part(entry.completed_stage)
    return f"{path.stem}.trace.{stage_type}.{completed_stage}.v2.*.md"


def _format_multiline_block(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return "_none_"
    return f"```\n{stripped}\n```"


def _short_text(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _render_persona_stage(entry: StageEntry) -> list[str]:
    lines = [
        f"- Persona stage: `{entry.completed_stage}`",
        f"- Items: `{len(entry.items)}`",
    ]
    for item in entry.items:
        item_uid = str(item.get("item_uid") or "unknown_item")
        item_data = dict(entry.persona_data.get(item_uid) or {})
        lines.append("")
        lines.append(f"## Item `{item_uid}`")
        if entry.completed_stage == "axes":
            axis_selection = dict(item_data.get("axis_selection") or {})
            axes = list(axis_selection.get("axes") or [])
            sampled_points = list(item_data.get("sampled_points") or [])
            lines.append(f"- Axis mode: `{axis_selection.get('mode')}`")
            lines.append(f"- Axis count: `{len(axes)}`")
            for idx, axis in enumerate(axes, start=1):
                lines.append(
                    f"  {idx}. `{axis.get('axis_id')}`: {axis.get('name')} "
                    f"({axis.get('kind')})"
                )
            if sampled_points:
                lines.append("")
                lines.append("### Sampled Persona Points")
                for idx, point in enumerate(sampled_points, start=1):
                    values = ", ".join(f"{k}={v}" for k, v in dict(point).items())
                    lines.append(f"- Persona {idx}: {values}")
        elif entry.completed_stage == "descriptors":
            descriptors = list(item_data.get("descriptors") or [])
            lines.append(f"- Descriptor count: `{len(descriptors)}`")
            for desc in descriptors:
                lines.append("")
                lines.append(f"### {desc.get('persona_id')}: {desc.get('name')}")
                lines.append(f"- Short rule: {_short_text(desc.get('short_rule'))}")
                lines.append(f"- Reasoning summary: {_short_text(desc.get('reasoning_summary'), limit=500)}")
        elif entry.completed_stage == "cards":
            cards = list(item_data.get("cards") or [])
            lines.append(f"- Card count: `{len(cards)}`")
            for card in cards:
                lines.append("")
                lines.append(f"### {card.get('persona_id')}: {card.get('title')}")
                lines.append(f"- Strategy: {_short_text(card.get('core_reasoning_strategy'), limit=500)}")
                priorities = list(card.get("priorities") or [])
                if priorities:
                    lines.append("- Priorities:")
                    for priority in priorities:
                        lines.append(f"  - {priority}")
                distrusts = list(card.get("distrusts") or [])
                if distrusts:
                    lines.append("- Distrusts:")
                    for distrust in distrusts:
                        lines.append(f"  - {distrust}")
                lines.append("- System prompt:")
                runtime_prompts = card.get("runtime_prompts") or {}
                active_prompt = (
                    runtime_prompts.get("initial_system_prompt")
                    if isinstance(runtime_prompts, dict)
                    else None
                ) or card.get("initial_system_prompt") or card.get("system_prompt") or ""
                lines.append(
                    _format_multiline_block(str(active_prompt))
                )
        elif entry.completed_stage == "judge_card":
            judge_card = dict(item_data.get("judge_card") or {})
            lines.append(f"- Judge family: `{judge_card.get('judge_family')}`")
            lines.append(f"- Domain scope: {_short_text(judge_card.get('domain_scope'), limit=500)}")
            priorities = list(judge_card.get("evaluation_priorities") or [])
            if priorities:
                lines.append("- Evaluation priorities:")
                for priority in priorities:
                    lines.append(f"  - {priority}")
            lines.append(f"- Tie-break policy: {_short_text(judge_card.get('tie_break_policy'), limit=500)}")
            lines.append("- System prompt:")
            lines.append(_format_multiline_block(str(judge_card.get("system_prompt") or "")))
        else:
            lines.append(_format_multiline_block(json.dumps(item_data, ensure_ascii=False, indent=2)))
    return lines


def _render_debate_stage(entry: StageEntry) -> list[str]:
    debate_data = dict(entry.debate_data or {})
    n_agents = debate_data.get("n_agents")
    n_rounds = debate_data.get("n_rounds")
    judge_rounds = debate_data.get("judge_rounds") or []
    lines = [
        f"- Debate stage: `{entry.completed_stage}`",
        f"- Agents: `{n_agents}`",
        f"- Debate rounds configured: `{n_rounds}`",
        f"- Judge rounds: `{', '.join(str(x) for x in judge_rounds) if judge_rounds else 'none'}`",
    ]

    persona_titles: list[str] = []
    persona_artifacts = list(debate_data.get("persona_artifacts") or [])
    if persona_artifacts:
        first_artifact = persona_artifacts[0] or {}
        slot_layout = first_artifact.get("slot_layout")
        cards = list(first_artifact.get("cards") or [])
        if slot_layout:
            card_iter = iter(cards)
            for slot_idx, slot_type in enumerate(slot_layout):
                if slot_type == "plain":
                    persona_titles.append(f"agent_{slot_idx + 1} (Plain)")
                else:
                    card = next(card_iter, None)
                    persona_titles.append(
                        str(card.get("title") or card.get("persona_id") or f"agent_{slot_idx + 1}")
                        if card else f"agent_{slot_idx + 1}"
                    )
        else:
            persona_titles = [
                str(card.get("title") or card.get("persona_id") or f"agent_{idx}")
                for idx, card in enumerate(cards)
            ]

    stage_name = str(entry.completed_stage)
    if stage_name.endswith("_judge"):
        round_num = stage_name.removeprefix("round_").removesuffix("_judge")
        results_by_round = dict(debate_data.get("results_by_round") or {})
        stage_rows = list(results_by_round.get(round_num) or [])
        lines.append("")
        lines.append(f"## Judge Results For Round `{round_num}`")
        if not stage_rows:
            lines.append("_No judge rows saved._")
        for idx, row in enumerate(stage_rows, start=1):
            lines.append(f"### Item {idx}: `{row.get('item_uid')}`")
            lines.append(f"- Judge answer: `{row.get('judge_final_answer') or row.get('judge_answer')}`")
            lines.append(f"- Judge correct: `{row.get('judge_final_correct')}`")
            lines.append(f"- Final answer: `{row.get('final_answer')}`")
            judge_trace = row.get("judge_trace")
            if judge_trace:
                lines.append("- Judge trace:")
                lines.append(_format_multiline_block(str(judge_trace)))
    else:
        try:
            round_idx = int(stage_name.removeprefix("round_"))
        except ValueError:
            round_idx = -1
        item_outputs_all = list(debate_data.get("agent_round_outputs_by_q") or [])
        lines.append("")
        lines.append(f"## Debater Outputs For Round `{round_idx}`")
        for item_idx, item_outputs in enumerate(item_outputs_all):
            item_uid = str(entry.items[item_idx].get("item_uid") if item_idx < len(entry.items) else f"item_{item_idx}")
            lines.append(f"### Item `{item_uid}`")
            for agent_idx, agent_outputs in enumerate(item_outputs):
                if round_idx < 0 or round_idx >= len(agent_outputs):
                    continue
                output = dict(agent_outputs[round_idx] or {})
                title = (
                    persona_titles[agent_idx]
                    if agent_idx < len(persona_titles)
                    else f"agent_{agent_idx + 1}"
                )
                lines.append("")
                lines.append(f"#### Agent {agent_idx + 1}: {title}")
                lines.append(f"- Final answer: `{output.get('final_answer')}`")
                scoring = dict(output.get("scoring_result") or {})
                if scoring:
                    lines.append(f"- Correct: `{scoring.get('correct')}`")
                visible_output = str(output.get("visible_output") or output.get("private_raw_response") or "").strip()
                if visible_output:
                    lines.append("- Visible output:")
                    lines.append(_format_multiline_block(visible_output))
                public_rationale = str(output.get("public_rationale") or "").strip()
                if public_rationale:
                    lines.append("- Public rationale:")
                    lines.append(_format_multiline_block(public_rationale))
                thought_summary = str(output.get("thought_summary") or "").strip()
                if thought_summary:
                    lines.append("- Thought summary:")
                    lines.append(_format_multiline_block(thought_summary))
    return lines


def _format_stage_trace(entry: StageEntry) -> str:
    lines = [
        f"# Stage Trace `{STAGE_TRACE_SCHEMA_VERSION}`",
        "",
        f"- Stage type: `{entry.stage_type}`",
        f"- Completed stage: `{entry.completed_stage}`",
        f"- Dataset: `{entry.dataset}`",
        f"- Created at: `{entry.created_at}`",
    ]
    if entry.stage_type == "persona":
        lines.extend(["", * _render_persona_stage(entry)])
    elif entry.stage_type == "debate":
        lines.extend(["", * _render_debate_stage(entry)])
    else:
        lines.extend(
            [
                "",
                "## Raw Entry",
                _format_multiline_block(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2)),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _write_stage_trace(*, path: Path, entry: StageEntry) -> Path:
    trace_path = _trace_file_path(path, entry)
    trace_path.write_text(_format_stage_trace(entry), encoding="utf-8")
    for old_trace in path.parent.glob(_trace_file_glob(path, entry)):
        if old_trace == trace_path:
            continue
        try:
            old_trace.unlink()
        except FileNotFoundError:
            continue
    return trace_path


def _iter_stage_entries(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield StageEntry.from_dict(json.loads(stripped))


def load_latest_stage_entry(path: Path) -> StageEntry:
    latest: StageEntry | None = None
    for entry in _iter_stage_entries(path):
        latest = entry
    if latest is None:
        raise ValueError(f"Stage state file is empty: {path}")
    return latest


def load_all_stage_entries(path: Path) -> list[StageEntry]:
    return list(_iter_stage_entries(path))


def load_latest_stage_entry_of_type(path: Path, stage_type: str) -> StageEntry | None:
    match: StageEntry | None = None
    for entry in _iter_stage_entries(path):
        if entry.stage_type == stage_type:
            match = entry
    return match


def load_stage_entry_by_name(path: Path, stage_name: str) -> StageEntry:
    """Find the last entry with the given ``completed_stage`` value."""
    match: StageEntry | None = None
    for entry in _iter_stage_entries(path):
        if entry.completed_stage == stage_name:
            match = entry
    if match is None:
        raise ValueError(f"No stage entry named {stage_name!r} in {path}")
    return match


__all__ = [
    "STAGE_ENTRY_SCHEMA_VERSION",
    "StageEntry",
    "append_stage_entry",
    "load_all_stage_entries",
    "load_latest_stage_entry",
    "load_latest_stage_entry_of_type",
    "load_stage_entry_by_name",
    "make_stage_entry",
    "path_setting",
    "subset_item_resume_signature",
]
