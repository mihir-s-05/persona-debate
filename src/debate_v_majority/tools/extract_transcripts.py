"""Extract readable persona debate transcripts from trace files and JSONL results.

Usage:
    python -m debate_v_majority.tools.extract_transcripts <trace_or_results_dir> [--output_dir DIR]

Examples:
    # Extract all persona debate traces from an experiment run
    python -m debate_v_majority.tools.extract_transcripts results/hle_experiment_20_text_only/hle_experiment_n20_seed55555_20260314_203317

    # Extract from a specific trace file
    python -m debate_v_majority.tools.extract_transcripts results/.../persona_debate/traces/008_hle_66fc5e61.txt

    # Specify output directory
    python -m debate_v_majority.tools.extract_transcripts results/... --output_dir my_transcripts
"""
from __future__ import annotations

import argparse
import json
import os
import re
import glob
import sys
from pathlib import Path


def break_sentences(text: str, max_width: int = 120) -> str:
    """Insert line breaks after sentences."""
    text = str(text).strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\*\(\\])', text)
    lines = []
    for part in parts:
        while len(part) > max_width:
            idx = part.rfind(' ', 0, max_width)
            if idx == -1:
                idx = max_width
            lines.append(part[:idx])
            part = part[idx:].lstrip()
        lines.append(part)
    return '\n'.join(lines)


def format_artifact_header(artifact: dict, label: str) -> str:
    """Build a readable header from a persona artifact JSON."""
    out = []
    cards = artifact.get('cards', [])
    descriptors = artifact.get('descriptors', [])
    axes = artifact.get('axes', [])
    judge_card = artifact.get('judge_card', {})

    out.append("=" * 80)
    out.append(f"PERSONA CARDS — {label}")
    out.append("=" * 80)
    out.append("")

    if axes:
        out.append("REASONING AXES:")
        out.append("")
        for i, ax in enumerate(axes):
            if isinstance(ax, dict):
                name = ax.get('name', f'Axis {i+1}')
                low = ax.get('low_label', '')
                high = ax.get('high_label', '')
                kind = "TASK-SPECIFIC" if ax.get('is_task_specific') else "FIXED"
                out.append(f"  {i+1}. {name} [{kind}]")
                out.append(f"     Low:  {low}")
                out.append(f"     High: {high}")
                out.append("")

    for i in range(max(len(cards), len(descriptors))):
        card = cards[i] if i < len(cards) else {}
        desc = descriptors[i] if i < len(descriptors) else {}

        name = desc.get('name', card.get('name', f'Persona {i+1}'))
        description = desc.get('description', card.get('description', ''))
        system_prompt = card.get('system_prompt', '')
        axis_values = desc.get('axis_values', {})

        out.append(f"AGENT {i+1}: {name}")
        out.append("~" * 60)
        out.append("")

        if description:
            out.append("Strategy:")
            out.append(break_sentences(description))
            out.append("")

        if axis_values:
            out.append("Axis Positions:")
            for ak, av in axis_values.items():
                val_str = f"{av:.3f}" if isinstance(av, float) else str(av)
                out.append(f"  {ak}: {val_str}")
            out.append("")

        if system_prompt:
            out.append("Full System Prompt:")
            out.append("-" * 40)
            out.append(break_sentences(system_prompt))
            out.append("-" * 40)
            out.append("")

    if isinstance(judge_card, dict) and judge_card:
        jname = judge_card.get('name', 'Domain Judge')
        out.append(f"JUDGE: {jname}")
        out.append("~" * 60)
        out.append("")
        jdesc = judge_card.get('description', '')
        if jdesc:
            out.append("Description:")
            out.append(break_sentences(jdesc))
            out.append("")
        jsp = judge_card.get('system_prompt', '')
        if jsp:
            out.append("Full System Prompt:")
            out.append("-" * 40)
            out.append(break_sentences(jsp))
            out.append("-" * 40)
            out.append("")

    return '\n'.join(out)


def extract_from_trace_file(trace_path: str, artifact_path: str | None, output_dir: str) -> str | None:
    """Extract a readable transcript from a trace .txt file + artifact .json."""
    if not os.path.exists(trace_path):
        return None

    with open(trace_path, encoding='utf-8') as f:
        trace = f.read()

    label = Path(trace_path).stem

    header = ""
    if artifact_path and os.path.exists(artifact_path):
        with open(artifact_path, encoding='utf-8') as f:
            artifact = json.load(f)
        header = format_artifact_header(artifact, label) + "\n\n"

    output = header + trace
    outfile = os.path.join(output_dir, f'{label}.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(output)
    return outfile


def _thought_from_round_output(output: dict) -> str:
    ts = str(output.get("thought_summary") or "").strip()
    if ts:
        return ts
    meta = output.get("call_metadata")
    if isinstance(meta, dict):
        ts2 = str(meta.get("thought_summary") or "").strip()
        if ts2:
            return ts2
    return ""


def extract_from_jsonl_row(row: dict, artifact: dict, label: str, output_dir: str) -> str | None:
    """Extract a readable transcript from a JSONL result row + artifact."""
    out = []

    out.append("=" * 80)
    out.append(f"PERSONA DEBATE TRANSCRIPT")
    out.append(f"Case: {label}")
    out.append("=" * 80)
    out.append("")

    out.append("QUESTION")
    out.append("-" * 60)
    out.append(break_sentences(str(row.get('question', ''))[:3000]))
    out.append("")
    out.append(f"Expected Answer: {row.get('answer', '')}")
    out.append(f"Domain: {row.get('family', '')}")
    out.append(f"Final Answer: {row.get('final_answer', '')}")
    out.append(f"Correct: {'YES' if row.get('final_correct') else 'NO'}")
    out.append("")

    if artifact:
        out.append(format_artifact_header(artifact, label))

    out.append("=" * 80)
    out.append("DEBATE ROUNDS")
    out.append("=" * 80)

    descriptors = artifact.get('descriptors', []) if artifact else []
    rpa = row.get('agent_round_parsed_answers', {})
    conv = row.get('convergence_per_round', [])
    changes = row.get('answer_changes_per_agent', [])

    if isinstance(rpa, dict):
        for r_key in sorted(rpa.keys(), key=lambda x: int(x)):
            agents = rpa[r_key]
            out.append("")
            out.append(f"ROUND {r_key}")
            out.append("-" * 40)
            if isinstance(agents, dict):
                for a_key in sorted(agents.keys(), key=lambda x: int(x)):
                    a_idx = int(a_key)
                    pname = descriptors[a_idx].get('name', f'Agent {a_idx+1}') if a_idx < len(descriptors) else f'Agent {a_idx+1}'
                    out.append(f"  {pname}: {agents[a_key]}")
            elif isinstance(agents, list):
                for a_idx, ans in enumerate(agents):
                    pname = descriptors[a_idx].get('name', f'Agent {a_idx+1}') if a_idx < len(descriptors) else f'Agent {a_idx+1}'
                    out.append(f"  {pname}: {ans}")

    out.append("")
    out.append("CONVERGENCE")
    out.append("-" * 40)
    for c in conv:
        if isinstance(c, dict):
            r = c.get('round', '?')
            vc = c.get('vote_counts', {})
            u = c.get('unanimous', False)
            out.append(f"  Round {r}: {vc}  {'(unanimous)' if u else ''}")

    if changes:
        out.append("")
        out.append("ANSWER CHANGES")
        out.append("-" * 40)
        for ch in changes:
            if isinstance(ch, dict):
                name = ch.get('agent_name', ch.get('persona_name', '?'))
                answers = ch.get('answers', [])
                changed_flags = ch.get('changed', [])
                if any(changed_flags):
                    out.append(f"  {name}: {' -> '.join(str(a) for a in answers)}")
                else:
                    out.append(f"  {name}: {answers[0] if answers else '?'} (never changed)")

    agent_round_outputs = row.get("agent_round_outputs") or []
    thought_lines: list[str] = []
    if isinstance(agent_round_outputs, list):
        for agent_idx, per_agent in enumerate(agent_round_outputs):
            if not isinstance(per_agent, list):
                continue
            pname = (
                descriptors[agent_idx].get('name', f'Agent {agent_idx + 1}')
                if agent_idx < len(descriptors)
                else f'Agent {agent_idx + 1}'
            )
            for round_idx, rout in enumerate(per_agent):
                if not isinstance(rout, dict):
                    continue
                ttxt = _thought_from_round_output(rout)
                if ttxt:
                    thought_lines.append(
                        f"Round {round_idx + 1} — {pname} — thought summary (model thinking):\n{ttxt}"
                    )
    if thought_lines:
        out.append("")
        out.append("=" * 80)
        out.append("THOUGHT SUMMARIES (DEBATERS)")
        out.append("=" * 80)
        out.append("")
        out.append("\n\n".join(thought_lines))

    jt_pre = row.get("judge_trace")
    if isinstance(jt_pre, dict):
        for meta_key, title in (
            ("judge_raw_call_metadata", "Judge (raw call)"),
            ("judge_retry_call_metadata", "Judge (retry call)"),
        ):
            meta = jt_pre.get(meta_key)
            if isinstance(meta, dict):
                jt = str(meta.get("thought_summary") or "").strip()
                if jt:
                    out.append("")
                    out.append("=" * 80)
                    out.append(f"THOUGHT SUMMARY — {title}")
                    out.append("=" * 80)
                    out.append("")
                    out.append(jt)

    out.append("")
    out.append("=" * 80)
    out.append("JUDGE OUTPUT")
    out.append("=" * 80)

    js = row.get('judge_summary', '')
    if js:
        out.append("")
        out.append("Judge Summary:")
        out.append(break_sentences(str(js)))

    jt = row.get('judge_trace', {})
    if isinstance(jt, dict):
        jr = jt.get('judge_raw_response', '')
        if jr:
            out.append("")
            out.append("Judge Full Response (visible output):")
            out.append(break_sentences(str(jr)))
        jretry = jt.get('judge_retry_raw_response', '')
        if jretry:
            out.append("")
            out.append("Judge Retry Response (visible output):")
            out.append(break_sentences(str(jretry)))

    out.append("")
    out.append(f"Final Answer: {row.get('final_answer', '')}")
    out.append(f"Correct: {'YES' if row.get('final_correct') else 'NO'}")

    outfile = os.path.join(output_dir, f'{label}.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    return outfile


def find_artifact_for_uid(uid_short: str, search_dirs: list[str]) -> str | None:
    """Find a persona artifact JSON matching a UID."""
    for d in search_dirs:
        matches = glob.glob(os.path.join(d, f'hle__{uid_short}--*.json'))
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract readable persona debate transcripts.")
    parser.add_argument("input", help="Path to experiment dir, trace file, or JSONL file")
    parser.add_argument("--output_dir", default="_transcripts", help="Output directory for transcript files")
    parser.add_argument("--uid", default=None, help="Extract only this UID (hex ID)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_path = args.input

    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        uid_match = re.search(r'hle_([a-f0-9]+)\.txt', input_path)
        uid_short = uid_match.group(1) if uid_match else None
        parent = str(Path(input_path).parent.parent.parent)
        artifact_dirs = [
            os.path.join(parent, 'persona_artifacts', 'hle'),
            os.path.join(parent, 'persona_debate', 'artifacts_llm', 'hle'),
        ]
        art_path = find_artifact_for_uid(uid_short, artifact_dirs) if uid_short else None
        outfile = extract_from_trace_file(input_path, art_path, args.output_dir)
        if outfile:
            print(f"Wrote {outfile}")

    elif os.path.isfile(input_path) and input_path.endswith('.jsonl'):
        artifact_dirs = []
        parent = str(Path(input_path).parent.parent)
        for subdir in ['persona_artifacts/hle', 'persona_debate/artifacts_llm/hle']:
            candidate = os.path.join(parent, subdir)
            if os.path.isdir(candidate):
                artifact_dirs.append(candidate)

        with open(input_path, encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                uid_short = row['item_uid'].split(':')[-1]
                if args.uid and uid_short != args.uid:
                    continue
                art_path = find_artifact_for_uid(uid_short, artifact_dirs)
                artifact = {}
                if art_path:
                    with open(art_path, encoding='utf-8') as af:
                        artifact = json.load(af)
                label = f"{uid_short}_{row.get('family', 'unknown')}"
                outfile = extract_from_jsonl_row(row, artifact, label, args.output_dir)
                if outfile:
                    print(f"Wrote {outfile}")

    elif os.path.isdir(input_path):
        trace_dir = os.path.join(input_path, 'persona_debate', 'traces')
        artifact_dirs = [
            os.path.join(input_path, 'persona_artifacts', 'hle'),
            os.path.join(input_path, 'persona_debate', 'artifacts_llm', 'hle'),
        ]

        if not any(os.path.isdir(d) for d in artifact_dirs):
            artifact_dirs = [os.path.join(input_path, 'persona_artifacts', 'hle')]

        if os.path.isdir(trace_dir):
            for trace_file in sorted(glob.glob(os.path.join(trace_dir, '*.txt'))):
                uid_match = re.search(r'hle_([a-f0-9]+)\.txt', trace_file)
                uid_short = uid_match.group(1) if uid_match else None
                if args.uid and uid_short != args.uid:
                    continue
                art_path = find_artifact_for_uid(uid_short, artifact_dirs) if uid_short else None
                outfile = extract_from_trace_file(trace_file, art_path, args.output_dir)
                if outfile:
                    size_kb = os.path.getsize(outfile) / 1024
                    print(f"Wrote {outfile} ({size_kb:.0f} KB)")
        else:
            def _is_results_jsonl(path: str) -> bool:
                base = os.path.basename(path)
                if 'stage_state' in base or base.startswith('personas_'):
                    return False
                return base.startswith('debate_') or base == 'results.jsonl'

            jsonl_files = glob.glob(os.path.join(input_path, 'persona_debate', '*.jsonl'))
            if not jsonl_files:
                jsonl_files = glob.glob(os.path.join(input_path, '*.jsonl'))
            jsonl_files = [f for f in jsonl_files if _is_results_jsonl(f)]
            for jsonl_file in jsonl_files:
                with open(jsonl_file, encoding='utf-8') as f:
                    for line in f:
                        row = json.loads(line)
                        uid_short = row['item_uid'].split(':')[-1]
                        if args.uid and uid_short != args.uid:
                            continue
                        art_path = find_artifact_for_uid(uid_short, artifact_dirs)
                        artifact = {}
                        if art_path:
                            with open(art_path, encoding='utf-8') as af:
                                artifact = json.load(af)
                        label = f"{uid_short}_{row.get('family', 'unknown')}"
                        outfile = extract_from_jsonl_row(row, artifact, label, args.output_dir)
                        if outfile:
                            print(f"Wrote {outfile}")
    else:
        print(f"Error: {input_path} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
