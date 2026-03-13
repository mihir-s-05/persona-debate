from __future__ import annotations

import json
import re
from typing import Any


ARTIFACT_VERSION = "phase0.v1"
AXIS_PROMPT_VERSION = "phase0.axes.v3"
DESCRIPTOR_PROMPT_VERSION = "phase0.descriptors.v6"
CARD_PROMPT_VERSION = "phase0.cards.v6"
JUDGE_PROMPT_VERSION = "phase0.judge.v4"
JUDGE_BANK_PROMPT_VERSION = "phase0.judge_bank.v1"

STAGE1_GUIDANCE = (
    "Generate operationally distinct reasoning personas optimized for support coverage rather than "
    "average-case plausibility. Deliberately cover axis extremes and informative intermediate niches. "
    "Vary decomposition style, verification policy, skepticism, revision behavior, search order, and "
    "how aggressively the persona prunes search. Each persona should imply a measurably different "
    "round-1 solution trajectory and a different revision policy under disagreement. Avoid collapsing "
    "multiple personas into the same safe middle strategy. Do not include likely answers, option hints, "
    "problem-specific facts, theorem giveaways, or hidden solution trajectories. "
    "Descriptors must stay problem-agnostic and execution-policy-focused: describe search order, branch "
    "management, verification timing, evidence preference, revision triggers, and confidence discipline."
)

STAGE2_GUIDANCE = (
    "Expand each descriptor into a compact but high-leverage system prompt that changes how the model "
    "reasons, not who it pretends to be. Encode concrete operating rules: what evidence to privilege, "
    "when to branch or prune, when to verify, when to distrust an intermediate, and when to revise "
    "under disagreement. The result should be an operational reasoning policy, not role-play, biography, "
    "tone, or generic advice."
)

JUDGE_GUIDANCE = (
    "The judge sees the full debate transcript and should act as a transcript-grounded selector. "
    "Select the answer with the strongest support in the transcript rather than running a fresh "
    "unconstrained solve. Use only limited independent checking to break close cases or catch obvious "
    "transcript mistakes. The judge must handle any number of debaters or debate turns, not only a "
    "two-participant exchange."
)


def _json_schema_block(schema_obj: dict[str, Any]) -> str:
    return json.dumps(schema_obj, indent=2, ensure_ascii=False)


def _extract_fenced_json(text: str) -> str | None:
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, handling fenced blocks and bare braces."""
    s = str(text or "").strip()
    for candidate in (s, _extract_fenced_json(s), _extract_braced_json(s)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError(f"Could not parse JSON object from model output: {s[:400]}")


def _user_content_with_media(
    text: str,
    *,
    question_media: list[dict[str, Any]] | None = None,
) -> str | list[dict[str, Any]]:
    if not question_media:
        return text
    return [{"type": "text", "text": text}, *question_media]


def build_task_axis_messages(
    *,
    dataset: str,
    benchmark_family: str,
    question: str,
    count: int,
    question_media: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema = {
        "axes": [
            {
                "axis_id": "short_snake_case_id",
                "name": "Human readable axis name",
                "low_desc": "Reasoning behavior at the low end",
                "high_desc": "Reasoning behavior at the high end",
                "notes": "Optional constraint or caution",
            }
        ]
    }
    image_note = "Relevant task images are attached with this prompt.\n\n" if question_media else ""
    user = (
        f"Dataset: {dataset}\n"
        f"Benchmark family: {benchmark_family}\n"
        f"Target axis count: {count}\n\n"
        f"Question:\n{question}\n\n"
        f"{image_note}"
        "Propose reasoning-relevant axes for diverse debaters.\n"
        "Good axes should create observable differences in search order, decomposition style, verification "
        "timing, pruning aggressiveness, evidence standards, or revision triggers.\n"
        "Prefer axes whose extremes would plausibly send agents down different first-pass solution paths on "
        "this question or task family.\n"
        "Favor support coverage over average-case density: include axes that help span rare but plausible "
        "reasoning policies, not just the most common safe personas.\n"
        "Do not propose biography, tone, verbosity, generic competence, answer hints, theorem giveaways, "
        "trap spoilers, or any axis that implicitly nudges toward a specific final answer.\n"
        "Return JSON only matching this schema:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate reasoning-diversity axes for multi-agent debate. "
                "Optimize for support coverage across plausible reasoning policies. "
                "Axes must change how the agent reasons, not what answer it is nudged toward."
            ),
        },
        {"role": "user", "content": _user_content_with_media(user, question_media=question_media)},
    ]


def build_stage1_messages(
    *,
    dataset: str,
    benchmark_family: str,
    axes: list[dict[str, Any]],
    sampled_points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    schema = {
        "descriptors": [
            {
                "persona_id": "persona_1",
                "name": "Short descriptive name",
                "axis_interpretation": {"axis_id": "how this persona sits on that axis"},
                "short_rule": "One-sentence operational rule",
                "reasoning_summary": "Brief explanation of how this persona reasons",
            }
        ]
    }
    user = (
        f"Dataset: {dataset}\n"
        f"Benchmark family: {benchmark_family}\n\n"
        "You are generating reasoning personas for this benchmark family.\n"
        "Descriptors must be reusable across any item in this family—do not target a specific problem.\n\n"
        "Axes:\n"
        f"{json.dumps(axes, indent=2, ensure_ascii=False)}\n\n"
        "Persona slots and sampled positions:\n"
        f"{json.dumps(sampled_points, indent=2, ensure_ascii=False)}\n\n"
        "Treat the sampled positions as hard anchors for population design.\n"
        "Plan the population jointly so that the full set covers the axis space rather than producing "
        "near-duplicate safe personas.\n"
        "Ensure the set contains informative contrasts: some extremes, some cross-axis mixtures, and at "
        "most one near-center compromise persona if forced by the sampled points.\n"
        "If two personas would likely produce similar reasoning traces, sharpen them until their search "
        "policy, verification habit, likely failure mode, and recovery path are clearly different.\n"
        "Translate axis positions into general execution tendencies, not domain-method templates. If an axis "
        "name sounds method-specific, abstract it to the underlying policy rather than repeating the method "
        "name.\n"
        "Allowed descriptor content: how the persona generates candidates, what evidence it privileges, when "
        "it prunes, when it verifies, when it revises, and what kinds of intermediate states it distrusts.\n"
        "Disallowed descriptor content: named math techniques, object-type commitments, calculus-specific "
        "framing, recurrence hunting, neighborhood search, graph-shape language, coordinate transforms, "
        "small-case enumeration, or any other domain workflow template.\n"
        "Good descriptor example: 'commits to a candidate path early, then stress-tests it with focused local "
        "checks before widening the search.'\n"
        "Good descriptor example: 'holds multiple hypotheses in parallel and only prunes after a concrete "
        "contradiction appears.'\n"
        "Bad descriptor example: 'start from small cases and derive a recurrence.'\n"
        "Bad descriptor example: 'analyze the derivative roots to count local minima.'\n"
        f"{STAGE1_GUIDANCE}\n"
        "Generate the full persona population jointly in one response.\n"
        "IMPORTANT: Return a single valid JSON object and nothing else—no prose, no markdown fencing, "
        "no explanation. Schema:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are generating a coordinated set of debater personas for a benchmark family. "
                "Plan the whole population jointly. They must be distinct, compact, operationally useful, "
                "and reusable across different items in the family. "
                "You will NOT be shown the specific problem—generate family-level reasoning policies only."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_stage2_messages(
    *,
    question: str,
    descriptor: dict[str, Any],
    question_media: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema = {
        "persona_id": descriptor.get("persona_id", "persona_1"),
        "title": "Compact title",
        "core_reasoning_strategy": "One compact paragraph",
        "priorities": ["priority 1", "priority 2"],
        "distrusts": ["thing to distrust 1", "thing to distrust 2"],
        "decomposition_style": "How this persona breaks work apart",
        "revision_policy": "When this persona should revise",
        "confidence_policy": "How uncertainty is handled",
        "failure_mode_to_avoid": "Main failure mode",
        "system_prompt": "Compact operational system prompt",
    }
    user = (
        "You are generating an operational card for a benchmark-family persona.\n"
        "The card must stay reusable across different items and must not target any specific problem.\n\n"
        "Descriptor:\n"
        f"{json.dumps(descriptor, indent=2, ensure_ascii=False)}\n\n"
        "Write in terms of operating rules, not biography.\n"
        "The card should specify how this persona chooses candidate paths, checks intermediate steps, "
        "handles disagreement, and decides whether to revise.\n"
        "Differences should survive contact with the task and be visible in execution, not just sound "
        "different on paper.\n"
        "Stay faithful to the descriptor, but compress it into a narrower policy vocabulary. Convert any "
        "domain-flavored descriptor wording into the underlying execution rule.\n"
        "Allowed card content: candidate-generation order, branch expansion vs pruning, verification cadence, "
        "evidence thresholds, contradiction handling, revision triggers, and confidence gating.\n"
        "Disallowed card content: named method families, object-type commitments, assumed formula forms, "
        "small-case enumeration, recurrence search, neighborhood search, geometric pairings, coordinate "
        "transforms, derivative-specific machinery, graph-shape stories, continuous optimization templates, "
        "or any other hidden task structure.\n"
        "If a descriptor phrase is method-shaped, rewrite it to the more general policy it implies.\n"
        "Example rewrite: 'uses a continuous model' -> 'forms a coarse global approximation before verifying "
        "locally'.\n"
        "Example rewrite: 'starts from small cases' -> 'builds confidence from simple concrete probes before "
        "generalizing'.\n"
        "The card must describe portable reasoning behavior only. It must not speculate about what specific "
        "object types, formulas, or latent patterns will appear in the problem.\n"
        "Good card move: 'propose a global invariant early, then reject branches that violate it and only "
        "switch strategies after an explicit contradiction.'\n"
        "Bad card move: 'start with n=1,2,3', 'test nearby integers', 'look for geometric means', or any "
        "other task template.\n"
        "Do not mention item-specific equations, constants, answer choices, variable names, images, "
        "or any solve plan for a particular question.\n"
        f"{STAGE2_GUIDANCE}\n"
        "Return JSON only matching:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are expanding a persona descriptor into a compact operational card. "
                "The card must reliably induce a distinct reasoning policy while staying concise, "
                "and remain reusable across different items in the benchmark family. "
                "You will NOT be shown the specific problem."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    *,
    dataset: str,
    benchmark_family: str,
    question: str,
    mode: str,
    question_media: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema = {
        "judge_id": "judge_identifier",
        "judge_family": "family or mode label",
        "domain_scope": "short scope string",
        "evaluation_priorities": ["priority 1", "priority 2", "priority 3"],
        "tie_break_policy": "How ties should be broken",
        "independent_resolve_policy": "limited_check_only",
        "answer_format_policy": "Strict answer format rule",
        "confidence_policy": "optional or null",
        "system_prompt": "Judge prompt text",
    }
    image_note = "Relevant task images are attached with this prompt.\n\n" if question_media else ""
    user = (
        f"Dataset: {dataset}\n"
        f"Benchmark family: {benchmark_family}\n"
        f"Judge mode: {mode}\n\n"
        f"Question:\n{question}\n\n"
        f"{image_note}"
        f"{JUDGE_GUIDANCE}\n"
        "Generate a constrained judge card tailored to this task family.\n"
        "The judge should reward transcript-grounded evidence, valid elimination, explicit verification, "
        "and coherent constraint tracking.\n"
        "The judge should discount unsupported confidence, verbosity, superficial consensus, and fresh "
        "reasoning that ignores the transcript.\n"
        "The judge card must be multi-agent compatible. It should refer to debaters, agents, or answers in "
        "the transcript rather than assuming exactly two participants.\n"
        "It must specify how to handle: majority-vs-best-argument conflicts, partial convergence between "
        "multiple agents, and cases where one agent provides the strongest support despite being in the "
        "minority.\n"
        "Generate a constrained judge card. It may be conditioned on task family and question, but must not "
        "embed answer hints before transcript review. Return JSON only matching:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate judge cards for debate evaluation. "
                "The judge is a transcript-based selector, not a fresh unconstrained solver. "
                "Tailor the judge to the task family without leaking the answer."
            ),
        },
        {"role": "user", "content": _user_content_with_media(user, question_media=question_media)},
    ]


def build_judge_bank_messages(
    *,
    dataset: str,
    judge_family: str,
    family_description: str,
) -> list[dict[str, str]]:
    schema = {
        "judge_id": "judge_identifier",
        "judge_family": judge_family,
        "domain_scope": "short scope string",
        "evaluation_priorities": ["priority 1", "priority 2", "priority 3"],
        "tie_break_policy": "How ties should be broken",
        "independent_resolve_policy": "limited_check_only",
        "answer_format_policy": "Strict answer format rule",
        "confidence_policy": "optional or null",
        "system_prompt": "Judge prompt text",
    }
    user = (
        f"Dataset: {dataset}\n"
        f"Judge family: {judge_family}\n"
        f"Family description: {family_description}\n\n"
        f"{JUDGE_GUIDANCE}\n"
        "Generate a reusable benchmark-level judge card for this family.\n"
        "This judge will be reused across many items and should be stable, transcript-grounded, and reproducible.\n"
        "It should evaluate debate traces, not solve items from scratch.\n"
        "The judge may use visible model outputs plus Gemini thought summaries when available.\n"
        "Return JSON only matching:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate reusable benchmark-level judge cards. "
                "The judge is transcript-grounded, reusable across items, and should avoid fresh solving."
            ),
        },
        {"role": "user", "content": user},
    ]
