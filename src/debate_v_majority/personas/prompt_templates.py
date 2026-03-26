from __future__ import annotations

import json
import re
from typing import Any


ARTIFACT_VERSION = "phase1.persona.v2"
AXIS_PROMPT_VERSION = "phase0.axes.v4"
DESCRIPTOR_PROMPT_VERSION = "phase1.descriptors.v2"
CARD_PROMPT_VERSION = "phase1.cards.v2"
JUDGE_PROMPT_VERSION = "phase0.judge.v4"
JUDGE_BANK_PROMPT_VERSION = "phase0.judge_bank.v1"

STAGE1_GUIDANCE = (
    "Generate a covered population of solver personas for this prompt. Maximize support coverage over plausible "
    "reasoning behaviors instead of clustering around the safest middle style. Each persona should imply a "
    "meaningfully different round-1 search trajectory, different evidence priorities, and a different failure mode. "
    "Condition on the prompt only at the level of approach: what this persona notices first, what it trusts, "
    "what it is vulnerable to, and how it reacts to disagreement. Do not include answer hints, option hints, "
    "hidden cruxes, theorem giveaways, or benchmark-specific solution templates."
)

STAGE2_GUIDANCE = (
    "Expand each descriptor into a compact stage-aware operating card. The card should preserve one coherent "
    "persona identity across rounds while changing which part of that identity is active in Round 1, Round 2, "
    "and Round 3. Encode concrete operating rules, not biography, tone, or generic advice."
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


def _persona_stage_policy_schema() -> dict[str, Any]:
    return {
        "solver_first": "Round-1 solve behavior, stated as the first-pass reasoning policy",
        "critique": "Round-2 critique behavior, stated as the peer-attack policy",
        "revise": "Round-3 revision behavior, stated as the defend/revise/switch trigger",
        "confidence": "Confidence or uncertainty gating rule",
        "failure_mode": "Main internal trap to avoid",
    }


def _round1_solver_profile_schema() -> dict[str, Any]:
    return {
        "candidate_generation_policy": "How the persona generates candidate answers in round 1",
        "hypothesis_management_policy": "Whether the persona keeps multiple hypotheses alive or commits early",
        "evidence_priority_policy": "What evidence the persona privileges first",
        "pruning_policy": "How the persona prunes weak lines",
        "verification_policy": "When and how the persona verifies decisive steps",
        "abstraction_policy": "Whether the persona starts concrete or abstract",
    }


def _debate_temperament_profile_schema() -> dict[str, Any]:
    return {
        "critique_policy_summary": "How the persona attacks peer reasoning in round 2",
        "revision_policy_summary": "How the persona updates in round 3",
        "peer_interaction_policy": "How the persona responds to disagreement or consensus",
    }


def _descriptor_profiles_schema() -> dict[str, Any]:
    return {
        "question_approach_summary": "How this persona approaches this prompt before seeing peer context",
        "disagreement_profile": "How this persona reacts to disagreement or peer critique",
        "revision_profile": "How this persona decides whether to defend, patch, or switch",
    }


def _round1_solver_policy_schema() -> dict[str, Any]:
    return {
        "opening_strategy": "The first round-1 move this persona makes",
        "candidate_generation_order": "How candidate paths are sequenced",
        "hypothesis_retention_rule": "How long alternatives remain live",
        "early_disqualifiers": "What causes an early rejection",
        "verification_trigger": "What forces a deeper check",
    }


def _round2_critique_policy_schema() -> dict[str, Any]:
    return {
        "primary_attack_rule": "Main round-2 attack rule",
        "preferred_target_type": "What kind of peer weakness this persona targets first",
        "what_to_ignore": "What this persona intentionally de-emphasizes in critique",
    }


def _round3_revision_policy_schema() -> dict[str, Any]:
    return {
        "default_stance": "Default round-3 stance before new evidence lands",
        "switch_triggers": "Evidence that forces a switch",
        "patch_vs_rebuild_rule": "How the persona decides between patching and rebuilding",
    }


def _runtime_prompts_schema() -> dict[str, Any]:
    return {
        "initial_system_prompt": "Round-1 system prompt assembled from the persona card",
        "round2_reminder": "Short reminder for round 2",
        "round3_reminder": "Short reminder for round 3",
    }


def _extract_fenced_json(text: str) -> str | None:
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_balanced_json_prefix(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, handling fenced blocks and bare braces."""
    s = str(text or "").strip()
    for candidate in (s, _extract_fenced_json(s), _extract_braced_json(s), _extract_balanced_json_prefix(s)):
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
                "axis_role": "solver",
                "canonical_dimension": "hypothesis_management",
                "family_scope": benchmark_family,
                "stage_affinity": "round1",
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
        "Good axes must create observable differences in round-1 solve behavior, round-2 critique behavior, or "
        "round-3 revision thresholds.\n"
        "Write axes as portable debate behaviors, not as summaries of the hidden task, theorem family, object "
        "type, benchmark mechanism, or domain ontology.\n"
        "Low and high descriptions should be framed in terms like evidence standard, comparison rule, search "
        "policy, pressure test, disagreement handling, or revision trigger.\n"
        "If an axis would reveal what kind of object is being analyzed, rewrite it as a general rule for how the "
        "debater gathers evidence, compares alternatives, attacks weak reasoning, or decides to update.\n"
        "Prefer axes whose extremes would plausibly send agents down different first-pass solution paths or make "
        "them attack different weaknesses in a rival answer.\n"
        "Favor support coverage over average-case density: include axes that help span rare but plausible "
        "reasoning policies, not just the most common safe personas.\n"
        "Do not propose biography, tone, verbosity, generic competence, answer hints, theorem names, object-family "
        "labels, symbolic object types, benchmark-specific mechanism language, or any axis that implicitly nudges "
        "toward a specific final answer.\n"
        "Bad axis wording: `flow_theory_vs_axiomatic_derivation`, `surface_ion_pairing_focus`, "
        "`metadata_sensitivity`, `cohen_forcing_first`.\n"
        "Good axis wording: `constraint_first_vs_hypothesis_first`, `earliest_weak_link_attack_vs_whole_model_attack`, "
        "`local_patch_vs_global_rebuild`, `consensus_agnostic_vs_convergence_sensitive`.\n"
        "Return JSON only matching this schema:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate reasoning-diversity axes for multi-agent debate. "
                "Optimize for support coverage across plausible reasoning policies. "
                "Axes must change how the agent reasons, not what answer it is nudged toward. "
                "Reject axes that sound like a theorem family, object class, mechanism label, or domain-specific lens."
            ),
        },
        {"role": "user", "content": _user_content_with_media(user, question_media=question_media)},
    ]


def build_stage1_messages(
    *,
    dataset: str,
    benchmark_family: str,
    question: str = "",
    axes: list[dict[str, Any]],
    sampled_points: list[dict[str, float]],
    question_media: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schema = {
        "descriptors": [
            {
                "persona_id": "persona_1",
                "name": "Short descriptive name",
                "axis_interpretation": {"axis_id": "how this persona sits on that axis"},
                "solver_role": "parallel_explorer",
                **_descriptor_profiles_schema(),
                "round1_solver_profile": _round1_solver_profile_schema(),
                "debate_temperament_profile": _debate_temperament_profile_schema(),
                "likely_failure_mode": "What this persona is most likely to get wrong",
                "stage_policy": _persona_stage_policy_schema(),
                "short_rule": "One-sentence operational rule",
                "reasoning_summary": "Brief explanation of how this persona reasons",
                "revision_policy": "Legacy alias for the round-3 revision trigger",
                "confidence_policy": "Legacy alias for the confidence gate",
                "failure_mode_to_watch": "Legacy alias for the main failure mode",
            }
        ]
    }
    user = (
        f"Dataset: {dataset}\n"
        f"Benchmark family: {benchmark_family}\n\n"
        "You are generating solver-first reasoning personas for this prompt.\n"
        "The resulting personas should still feel reusable across any item with a similar reasoning shape; tailor the approach, not the answer.\n\n"
        "Prompt:\n"
        f"{question}\n\n"
        "Axes:\n"
        f"{json.dumps(axes, indent=2, ensure_ascii=False)}\n\n"
        "Persona slots and sampled positions:\n"
        f"{json.dumps(sampled_points, indent=2, ensure_ascii=False)}\n\n"
        "Treat the sampled positions as hard anchors for population design.\n"
        "Plan the population jointly so that the full set covers the axis space rather than producing "
        "near-duplicate safe personas.\n"
        "Ensure the set contains informative contrasts: some extremes, some cross-axis mixtures, and at "
        "most one near-center compromise persona if forced by the sampled points.\n"
        "If two personas would likely produce similar reasoning traces, sharpen them until their solve "
        "policy, critique habit, revision trigger, and recovery path are clearly different.\n"
        "Translate axis positions into general execution tendencies. You may tailor them to this prompt, but only "
        "at the level of approach: what the persona notices first, what kind of evidence it privileges, what "
        "shape of mistake it is likely to make, and how it reacts to disagreement.\n"
        "Allowed descriptor content: how the persona generates candidates, what evidence it privileges, when "
        "it prunes, when it verifies, how it reacts to disagreement, and what kinds of mistakes it makes on this prompt.\n"
        "Disallowed descriptor content: answer predictions, option elimination, hidden cruxes, theorem families, "
        "benchmark-specific mechanisms, or narrow method templates unless they are explicitly present in the prompt "
        "and are abstracted into a reasoning behavior rather than a solve recipe.\n"
        "Return both the structured `stage_policy` object and the legacy flat aliases. The structured object "
        "is canonical; the aliases should repeat the same content.\n"
        "`question_approach_summary` should explain how this persona approaches *this prompt* before seeing peers.\n"
        "`disagreement_profile` should explain how this persona handles peer disagreement.\n"
        "`revision_profile` should explain how this persona updates once criticism lands.\n"
        "`stage_policy.solver_first` is the canonical round-1 solve policy.\n"
        "`stage_policy.critique` is the canonical round-2 peer critique policy.\n"
        "`stage_policy.revise` is the canonical round-3 revision trigger.\n"
        "`stage_policy.confidence` is the canonical confidence gate.\n"
        "`stage_policy.failure_mode` is the canonical main trap to avoid.\n"
        "`solver_role` should identify the round-1 search role, like `parallel_explorer`, `committed_builder`, "
        "`global_theorist`, `local_verifier`, or `skeptical_falsifier`.\n"
        "`round1_solver_profile` is the detailed round-1 search policy.\n"
        "`debate_temperament_profile` summarizes how the persona critiques and updates later.\n"
        "`likely_failure_mode` should name the persona's characteristic regression risk.\n"
        "`short_rule` should be a compact solver-first summary.\n"
        "`reasoning_summary` should explain how the persona behaves across stages.\n"
        "`revision_policy`, `confidence_policy`, and `failure_mode_to_watch` should mirror the structured object.\n"
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
                "You are generating a coordinated set of debater personas for a prompt. "
                "Plan the whole population jointly. They must feel like stable reasoning personas, not debate staff roles. "
                "Condition on the prompt only at the level of approach, not answer hints or hidden solution structure."
            ),
        },
        {"role": "user", "content": _user_content_with_media(user, question_media=question_media)},
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
        "base_identity": "Stable one-to-two sentence identity",
        "round1_solver_policy": _round1_solver_policy_schema(),
        "round2_critique_policy": _round2_critique_policy_schema(),
        "round3_revision_policy": _round3_revision_policy_schema(),
        "stage_policy": _persona_stage_policy_schema(),
        "core_reasoning_strategy": "Round 1 solve behavior",
        "priorities": ["priority 1", "priority 2"],
        "distrusts": ["thing to distrust 1", "thing to distrust 2"],
        "decomposition_style": "Round 2 critique behavior: concrete failure pattern to expose",
        "revision_policy": "Round 3 defend/revise/switch trigger with explicit condition",
        "confidence_policy": "Short confidence and uncertainty rule",
        "failure_mode_to_avoid": "Main internal trap to avoid",
        "system_prompt": "Compact operational system prompt",
        "runtime_prompts": _runtime_prompts_schema(),
    }
    descriptor_payload = dict(descriptor)
    descriptor_stage_policy = descriptor_payload.get("stage_policy")
    if not isinstance(descriptor_stage_policy, dict):
        descriptor_stage_policy = {}
    descriptor_payload["stage_policy"] = {
        "solver_first": descriptor_stage_policy.get("solver_first")
        or descriptor_payload.get("core_reasoning_strategy")
        or descriptor_payload.get("core_policy")
        or descriptor_payload.get("short_rule")
        or "",
        "critique": descriptor_stage_policy.get("critique")
        or descriptor_payload.get("decomposition_style")
        or descriptor_payload.get("critique_policy")
        or descriptor_payload.get("reasoning_summary")
        or "",
        "revise": descriptor_stage_policy.get("revise")
        or descriptor_payload.get("revision_policy")
        or "",
        "confidence": descriptor_stage_policy.get("confidence")
        or descriptor_payload.get("confidence_policy")
        or descriptor_payload.get("round_reminder")
        or "",
        "failure_mode": descriptor_stage_policy.get("failure_mode")
        or descriptor_payload.get("failure_mode_to_avoid")
        or descriptor_payload.get("failure_mode_to_watch")
        or "",
    }
    image_note = "Relevant task images are attached with this prompt.\n\n" if question_media else ""
    user = (
        "You are generating a solver-first operational card for a prompt-conditioned persona.\n"
        "The card must not target any specific problem answer or hidden solve path.\n\n"
        "Prompt:\n"
        f"{question}\n\n"
        f"{image_note}"
        "Descriptor:\n"
        f"{json.dumps(descriptor_payload, indent=2, ensure_ascii=False)}\n\n"
        "Write in terms of operating rules, not biography.\n"
        "Interpret the fields as stage-specific policies, not generic style labels.\n"
        "`stage_policy.solver_first` and `core_reasoning_strategy` are the round-1 solve behavior only: how the persona forms an initial answer or case.\n"
        "`stage_policy.critique` and `decomposition_style` are the round-2 critique behavior only: one concrete attack rule or failure pattern to expose in a peer answer.\n"
        "`stage_policy.revise` and `revision_policy` are the round-3 response behavior only: an explicit defend/revise/switch trigger using if/when/unless language tied to evidence.\n"
        "`stage_policy.confidence` and `confidence_policy` are the confidence or uncertainty gating rule.\n"
        "`stage_policy.failure_mode` and `failure_mode_to_avoid` are the main internal trap this persona should guard against.\n"
        "`base_identity` is the stable through-line that stays present in all rounds.\n"
        "`round1_solver_policy` is the canonical structured round-1 policy.\n"
        "`round2_critique_policy` is the canonical structured round-2 policy.\n"
        "`round3_revision_policy` is the canonical structured round-3 policy.\n"
        "`runtime_prompts.initial_system_prompt` is the assembled round-1 system prompt.\n"
        "`runtime_prompts.round2_reminder` and `runtime_prompts.round3_reminder` are short round-specific reminders.\n"
        "`priorities` and `distrusts` are optional, but if present they should support the same solve/critique/revise behavior.\n"
        "The card should specify how this persona chooses candidate paths for this prompt, attacks peer reasoning, handles disagreement, and decides whether to revise.\n"
        "Differences should survive contact with the task and be visible in execution, not just sound different on paper.\n"
        "Stay faithful to the descriptor, but compress it into a narrower policy vocabulary. Convert any narrow wording into the underlying execution rule.\n"
        "Allowed card content: candidate-generation order, branch expansion vs pruning, verification cadence, evidence thresholds, contradiction handling, revision triggers, and confidence gating.\n"
        "Disallowed card content: answer predictions, option elimination, hidden cruxes, narrow solve templates, or any wording that bakes in a final answer.\n"
        "If a descriptor phrase is method-shaped, rewrite it to the more general policy it implies.\n"
        "Example rewrite: 'uses a continuous model' -> 'forms a coarse global approximation before verifying locally'.\n"
        "Example rewrite: 'starts from small cases' -> 'builds confidence from simple concrete probes before generalizing'.\n"
        "The card must describe portable reasoning behavior only. It must not speculate about what specific object types, formulas, or latent patterns will appear in the problem.\n"
        "Good decomposition_style: 'Attack the earliest unsupported step the peer answer depends on.'\n"
        "Good decomposition_style: 'Flag when a peer answer survives only by skipping a necessary case or rule.'\n"
        "Good revision_policy: 'Defend unless a critique exposes a contradiction or missing necessary case; switch only if the conclusion no longer holds.'\n"
        "Good revision_policy: 'Revise if a peer breaks a required step; otherwise answer objections without changing the core line.'\n"
        "Forbidden wording: do not write `the answer is`, `likely answer`, `correct answer`, `answer:`, `option A`, `option B`, `option C`, `option D`, `option E`, `rule out option`, or any text that predicts or names a final answer.\n"
        "Good card move: 'propose a global invariant early, then reject branches that violate it and only switch strategies after an explicit contradiction.'\n"
        "Bad card move: 'start with n=1,2,3', 'test nearby integers', 'look for geometric means', or any other task template.\n"
        "Do not mention item-specific equations, constants, answer choices, variable names, images, or any solve plan for a particular question.\n"
        "Return both the structured `stage_policy` object and the legacy flat aliases. The structured object is canonical; the aliases should repeat the same content.\n"
        f"{STAGE2_GUIDANCE}\n"
        "Return JSON only matching:\n"
        f"{_json_schema_block(schema)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are expanding a persona descriptor into a compact operational card. "
                "The card must reliably induce a distinct reasoning policy while staying concise. "
                "Use the prompt to tailor the approach, but never leak an answer or encode a narrow solve recipe."
            ),
        },
        {"role": "user", "content": _user_content_with_media(user, question_media=question_media)},
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
