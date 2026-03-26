from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


AxisKind = Literal["fixed", "task", "hybrid_component"]
AxisSelectionMode = Literal["fixed", "task", "hybrid", "file", "replay"]
AxisRole = Literal["solver", "debate"]
AxisCanonicalDimension = Literal[
    "commitment_style",
    "abstraction_preference",
    "evidence_preference",
    "search_strategy",
    "verification_timing",
    "social_update_style",
    "hypothesis_management",
    "construction_mode",
    "search_direction",
    "verification_style",
    "abstraction_mode",
    "critique_mode",
    "revision_mode",
    "confidence_mode",
]
JudgePersonaMode = Literal[
    "neutral_baseline",
    "task_family_generated",
    "question_conditioned_generated",
    "benchmark_family_bank",
]
ValidatorStatus = Literal["accept", "retry", "reject_hard"]


def _coerce_str(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(k): float(v) for k, v in value.items()}


def _coerce_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


@dataclass(frozen=True)
class Axis:
    axis_id: str
    name: str
    kind: AxisKind
    low_desc: str
    high_desc: str
    axis_role: AxisRole = "solver"
    canonical_dimension: AxisCanonicalDimension = "hypothesis_management"
    family_scope: str | None = None
    stage_affinity: str = "all"
    notes: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Axis":
        return cls(
            axis_id=str(data["axis_id"]),
            name=str(data["name"]),
            kind=data["kind"],
            low_desc=str(data["low_desc"]),
            high_desc=str(data["high_desc"]),
            axis_role=str(data.get("axis_role") or "solver"),
            canonical_dimension=str(data.get("canonical_dimension") or "hypothesis_management"),
            family_scope=_coerce_str(data.get("family_scope"), default="") or None,
            stage_affinity=_coerce_str(data.get("stage_affinity"), default="all"),
            notes=data.get("notes"),
            source=dict(data.get("source") or {}),
        )


@dataclass(frozen=True)
class AxisSelection:
    mode: AxisSelectionMode
    axes: list[Axis]
    benchmark_family: str | None
    question_summary: str | None
    generator_prompt_version: str
    generator_model: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AxisSelection":
        return cls(
            mode=data["mode"],
            axes=[Axis.from_dict(x) for x in data.get("axes", [])],
            benchmark_family=data.get("benchmark_family"),
            question_summary=data.get("question_summary"),
            generator_prompt_version=str(data.get("generator_prompt_version") or ""),
            generator_model=data.get("generator_model"),
        )


@dataclass(frozen=True)
class PersonaStagePolicy:
    solver_first: str = ""
    critique: str = ""
    revise: str = ""
    confidence: str = ""
    failure_mode: str = ""

    @property
    def solve_style(self) -> str:
        return self.solver_first

    @property
    def critique_style(self) -> str:
        return self.critique

    @property
    def revision_policy(self) -> str:
        return self.revise

    @property
    def confidence_policy(self) -> str:
        return self.confidence

    @property
    def failure_mode_to_watch(self) -> str:
        return self.failure_mode

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "PersonaStagePolicy":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls()
        return cls(
            solver_first=_coerce_str(
                data.get("solver_first")
                or data.get("solve_first")
                or data.get("round_1")
                or data.get("round1")
                or data.get("solve")
                or data.get("solve_policy")
                or data.get("core_reasoning_strategy")
                or data.get("short_rule")
            ),
            critique=_coerce_str(
                data.get("critique")
                or data.get("round_2")
                or data.get("round2")
                or data.get("critique_style")
                or data.get("critique_policy")
                or data.get("decomposition_style")
                or data.get("reasoning_summary")
            ),
            revise=_coerce_str(
                data.get("revise")
                or data.get("revision")
                or data.get("round_3")
                or data.get("round3")
                or data.get("revision_policy")
                or data.get("revision_trigger")
            ),
            confidence=_coerce_str(
                data.get("confidence")
                or data.get("confidence_policy")
                or data.get("confidence_rule")
                or data.get("round_reminder")
            ),
            failure_mode=_coerce_str(
                data.get("failure_mode")
                or data.get("failure_mode_to_avoid")
                or data.get("failure_mode_to_watch")
            ),
        )


def _merge_stage_policy(
    *,
    stage_policy: Any,
    solver_first: str = "",
    critique: str = "",
    revise: str = "",
    confidence: str = "",
    failure_mode: str = "",
) -> PersonaStagePolicy:
    policy = PersonaStagePolicy.from_dict(stage_policy)
    return PersonaStagePolicy(
        solver_first=policy.solver_first or _coerce_str(solver_first),
        critique=policy.critique or _coerce_str(critique),
        revise=policy.revise or _coerce_str(revise),
        confidence=policy.confidence or _coerce_str(confidence),
        failure_mode=policy.failure_mode or _coerce_str(failure_mode),
    )


@dataclass(frozen=True)
class PersonaDescriptor:
    persona_id: str
    name: str
    axis_values: dict[str, float]
    axis_interpretation: dict[str, str]
    short_rule: str
    reasoning_summary: str
    question_approach_summary: str = ""
    disagreement_profile: str = ""
    revision_profile: str = ""
    solver_role: str = ""
    round1_solver_profile: dict[str, str] = field(default_factory=dict)
    debate_temperament_profile: dict[str, str] = field(default_factory=dict)
    likely_failure_mode: str = ""
    revision_policy: str = ""
    confidence_policy: str = ""
    failure_mode_to_watch: str = ""
    stage_policy: PersonaStagePolicy = field(default_factory=PersonaStagePolicy)

    @property
    def axis_signature(self) -> dict[str, str]:
        return self.axis_interpretation

    @property
    def solve_style(self) -> str:
        return self.question_approach_summary or self.short_rule

    @property
    def critique_style(self) -> str:
        return self.disagreement_profile or self.reasoning_summary

    @property
    def solver_first_policy(self) -> str:
        return self.stage_policy.solver_first or self.short_rule

    @property
    def stage_policy_dict(self) -> dict[str, str]:
        return self.stage_policy.to_dict()

    def __post_init__(self) -> None:
        stage_policy = _merge_stage_policy(
            stage_policy=self.stage_policy,
            solver_first=self.question_approach_summary or self.short_rule,
            critique=self.disagreement_profile or self.reasoning_summary,
            revise=self.revision_profile or self.revision_policy,
            confidence=self.confidence_policy,
            failure_mode=self.failure_mode_to_watch or self.likely_failure_mode,
        )
        object.__setattr__(self, "stage_policy", stage_policy)
        if not self.question_approach_summary and (self.short_rule or stage_policy.solver_first):
            object.__setattr__(
                self,
                "question_approach_summary",
                self.short_rule or stage_policy.solver_first,
            )
        if not self.disagreement_profile and (self.reasoning_summary or stage_policy.critique):
            object.__setattr__(
                self,
                "disagreement_profile",
                self.reasoning_summary or stage_policy.critique,
            )
        if not self.revision_profile and (self.revision_policy or stage_policy.revise):
            object.__setattr__(
                self,
                "revision_profile",
                self.revision_policy or stage_policy.revise,
            )
        if not self.short_rule and stage_policy.solver_first:
            object.__setattr__(self, "short_rule", stage_policy.solver_first)
        if not self.reasoning_summary and stage_policy.critique:
            object.__setattr__(self, "reasoning_summary", stage_policy.critique)
        if not self.revision_policy and stage_policy.revise:
            object.__setattr__(self, "revision_policy", stage_policy.revise)
        if not self.confidence_policy and stage_policy.confidence:
            object.__setattr__(self, "confidence_policy", stage_policy.confidence)
        if not self.failure_mode_to_watch and stage_policy.failure_mode:
            object.__setattr__(self, "failure_mode_to_watch", stage_policy.failure_mode)
        if not self.likely_failure_mode and stage_policy.failure_mode:
            object.__setattr__(self, "likely_failure_mode", stage_policy.failure_mode)
        if not self.round1_solver_profile:
            object.__setattr__(
                self,
                "round1_solver_profile",
                {
                    "candidate_generation_policy": stage_policy.solver_first or self.short_rule,
                    "hypothesis_management_policy": self.short_rule,
                    "evidence_priority_policy": self.question_approach_summary or self.reasoning_summary,
                    "pruning_policy": stage_policy.revise or self.revision_policy or self.short_rule,
                    "verification_policy": stage_policy.confidence or self.confidence_policy or self.reasoning_summary,
                    "abstraction_policy": self.question_approach_summary or self.reasoning_summary,
                },
            )
        if not self.debate_temperament_profile:
            object.__setattr__(
                self,
                "debate_temperament_profile",
                {
                    "critique_policy_summary": stage_policy.critique or self.disagreement_profile or self.reasoning_summary,
                    "revision_policy_summary": stage_policy.revise or self.revision_profile or self.revision_policy,
                    "peer_interaction_policy": stage_policy.confidence or self.confidence_policy,
                },
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaDescriptor":
        stage_policy_data = data.get("stage_policy") or data.get("stage_profile") or {}
        if not isinstance(stage_policy_data, dict):
            stage_policy_data = {}
        stage_policy = _merge_stage_policy(
            stage_policy=stage_policy_data,
            solver_first=data.get("short_rule")
            or data.get("solve_style")
            or data.get("solver_first_policy")
            or "",
            critique=data.get("reasoning_summary") or data.get("critique_style") or "",
            revise=data.get("revision_policy") or "",
            confidence=data.get("confidence_policy") or "",
            failure_mode=data.get("failure_mode_to_watch") or data.get("likely_failure_mode") or "",
        )
        return cls(
            persona_id=str(data["persona_id"]),
            name=str(data["name"]),
            axis_values=_coerce_float_dict(data.get("axis_values")),
            axis_interpretation=_coerce_string_dict(
                data.get("axis_interpretation")
                or data.get("axis_signature")
                or data.get("axis_interpretations")
            ),
            question_approach_summary=str(
                data.get("question_approach_summary")
                or data.get("solver_identity_summary")
                or data.get("short_rule")
                or data.get("solve_style")
                or stage_policy_data.get("solver_first")
                or "approach the prompt with a distinct operational solving policy"
            ),
            disagreement_profile=str(
                data.get("disagreement_profile")
                or data.get("reasoning_summary")
                or data.get("critique_style")
                or stage_policy_data.get("critique")
                or "attack the most consequential unsupported step in competing answers"
            ),
            revision_profile=str(
                data.get("revision_profile")
                or data.get("revision_policy")
                or stage_policy_data.get("revise")
                or ""
            ),
            short_rule=str(
                data.get("short_rule")
                or data.get("question_approach_summary")
                or data.get("solve_style")
                or data.get("solver_first_policy")
                or stage_policy_data.get("solver_first")
                or "apply a distinct operational reasoning policy"
            ),
            reasoning_summary=str(
                data.get("reasoning_summary")
                or data.get("disagreement_profile")
                or data.get("critique_style")
                or stage_policy_data.get("critique")
                or "attack the most consequential unsupported step in competing answers"
            ),
            solver_role=str(data.get("solver_role") or ""),
            round1_solver_profile=_coerce_string_dict(data.get("round1_solver_profile")),
            debate_temperament_profile=_coerce_string_dict(data.get("debate_temperament_profile")),
            likely_failure_mode=str(
                data.get("likely_failure_mode")
                or stage_policy_data.get("failure_mode")
                or data.get("failure_mode_to_watch")
                or ""
            ),
            revision_policy=str(
                data.get("revision_profile")
                or data.get("revision_policy")
                or stage_policy_data.get("revise")
                or ""
            ),
            confidence_policy=str(
                data.get("confidence_policy")
                or stage_policy_data.get("confidence")
                or ""
            ),
            failure_mode_to_watch=str(
                data.get("failure_mode_to_watch")
                or stage_policy_data.get("failure_mode")
                or ""
            ),
            stage_policy=stage_policy,
        )


@dataclass(frozen=True)
class PersonaCard:
    persona_id: str
    title: str
    core_reasoning_strategy: str
    priorities: list[str]
    distrusts: list[str]
    decomposition_style: str
    revision_policy: str
    confidence_policy: str
    failure_mode_to_avoid: str
    system_prompt: str
    card_version: str
    base_identity: str = ""
    round1_solver_policy: dict[str, str] = field(default_factory=dict)
    round2_critique_policy: dict[str, str] = field(default_factory=dict)
    round3_revision_policy: dict[str, str] = field(default_factory=dict)
    runtime_prompts: dict[str, str] = field(default_factory=dict)
    stage_policy: PersonaStagePolicy = field(default_factory=PersonaStagePolicy)

    @property
    def core_policy(self) -> str:
        return self.core_reasoning_strategy

    @property
    def critique_policy(self) -> str:
        return self.decomposition_style

    @property
    def failure_mode_to_watch(self) -> str:
        return self.failure_mode_to_avoid

    @property
    def round_reminder(self) -> str:
        return self.confidence_policy

    @property
    def axis_signature(self) -> dict[str, str]:
        return {}

    @property
    def solver_first_policy(self) -> str:
        return self.stage_policy.solver_first or self.core_reasoning_strategy

    @property
    def stage_policy_dict(self) -> dict[str, str]:
        return self.stage_policy.to_dict()

    @property
    def initial_system_prompt(self) -> str:
        return self.runtime_prompts.get("initial_system_prompt") or self.system_prompt

    @property
    def round2_reminder(self) -> str:
        return self.runtime_prompts.get("round2_reminder") or self.decomposition_style

    @property
    def round3_reminder(self) -> str:
        return self.runtime_prompts.get("round3_reminder") or self.revision_policy

    def __post_init__(self) -> None:
        stage_policy = _merge_stage_policy(
            stage_policy=self.stage_policy,
            solver_first=self.core_reasoning_strategy,
            critique=self.decomposition_style,
            revise=self.revision_policy,
            confidence=self.confidence_policy,
            failure_mode=self.failure_mode_to_avoid,
        )
        object.__setattr__(self, "stage_policy", stage_policy)
        if not self.core_reasoning_strategy and stage_policy.solver_first:
            object.__setattr__(self, "core_reasoning_strategy", stage_policy.solver_first)
        if not self.decomposition_style and stage_policy.critique:
            object.__setattr__(self, "decomposition_style", stage_policy.critique)
        if not self.revision_policy and stage_policy.revise:
            object.__setattr__(self, "revision_policy", stage_policy.revise)
        if not self.confidence_policy and stage_policy.confidence:
            object.__setattr__(self, "confidence_policy", stage_policy.confidence)
        if not self.failure_mode_to_avoid and stage_policy.failure_mode:
            object.__setattr__(self, "failure_mode_to_avoid", stage_policy.failure_mode)
        if not self.base_identity:
            object.__setattr__(self, "base_identity", self.title)
        if not self.round1_solver_policy:
            object.__setattr__(
                self,
                "round1_solver_policy",
                {
                    "opening_strategy": stage_policy.solver_first or self.core_reasoning_strategy,
                    "candidate_generation_order": self.core_reasoning_strategy,
                    "hypothesis_retention_rule": self.confidence_policy,
                    "early_disqualifiers": ", ".join(self.distrusts),
                    "verification_trigger": self.failure_mode_to_avoid,
                },
            )
        if not self.round2_critique_policy:
            object.__setattr__(
                self,
                "round2_critique_policy",
                {
                    "primary_attack_rule": stage_policy.critique or self.decomposition_style,
                    "preferred_target_type": ", ".join(self.priorities),
                    "what_to_ignore": self.confidence_policy,
                },
            )
        if not self.round3_revision_policy:
            object.__setattr__(
                self,
                "round3_revision_policy",
                {
                    "default_stance": stage_policy.revise or self.revision_policy,
                    "switch_triggers": stage_policy.revise or self.revision_policy,
                    "patch_vs_rebuild_rule": self.failure_mode_to_avoid,
                },
            )
        if not self.runtime_prompts:
            object.__setattr__(
                self,
                "runtime_prompts",
                {
                    "initial_system_prompt": self.system_prompt,
                    "round2_reminder": stage_policy.critique or self.decomposition_style,
                    "round3_reminder": stage_policy.revise or self.revision_policy,
                },
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaCard":
        stage_policy = _merge_stage_policy(
            stage_policy=data.get("stage_policy") or data.get("stage_profile"),
            solver_first=data.get("core_reasoning_strategy")
            or data.get("core_policy")
            or data.get("solver_first_policy")
            or "",
            critique=data.get("decomposition_style")
            or data.get("critique_policy")
            or "",
            revise=data.get("revision_policy") or "",
            confidence=data.get("confidence_policy") or data.get("round_reminder") or "",
            failure_mode=data.get("failure_mode_to_avoid")
            or data.get("failure_mode_to_watch")
            or "",
        )
        runtime_prompts = _coerce_string_dict(data.get("runtime_prompts"))
        round1_solver_policy = _coerce_string_dict(data.get("round1_solver_policy"))
        round2_critique_policy = _coerce_string_dict(data.get("round2_critique_policy"))
        round3_revision_policy = _coerce_string_dict(data.get("round3_revision_policy"))
        return cls(
            persona_id=str(data["persona_id"]),
            title=str(data["title"]),
            core_reasoning_strategy=str(
                data.get("core_reasoning_strategy")
                or data.get("core_policy")
                or stage_policy.solver_first
                or ""
            ),
            priorities=_coerce_str_list(data.get("priorities")),
            distrusts=_coerce_str_list(data.get("distrusts")),
            decomposition_style=str(
                data.get("decomposition_style")
                or data.get("critique_policy")
                or stage_policy.critique
                or ""
            ),
            revision_policy=str(data.get("revision_policy") or stage_policy.revise or ""),
            confidence_policy=str(
                data.get("confidence_policy")
                or data.get("round_reminder")
                or stage_policy.confidence
                or ""
            ),
            failure_mode_to_avoid=str(
                data.get("failure_mode_to_avoid")
                or data.get("failure_mode_to_watch")
                or stage_policy.failure_mode
                or ""
            ),
            system_prompt=str(data["system_prompt"]),
            card_version=str(data["card_version"]),
            base_identity=str(data.get("base_identity") or data.get("title") or ""),
            round1_solver_policy=round1_solver_policy,
            round2_critique_policy=round2_critique_policy,
            round3_revision_policy=round3_revision_policy,
            runtime_prompts=runtime_prompts,
            stage_policy=stage_policy,
        )


@dataclass(frozen=True)
class JudgeCard:
    judge_id: str
    judge_family: str
    domain_scope: str
    evaluation_priorities: list[str]
    tie_break_policy: str
    independent_resolve_policy: str
    answer_format_policy: str
    confidence_policy: str | None
    system_prompt: str
    card_version: str
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeCard":
        return cls(
            judge_id=str(data["judge_id"]),
            judge_family=str(data["judge_family"]),
            domain_scope=str(data["domain_scope"]),
            evaluation_priorities=[str(x) for x in data.get("evaluation_priorities", [])],
            tie_break_policy=str(data["tie_break_policy"]),
            independent_resolve_policy=str(data["independent_resolve_policy"]),
            answer_format_policy=str(data["answer_format_policy"]),
            confidence_policy=data.get("confidence_policy"),
            system_prompt=str(data["system_prompt"]),
            card_version=str(data["card_version"]),
            source=dict(data.get("source") or {}),
        )


SlotMode = Literal["persona", "plain"]


def build_slot_layout(*, n_agents: int, n_plain_agents: int) -> list[SlotMode]:
    """Build a slot layout placing plain agents first, then persona agents."""
    return (["plain"] * n_plain_agents) + (["persona"] * (n_agents - n_plain_agents))


@dataclass(frozen=True)
class PersonaArtifact:
    artifact_version: str
    dataset: str
    item_uid: str
    dataset_revision: str | None
    item_display_id: str | int | None
    persona_seed: int | None
    generator_model: str | None
    judge_generator_model: str | None
    axes: AxisSelection
    sampled_points: list[dict[str, float]]
    descriptors: list[PersonaDescriptor]
    cards: list[PersonaCard]
    judge_card: JudgeCard | None
    prompt_versions: dict[str, str]
    created_at: str
    generation_settings: dict[str, Any] = field(default_factory=dict)
    validator_metadata: dict[str, Any] = field(default_factory=dict)
    slot_layout: list[SlotMode] | None = None

    def card_for_agent(self, agent_idx: int) -> PersonaCard | None:
        """Return the persona card assigned to *agent_idx*, or ``None`` for
        plain slots / out-of-range indices."""
        if self.slot_layout is None:
            if agent_idx < len(self.cards):
                return self.cards[agent_idx]
            return None
        if agent_idx >= len(self.slot_layout):
            return None
        if self.slot_layout[agent_idx] != "persona":
            return None
        card_idx = sum(1 for i in range(agent_idx) if self.slot_layout[i] == "persona")
        if card_idx < len(self.cards):
            return self.cards[card_idx]
        return None

    @property
    def n_plain_agents(self) -> int:
        if self.slot_layout is None:
            return 0
        return sum(1 for s in self.slot_layout if s == "plain")

    @property
    def n_total_agents(self) -> int:
        if self.slot_layout is None:
            return len(self.cards)
        return len(self.slot_layout)

    def persona_agent_indices(self) -> list[int]:
        if self.slot_layout is None:
            return list(range(len(self.cards)))
        return [i for i, s in enumerate(self.slot_layout) if s == "persona"]

    def plain_agent_indices(self) -> list[int]:
        if self.slot_layout is None:
            return []
        return [i for i, s in enumerate(self.slot_layout) if s == "plain"]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.slot_layout is None:
            data.pop("slot_layout", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaArtifact":
        raw_layout = data.get("slot_layout")
        slot_layout: list[SlotMode] | None = None
        if isinstance(raw_layout, list) and raw_layout:
            slot_layout = [str(s) for s in raw_layout]  # type: ignore[misc]
        return cls(
            artifact_version=str(data["artifact_version"]),
            dataset=str(data["dataset"]),
            item_uid=str(data["item_uid"]),
            dataset_revision=data.get("dataset_revision"),
            item_display_id=data.get("item_display_id"),
            persona_seed=data.get("persona_seed"),
            generator_model=data.get("generator_model"),
            judge_generator_model=data.get("judge_generator_model"),
            axes=AxisSelection.from_dict(data["axes"]),
            sampled_points=[
                {str(k): float(v) for k, v in point.items()}
                for point in data.get("sampled_points", [])
            ],
            descriptors=[PersonaDescriptor.from_dict(x) for x in data.get("descriptors", [])],
            cards=[PersonaCard.from_dict(x) for x in data.get("cards", [])],
            judge_card=JudgeCard.from_dict(data["judge_card"]) if data.get("judge_card") else None,
            prompt_versions={str(k): str(v) for k, v in (data.get("prompt_versions") or {}).items()},
            created_at=str(data["created_at"]),
            generation_settings=dict(data.get("generation_settings") or {}),
            validator_metadata=dict(data.get("validator_metadata") or {}),
            slot_layout=slot_layout,
        )


@dataclass(frozen=True)
class JudgeBankArtifact:
    artifact_version: str
    dataset: str
    judge_family: str
    generator_model: str | None
    backend: str
    prompt_versions: dict[str, str]
    created_at: str
    judge_card: JudgeCard
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeBankArtifact":
        return cls(
            artifact_version=str(data["artifact_version"]),
            dataset=str(data["dataset"]),
            judge_family=str(data["judge_family"]),
            generator_model=data.get("generator_model"),
            backend=str(data.get("backend") or "llm"),
            prompt_versions={str(k): str(v) for k, v in (data.get("prompt_versions") or {}).items()},
            created_at=str(data["created_at"]),
            judge_card=JudgeCard.from_dict(data["judge_card"]),
            source=dict(data.get("source") or {}),
        )


@dataclass(frozen=True)
class ValidationResult:
    status: ValidatorStatus
    reasons: list[str]


@dataclass(frozen=True)
class PersonaGenerationConfig:
    dataset: str
    question: str
    raw_task: dict[str, Any]
    item_uid: str
    item_display_id: str | int | None
    dataset_revision: str | None
    n_personas: int = 5
    persona_seed: int = 0
    axis_mode: AxisSelectionMode = "fixed"
    fixed_axis_count: int = 6
    task_axis_count: int = 2
    sampling_method: Literal["maximin", "halton"] = "maximin"
    generator_model: str | None = None
    judge_generator_model: str | None = None
    judge_persona_mode: JudgePersonaMode = "task_family_generated"
    backend: Literal["llm"] = "llm"
    axes_file: Path | None = None
    n_plain_agents: int = 0
