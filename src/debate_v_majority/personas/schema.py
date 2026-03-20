from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


AxisKind = Literal["fixed", "task", "hybrid_component"]
AxisSelectionMode = Literal["fixed", "task", "hybrid", "file", "replay"]
JudgePersonaMode = Literal[
    "neutral_baseline",
    "task_family_generated",
    "question_conditioned_generated",
    "benchmark_family_bank",
]
ValidatorStatus = Literal["accept", "retry", "reject_hard"]


@dataclass(frozen=True)
class Axis:
    axis_id: str
    name: str
    kind: AxisKind
    low_desc: str
    high_desc: str
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
class PersonaDescriptor:
    persona_id: str
    name: str
    axis_values: dict[str, float]
    axis_interpretation: dict[str, str]
    short_rule: str
    reasoning_summary: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaDescriptor":
        return cls(
            persona_id=str(data["persona_id"]),
            name=str(data["name"]),
            axis_values={str(k): float(v) for k, v in (data.get("axis_values") or {}).items()},
            axis_interpretation={str(k): str(v) for k, v in (data.get("axis_interpretation") or {}).items()},
            short_rule=str(data["short_rule"]),
            reasoning_summary=str(data["reasoning_summary"]),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaCard":
        return cls(
            persona_id=str(data["persona_id"]),
            title=str(data["title"]),
            core_reasoning_strategy=str(data["core_reasoning_strategy"]),
            priorities=[str(x) for x in data.get("priorities", [])],
            distrusts=[str(x) for x in data.get("distrusts", [])],
            decomposition_style=str(data["decomposition_style"]),
            revision_policy=str(data["revision_policy"]),
            confidence_policy=str(data["confidence_policy"]),
            failure_mode_to_avoid=str(data["failure_mode_to_avoid"]),
            system_prompt=str(data["system_prompt"]),
            card_version=str(data["card_version"]),
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
    axis_mode: AxisSelectionMode = "hybrid"
    fixed_axis_count: int = 4
    task_axis_count: int = 2
    sampling_method: Literal["maximin", "halton"] = "maximin"
    generator_model: str | None = None
    judge_generator_model: str | None = None
    judge_persona_mode: JudgePersonaMode = "task_family_generated"
    backend: Literal["llm"] = "llm"
    axes_file: Path | None = None
    n_plain_agents: int = 0
