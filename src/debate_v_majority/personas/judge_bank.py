from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..engines import ensure_inference_results, inference_result_metadata
from .prompt_templates import (
    JUDGE_BANK_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    JUDGE_GUIDANCE,
    build_judge_bank_messages,
    parse_json_payload,
)
from .schema import JudgeBankArtifact, JudgeCard

JUDGE_BANK_ARTIFACT_VERSION = "judge_bank.v1"
JUDGE_BANK_GENERATION_RETRIES = 2
AIME25_JUDGE_FAMILIES = ("math",)
GPQA_JUDGE_FAMILIES = ("biology", "chemistry", "physics")
HLE_JUDGE_FAMILIES = (
    "math",
    "physical_sciences",
    "medicine",
    "computer_science",
    "humanities",
    "applied_professional_reasoning",
)


@dataclass(frozen=True)
class JudgeFamilyAssignment:
    judge_family: str
    source: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_family": self.judge_family,
            "source": self.source,
            "details": dict(self.details),
        }


class JudgeBankGenerationExhaustedError(ValueError):
    def __init__(self, message: str, *, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def default_judge_bank_dir(*, artifacts_dir: Path) -> Path:
    return artifacts_dir / "judge_banks"


def default_gpqa_family_cache_path(*, judge_bank_dir: Path) -> Path:
    return judge_bank_dir / "gpqa_family_cache.json"


def judge_bank_path(*, judge_bank_dir: Path, dataset: str, judge_family: str) -> Path:
    safe_family = str(judge_family).replace("/", "_").replace(":", "_")
    return judge_bank_dir / dataset / f"{safe_family}.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_gpqa_family_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _save_gpqa_family_cache(path: Path, payload: dict[str, Any]) -> None:
    _save_json(path, payload)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _family_description(dataset: str, judge_family: str) -> str:
    if dataset == "aime25":
        return "Choose the mathematically sound answer from the debate transcript."
    if dataset == "gpqa":
        mapping = {
            "biology": "Choose the biologically sound answer from the debate transcript.",
            "chemistry": "Choose the chemically sound answer from the debate transcript.",
            "physics": "Choose the physically sound answer from the debate transcript.",
        }
        return mapping[judge_family]
    if dataset == "hle":
        mapping = {
            "math": "Choose the mathematically sound answer from the debate transcript.",
            "physical_sciences": "Choose the strongest physics or chemistry answer from the debate transcript.",
            "medicine": "Choose the strongest medicine or life-science answer from the debate transcript.",
            "computer_science": "Choose the strongest computer science answer from the debate transcript.",
            "humanities": "Choose the strongest humanities answer from the debate transcript.",
            "applied_professional_reasoning": "Choose the strongest applied or professional reasoning answer from the debate transcript.",
        }
        return mapping[judge_family]
    return f"Choose the strongest {judge_family} answer from the debate transcript."


def _heuristic_benchmark_judge_card(*, dataset: str, judge_family: str) -> JudgeCard:
    family_description = _family_description(dataset, judge_family)
    return JudgeCard(
        judge_id=f"judge_bank_{dataset}_{judge_family}",
        judge_family=judge_family,
        domain_scope=judge_family,
        evaluation_priorities=[
            "select the answer best supported by the transcript",
            "prefer explicit constraint tracking, valid elimination, and concrete verification",
            "discount unsupported confidence, verbosity, and superficial consensus",
        ],
        tie_break_policy="if two answers are close, prefer the answer with clearer transcript-grounded support and fewer unsupported jumps",
        independent_resolve_policy="limited_check_only",
        answer_format_policy="return one final answer in the strict format expected by the dataset",
        confidence_policy="optional",
        system_prompt=(
            f"{JUDGE_GUIDANCE}\n"
            f"Judge family: {judge_family}\n"
            f"Family scope: {family_description}\n"
            "You will review visible model outputs from every round and Gemini thought summaries when available.\n"
            "Use those traces to select the strongest supported answer. Do not run a fresh unconstrained solve."
        ),
        card_version=JUDGE_PROMPT_VERSION,
        source={"backend": "heuristic", "dataset": dataset, "judge_family": judge_family},
    )


def _llm_benchmark_judge_card(
    *,
    dataset: str,
    judge_family: str,
    engine: Any,
    generator_model: str | None,
    backend: str,
) -> JudgeCard:
    messages = build_judge_bank_messages(
        dataset=dataset,
        judge_family=judge_family,
        family_description=_family_description(dataset, judge_family),
    )
    attempt_audits: list[dict[str, Any]] = []
    for attempt in range(JUDGE_BANK_GENERATION_RETRIES + 1):
        result = ensure_inference_results(
            engine,
            [messages],
            batch_size=1,
            sampling_kwargs={"max_tokens": 4096},
            model_role="judge_generator",
        )[0]
        raw_result_text = str(result.text)
        call_metadata = inference_result_metadata(result)
        try:
            payload = parse_json_payload(raw_result_text)
        except ValueError as exc:
            attempt_audits.append(
                {
                    "attempt": attempt,
                    "request_messages": messages,
                    "raw_result_text": raw_result_text,
                    "parse_error": str(exc),
                    "call_metadata": call_metadata,
                }
            )
            continue
        return JudgeCard(
            judge_id=str(payload.get("judge_id") or f"judge_bank_{dataset}_{judge_family}"),
            judge_family=str(payload.get("judge_family") or judge_family),
            domain_scope=str(payload.get("domain_scope") or judge_family),
            evaluation_priorities=[str(x) for x in payload.get("evaluation_priorities", [])],
            tie_break_policy=str(payload.get("tie_break_policy") or "prefer clearer transcript-grounded support"),
            independent_resolve_policy=str(payload.get("independent_resolve_policy") or "limited_check_only"),
            answer_format_policy=str(payload.get("answer_format_policy") or "return one final answer in strict format"),
            confidence_policy=str(payload.get("confidence_policy")) if payload.get("confidence_policy") is not None else None,
            system_prompt=str(payload.get("system_prompt") or JUDGE_GUIDANCE),
            card_version=JUDGE_PROMPT_VERSION,
            source={
                "backend": backend,
                "generator_model": generator_model or "llm",
                "dataset": dataset,
                "judge_family": judge_family,
                "call_metadata": call_metadata,
            },
        )
    raise JudgeBankGenerationExhaustedError(
        f"Judge-bank generation exhausted retries for dataset={dataset} judge_family={judge_family}",
        metadata={
            "dataset": dataset,
            "judge_family": judge_family,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_bank_prompt_version": JUDGE_BANK_PROMPT_VERSION,
            "attempt_audits": attempt_audits,
        },
    )


def ensure_judge_bank_card(
    *,
    judge_bank_dir: Path,
    dataset: str,
    judge_family: str,
    engine: Any | None,
    generator_model: str | None,
    backend: str,
    refresh: bool = False,
) -> tuple[JudgeCard, JudgeBankArtifact, Path]:
    path = judge_bank_path(judge_bank_dir=judge_bank_dir, dataset=dataset, judge_family=judge_family)
    if path.exists() and not refresh:
        artifact = JudgeBankArtifact.from_dict(_load_json(path))
        return artifact.judge_card, artifact, path

    use_llm = backend != "heuristic" and engine is not None
    judge_card = (
        _llm_benchmark_judge_card(
            dataset=dataset,
            judge_family=judge_family,
            engine=engine,
            generator_model=generator_model,
            backend=backend,
        )
        if use_llm
        else _heuristic_benchmark_judge_card(dataset=dataset, judge_family=judge_family)
    )
    artifact = JudgeBankArtifact(
        artifact_version=JUDGE_BANK_ARTIFACT_VERSION,
        dataset=dataset,
        judge_family=judge_family,
        generator_model=generator_model,
        backend=backend if use_llm else "heuristic",
        prompt_versions={"judge": JUDGE_PROMPT_VERSION, "judge_bank": JUDGE_BANK_PROMPT_VERSION},
        created_at=_utc_now_iso(),
        judge_card=judge_card,
        source={"dataset": dataset, "judge_family": judge_family},
    )
    _save_json(path, artifact.to_dict())
    return judge_card, artifact, path


def _resolve_hle_family(raw_task: dict[str, Any]) -> JudgeFamilyAssignment:
    prepared_family = str(raw_task.get("domain_family") or "").strip().lower()
    if prepared_family in HLE_JUDGE_FAMILIES:
        return JudgeFamilyAssignment(
            judge_family=prepared_family,
            source="hle_domain_family",
            details={"domain_family": prepared_family},
        )
    category = str(raw_task.get("category") or "").strip().lower()
    if category in {"math", "mathematics"}:
        family = "math"
    elif category in {"physics", "chemistry"}:
        family = "physical_sciences"
    elif category in {"biology/medicine", "biology", "medicine"}:
        family = "medicine"
    elif category in {"computer science/ai", "computer science", "artificial intelligence", "engineering"}:
        family = "computer_science"
    elif category in {"humanities/social science", "humanities", "social science"}:
        family = "humanities"
    else:
        family = "applied_professional_reasoning"
    return JudgeFamilyAssignment(
        judge_family=family,
        source="hle_category",
        details={"category": category},
    )


def _heuristic_gpqa_family(question: str) -> str:
    """Keyword-based fallback for GPQA family classification."""
    q = question.lower()
    bio_keywords = ("cell", "protein", "gene", "organism", "dna", "rna")
    chem_keywords = ("reaction", "compound", "molecule", "bond", "acid", "ph", "oxidation")
    phys_keywords = ("force", "energy", "mass", "velocity", "quantum", "electric", "magnetic", "wave")
    bio = sum(1 for kw in bio_keywords if kw in q)
    chem = sum(1 for kw in chem_keywords if kw in q)
    phys = sum(1 for kw in phys_keywords if kw in q)
    if bio > chem and bio > phys:
        return "biology"
    if chem > bio and chem > phys:
        return "chemistry"
    # Default to physics (largest GPQA category), including ties.
    return "physics"


def _classify_gpqa_family(
    *,
    item_uid: str,
    question: str,
    raw_task: dict[str, Any],
    cache_path: Path,
    engine: Any,
    model_name: str | None,
) -> JudgeFamilyAssignment:
    cache = _load_gpqa_family_cache(cache_path)
    cached = cache.get(item_uid)
    if isinstance(cached, dict):
        family = str(cached.get("judge_family") or "").strip().lower()
        if family in GPQA_JUDGE_FAMILIES:
            return JudgeFamilyAssignment(
                judge_family=family,
                source=str(cached.get("source") or "gpqa_family_cache"),
                details=dict(cached),
            )

    schema = {"judge_family": "biology|chemistry|physics"}
    option_lines: list[str] = []
    for key in (
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "correct_answer",
        "incorrect_answer_1",
        "incorrect_answer_2",
        "incorrect_answer_3",
    ):
        value = raw_task.get(key)
        if value is not None:
            option_lines.append(f"{key}: {value}")
    user = (
        "Classify this GPQA item into exactly one domain family: biology, chemistry, or physics.\n"
        "Use the question and answer choices. Return JSON only.\n\n"
        f"Item UID: {item_uid}\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{chr(10).join(option_lines)}\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )
    result = ensure_inference_results(
        engine,
        [[
            {
                "role": "system",
                "content": "You classify GPQA questions into biology, chemistry, or physics. Return JSON only.",
            },
            {"role": "user", "content": user},
        ]],
        batch_size=1,
        sampling_kwargs={"max_tokens": 128},
        model_role="judge_generator",
    )[0]
    payload = parse_json_payload(str(result.text))
    family = str(payload.get("judge_family") or "").strip().lower()
    if family not in GPQA_JUDGE_FAMILIES:
        family = _heuristic_gpqa_family(question)
        source = "gpqa_family_heuristic_fallback"
    else:
        source = "gpqa_family_llm"
    assignment_payload = {
        "judge_family": family,
        "source": source,
        "item_uid": item_uid,
        "model_name": model_name,
        "call_metadata": inference_result_metadata(result),
        "created_at": _utc_now_iso(),
    }
    cache[item_uid] = assignment_payload
    _save_gpqa_family_cache(cache_path, cache)
    return JudgeFamilyAssignment(
        judge_family=family,
        source=source,
        details=assignment_payload,
    )


def resolve_judge_family_assignment(
    *,
    dataset: str,
    item_uid: str,
    question: str,
    raw_task: dict[str, Any],
    gpqa_family_cache_path: Path | None = None,
    gpqa_classifier_engine: Any | None = None,
    gpqa_classifier_model: str | None = None,
) -> JudgeFamilyAssignment:
    if dataset == "aime25":
        return JudgeFamilyAssignment(
            judge_family="math",
            source="dataset_fixed",
            details={"dataset": dataset},
        )
    if dataset == "hle":
        return _resolve_hle_family(raw_task)
    if dataset != "gpqa":
        return JudgeFamilyAssignment(
            judge_family=str(raw_task.get("family") or f"{dataset}_reasoning"),
            source="task_family_fallback",
            details={"dataset": dataset},
        )
    if gpqa_family_cache_path is not None and gpqa_family_cache_path.exists():
        cached = _load_gpqa_family_cache(gpqa_family_cache_path).get(item_uid)
        if isinstance(cached, dict):
            family = str(cached.get("judge_family") or "").strip().lower()
            if family in GPQA_JUDGE_FAMILIES:
                return JudgeFamilyAssignment(
                    judge_family=family,
                    source=str(cached.get("source") or "gpqa_family_cache"),
                    details=dict(cached),
                )
    if gpqa_family_cache_path is None or gpqa_classifier_engine is None:
        family = _heuristic_gpqa_family(question)
        return JudgeFamilyAssignment(
            judge_family=family,
            source="gpqa_family_heuristic",
            details={"item_uid": item_uid, "question_preview": question[:160]},
        )
    return _classify_gpqa_family(
        item_uid=item_uid,
        question=question,
        raw_task=raw_task,
        cache_path=gpqa_family_cache_path,
        engine=gpqa_classifier_engine,
        model_name=gpqa_classifier_model,
    )
