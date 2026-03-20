from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...datasets import list_dataset_adapters


@dataclass(frozen=True)
class RunMeta:
    path: str
    dataset: str
    mode: str
    method_label: str
    n: int | None
    seed: int | None
    n_samples: int | None
    n_agents: int | None
    n_rounds: int | None
    tag_all: bool
    schema_version: str | None
    row_origin: str | None
    model_tag: str | None
    ts: str | None


@dataclass
class RunSummary:
    meta: RunMeta
    n_questions: int
    final_correct: int
    final_incorrect: int
    final_none: int
    judge_correct: int | None = None
    judge_incorrect: int | None = None
    judge_none: int | None = None
    majority_correct: int | None = None
    majority_incorrect: int | None = None
    majority_none: int | None = None
    round1_majority_correct: int | None = None
    round1_majority_incorrect: int | None = None
    round1_majority_none: int | None = None


FILENAME_RE = re.compile(
    r"^(?P<mode>debate|majority|single)_(?P<dataset>aime|gpqa|hle)"
    r"(?:_agents(?P<agents>\d+)_r(?P<rounds>\d+))?"
    r"(?:_samples(?P<samples>\d+))?"
    r"_n(?P<n>\d+)"
    r"_seed(?P<seed>\d+)"
    r"(?P<all>_all)?"
    r"_(?P<ts>\d{8}_\d{6})"
    r"(?:_(?P<org>[^_]+)_(?P<model>.+?))?"
    r"\.jsonl$"
)


def normalize_dataset_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"aime", "aime25"}:
        return "aime25"
    if lowered == "gpqa":
        return "gpqa"
    if lowered == "hle":
        return "hle"
    return None


def parse_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def method_label_from_row(row: dict[str, Any] | None, *, mode: str) -> str:
    row = row or {}
    if mode == "single":
        return str(row.get("row_origin") or "single_independent")
    if mode == "majority":
        origin = (
            row.get("majority_origin")
            or (row.get("majority_result") or {}).get("result_origin")
            or row.get("row_origin")
        )
        if origin == "standalone_persona":
            return "standalone_persona_majority"
        if origin == "standalone_sampling":
            return "standalone_sampling_majority"
        return "standalone_majority"
    if mode == "debate":
        strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
        persona_meta = row.get("persona_meta") if isinstance(row.get("persona_meta"), dict) else {}
        n_plain = (
            strategy.get("persona_plain_agents")
            or persona_meta.get("n_plain_agents")
            or 0
        )
        if int(n_plain) > 0:
            return "debate_judge_mixed"
        return "debate_judge"
    return mode


def legacy_parse_run_meta(path: Path) -> RunMeta | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    mode = m.group("mode")
    dataset_short = m.group("dataset")
    dataset = "aime25" if dataset_short == "aime" else dataset_short
    n_agents = int(m.group("agents")) if m.group("agents") else None
    n_rounds = int(m.group("rounds")) if m.group("rounds") else None
    n_samples = int(m.group("samples")) if m.group("samples") else None
    n = int(m.group("n")) if m.group("n") else None
    seed = int(m.group("seed")) if m.group("seed") else None
    tag_all = bool(m.group("all"))
    if m.group("org") and m.group("model"):
        model_tag = f"{m.group('org')}/{m.group('model')}".replace("__", "_")
    else:
        model_tag = None
    ts = m.group("ts") if m.group("ts") else None
    return RunMeta(
        path=str(path),
        dataset=dataset,
        mode=mode,
        method_label=method_label_from_row(None, mode=mode),
        n=n,
        seed=seed,
        n_samples=n_samples,
        n_agents=n_agents,
        n_rounds=n_rounds,
        tag_all=tag_all,
        schema_version=None,
        row_origin=None,
        model_tag=model_tag,
        ts=ts,
    )


def parse_run_meta(path: Path, first_row: dict[str, Any] | None = None) -> RunMeta:
    legacy = legacy_parse_run_meta(path)
    row = first_row or {}
    run_meta = row.get("run_meta") if isinstance(row.get("run_meta"), dict) else {}
    strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    mode = str(row.get("mode") or strategy.get("mode") or (legacy.mode if legacy is not None else "")).strip().lower()
    dataset = (
        normalize_dataset_name(row.get("dataset"))
        or normalize_dataset_name(task.get("dataset"))
        or normalize_dataset_name(run_meta.get("dataset"))
        or (legacy.dataset if legacy is not None else None)
    )
    if mode not in {"debate", "majority", "single"} or dataset is None:
        raise ValueError(f"Unrecognized run metadata for {path.name}")

    row_model_tag = row.get("debater_model") if mode == "debate" else row.get("model_name")
    if not row_model_tag:
        row_model_tag = row.get("model_tag")

    return RunMeta(
        path=str(path),
        dataset=dataset,
        mode=mode,
        method_label=method_label_from_row(row, mode=mode),
        n=(
            parse_int_or_none(row.get("subset_size"))
            or parse_int_or_none(run_meta.get("dataset_meta", {}).get("subset_size") if isinstance(run_meta.get("dataset_meta"), dict) else None)
            or (legacy.n if legacy is not None else None)
        ),
        seed=(
            parse_int_or_none(row.get("subset_seed"))
            or parse_int_or_none(run_meta.get("dataset_meta", {}).get("seed") if isinstance(run_meta.get("dataset_meta"), dict) else None)
            or (legacy.seed if legacy is not None else None)
        ),
        n_samples=parse_int_or_none(row.get("n_samples")) or parse_int_or_none(strategy.get("n_samples")) or (legacy.n_samples if legacy is not None else None),
        n_agents=parse_int_or_none(row.get("n_agents")) or parse_int_or_none(strategy.get("n_agents")) or (legacy.n_agents if legacy is not None else None),
        n_rounds=parse_int_or_none(row.get("n_rounds")) or parse_int_or_none(strategy.get("n_rounds")) or (legacy.n_rounds if legacy is not None else None),
        tag_all=bool(row.get("tag_all")) if "tag_all" in row else (legacy.tag_all if legacy is not None else False),
        schema_version=(
            str(row.get("schema_version"))
            if row.get("schema_version") is not None
            else (str(run_meta.get("output_schema_version")) if run_meta.get("output_schema_version") is not None else None)
        ),
        row_origin=(str(row.get("row_origin")) if row.get("row_origin") is not None else None),
        model_tag=(str(row_model_tag) if row_model_tag is not None else (legacy.model_tag if legacy is not None else None)),
        ts=str(row.get("run_timestamp")) if row.get("run_timestamp") is not None else (legacy.ts if legacy is not None else None),
    )


def infer_model_tag_from_siblings(path: Path) -> str | None:
    meta = legacy_parse_run_meta(path)
    if meta is None:
        return None
    if meta.model_tag:
        return meta.model_tag
    if meta.seed is None or meta.ts is None:
        return None

    sibling_model_tags: set[str] = set()
    for sib in path.parent.glob(f"*seed{meta.seed}*_{meta.ts}_*.jsonl"):
        if sib == path:
            continue
        sib_meta = legacy_parse_run_meta(sib)
        if sib_meta is None:
            continue
        if sib_meta.model_tag:
            sibling_model_tags.add(sib_meta.model_tag)
    if len(sibling_model_tags) == 1:
        return next(iter(sibling_model_tags))
    return None


def should_include_path(path: Path, *, target_model_tag: str, first_row: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    explicit = None
    if first_row is not None:
        explicit = first_row.get("debater_model") or first_row.get("model_name") or first_row.get("model_tag")
        if explicit is not None:
            explicit = str(explicit)
    effective_model_tag = explicit or infer_model_tag_from_siblings(path)
    if effective_model_tag is None:
        return (False, None)
    return (effective_model_tag == target_model_tag, effective_model_tag)


def load_adapters() -> dict[str, Any]:
    return {adapter.dataset_name: adapter for adapter in list_dataset_adapters()}
