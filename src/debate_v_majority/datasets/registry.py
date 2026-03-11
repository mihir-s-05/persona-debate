from __future__ import annotations

import inspect
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from .base import DatasetAdapter, DatasetLoadResult


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset: str
    module: str
    source_dataset_id: str | None
    source_dataset_config: str | None
    source_dataset_split: str | None
    source_dataset_revision: str | None = None
    default_filename: str = "test.jsonl"
    variant_filenames: dict[str, str] | None = None
    family_field: str | None = None
    prompt_metadata_fields: tuple[str, ...] = ()


class ModuleDatasetAdapter(DatasetAdapter):
    def __init__(self, entry: DatasetRegistryEntry) -> None:
        self.entry = entry
        self.dataset_name = entry.dataset
        self.module = importlib.import_module(entry.module)
        self._check_answer_correctness_uses_raw_task = _scorer_uses_raw_task(
            self.module.check_answer_correctness,
            module_name=entry.module,
        )

    @property
    def judge_prompt(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.module.JUDGE_PROMPT)

    def parse_question_answer(self, sample: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
        return cast(tuple[str, Any, dict[str, Any]], self.module.parse_question_answer(sample))

    def render_prompt(self, raw_task: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
        return self.parse_question_answer(raw_task)

    def parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        return self.module.parse_answer(text, task_info)

    def strict_parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        fn = getattr(self.module, "strict_parse_answer", None)
        if fn is None:
            return self.parse_answer(text, task_info)
        return fn(text, task_info)

    def recover_parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        fn = getattr(self.module, "recover_parse_answer", None)
        if fn is None:
            return self.parse_answer(text, task_info)
        return fn(text, task_info)

    def check_answer_correctness(
        self,
        answer: Any,
        gt: Any,
        raw_task: dict[str, Any] | None = None,
    ) -> int:
        if self._check_answer_correctness_uses_raw_task:
            return int(self.module.check_answer_correctness(answer, gt, raw_task))
        return int(self.module.check_answer_correctness(answer, gt))

    def construct_debate_message(self, other_agent_answers: list[str]) -> dict[str, str]:
        return cast(dict[str, str], self.module.construct_debate_message(other_agent_answers))

    def materialize(self, out_path: Path, *, variant: str | None = None) -> None:
        try:
            import datasets
        except ImportError as exc:
            raise FileNotFoundError(
                f"Missing dataset file at {out_path} and `datasets` is unavailable: {exc}"
            ) from exc

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dataset_name == "aime25":
            ds = datasets.load_dataset(
                self.entry.source_dataset_id,
                split=self.entry.source_dataset_split,
            )
            _write_jsonl(out_path, (dict(ex) for ex in ds))
            return
        if self.dataset_name == "gpqa":
            ds = datasets.load_dataset(
                self.entry.source_dataset_id,
                self.entry.source_dataset_config,
                split=self.entry.source_dataset_split,
            )
            _write_jsonl(out_path, (dict(ex) for ex in ds))
            return
        if self.dataset_name == "hle":
            ds = datasets.load_dataset(
                self.entry.source_dataset_id,
                split=self.entry.source_dataset_split,
                revision=self.entry.source_dataset_revision,
            )
            allowed = self.module.allowed_subset_labels(str(variant or "verified"))
            rows: list[dict[str, Any]] = []
            for ex in ds:
                raw_row = self.module.prepare_task(dict(ex))
                subset_label = str(raw_row.get("Verified_Classes") or "").strip()
                if subset_label not in allowed:
                    continue
                raw_row["source_variant"] = str(variant or "verified")
                raw_row["source_subset_label"] = subset_label
                raw_row["source_dataset_id"] = self.entry.source_dataset_id
                raw_row["source_dataset_config"] = self.entry.source_dataset_config
                raw_row["source_dataset_split"] = self.entry.source_dataset_split
                raw_row["source_dataset_revision"] = self.entry.source_dataset_revision
                raw_row["source_paper_version"] = self.module.HLE_PAPER_VERSION
                rows.append(raw_row)
            _write_jsonl(out_path, rows)
            return
        raise ValueError(f"Unknown adapter dataset {self.dataset_name}")

    def materialize_jsonl(self, out_path: Path, *, variant: str | None = None) -> None:
        self.materialize(out_path, variant=variant)

    def task_family(self, raw_task: dict[str, Any]) -> str | None:
        if self.entry.family_field and raw_task.get(self.entry.family_field) is not None:
            return str(raw_task.get(self.entry.family_field))
        if self.dataset_name == "hle":
            prepared = self.module.prepare_task(raw_task)
            return cast(str | None, prepared.get("domain_family"))
        return None

    def task_prompt_metadata(self, raw_task: dict[str, Any]) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for field in self.entry.prompt_metadata_fields:
            if raw_task.get(field) is not None:
                meta[field] = raw_task.get(field)
        if self.dataset_name == "hle":
            prepared = self.module.prepare_task(raw_task)
            for field in ("answer_format_type", "domain_family", "source_variant"):
                if prepared.get(field) is not None:
                    meta[field] = prepared.get(field)
        return meta

    def score(self, answer: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self.module, "score_answer", None)
        if fn is None:
            return super().score(answer, raw_task)
        return cast(dict[str, Any], fn(answer, raw_task))

    def score_answer(self, answer: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
        return self.score(answer, raw_task)

    def load_items(self, path: Path, *, registry_meta: dict[str, Any]) -> DatasetLoadResult:
        meta = dict(registry_meta)
        meta.setdefault("source_dataset_id", self.entry.source_dataset_id)
        meta.setdefault("source_dataset_config", self.entry.source_dataset_config)
        meta.setdefault("source_dataset_split", self.entry.source_dataset_split)
        meta.setdefault("source_dataset_revision", self.entry.source_dataset_revision)
        return super().load_items(path, registry_meta=meta)


_REGISTRY_CACHE: dict[str, DatasetRegistryEntry] | None = None
_ADAPTER_CACHE: dict[str, ModuleDatasetAdapter] = {}


def _registry_path() -> Path:
    return Path(__file__).with_name("registry.yaml")


def _load_registry_entries() -> dict[str, DatasetRegistryEntry]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    payload = yaml.safe_load(_registry_path().read_text(encoding="utf-8"))
    entries = {}
    for name, entry in payload["datasets"].items():
        entries[name] = DatasetRegistryEntry(
            dataset=name,
            module=str(entry["module"]),
            source_dataset_id=entry.get("source_dataset_id"),
            source_dataset_config=entry.get("source_dataset_config"),
            source_dataset_split=entry.get("source_dataset_split"),
            source_dataset_revision=entry.get("source_dataset_revision"),
            default_filename=str(entry.get("default_filename") or "test.jsonl"),
            variant_filenames={str(k): str(v) for k, v in dict(entry.get("variant_filenames") or {}).items()},
            family_field=entry.get("family_field"),
            prompt_metadata_fields=tuple(entry.get("prompt_metadata_fields", [])),
        )
    _REGISTRY_CACHE = entries
    return entries


def get_registry_entry(dataset: str) -> DatasetRegistryEntry:
    entries = _load_registry_entries()
    if dataset not in entries:
        raise KeyError(f"Unknown dataset {dataset!r}")
    return entries[dataset]


def get_dataset_adapter(dataset: str) -> ModuleDatasetAdapter:
    adapter = _ADAPTER_CACHE.get(dataset)
    if adapter is not None:
        return adapter
    adapter = ModuleDatasetAdapter(get_registry_entry(dataset))
    _ADAPTER_CACHE[dataset] = adapter
    return adapter


def list_dataset_adapters() -> list[ModuleDatasetAdapter]:
    return [get_dataset_adapter(name) for name in sorted(_load_registry_entries())]


def _write_jsonl(path: Path, rows: Any) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scorer_uses_raw_task(fn: Any, *, module_name: str) -> bool:
    signature = inspect.signature(fn)
    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    unsupported_params = [
        param
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    required_positional = [param for param in positional_params if param.default is inspect.Signature.empty]

    if unsupported_params or len(positional_params) not in (2, 3) or len(required_positional) < 2:
        raise TypeError(
            f"{module_name}.check_answer_correctness must accept either "
            f"(answer, gt) or (answer, gt, raw_task); got {signature}"
        )

    return len(positional_params) == 3
