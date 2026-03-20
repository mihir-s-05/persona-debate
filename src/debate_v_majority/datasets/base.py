from __future__ import annotations

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..personas.artifacts import make_item_uid, normalize_task_payload


@dataclass(frozen=True)
class TaskItem:
    item_uid: str
    display_id: str | int | None
    dataset: str
    raw_task: dict[str, Any]
    family: str | None
    prompt_metadata: dict[str, Any]
    answer_key: Any | None
    dataset_revision: str | None


@dataclass(frozen=True)
class DatasetLoadResult:
    items: list[TaskItem]
    dataset_revision: str | None
    registry_meta: dict[str, Any]
    source_path: str


class DatasetAdapter(ABC):
    dataset_name: str

    @property
    @abstractmethod
    def judge_prompt(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_question_answer(self, sample: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def check_answer_correctness(
        self,
        answer: Any,
        gt: Any,
        raw_task: dict[str, Any] | None = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def construct_debate_message(self, other_agent_answers: list[str]) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, out_path: Path, *, variant: str | None = None) -> None:
        raise NotImplementedError

    def materialize_jsonl(self, out_path: Path, *, variant: str | None = None) -> None:
        self.materialize(out_path, variant=variant)

    def strict_parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        return self.parse_answer(text, task_info)

    def recover_parse_answer(self, text: str, task_info: dict[str, Any]) -> Any:
        return self.parse_answer(text, task_info)

    def build_judge_question(self, raw_task: dict[str, Any]) -> str | None:
        """Return the question text for the judge, without agent-facing solving instructions.

        Returns None to fall back to the agent-formatted question.
        """
        return None

    def render_prompt(self, raw_task: dict[str, Any]) -> tuple[str, Any, dict[str, Any]]:
        return self.parse_question_answer(raw_task)

    def task_family(self, raw_task: dict[str, Any]) -> str | None:
        return None

    def task_prompt_metadata(self, raw_task: dict[str, Any]) -> dict[str, Any]:
        return {}

    def score(self, answer: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
        _question, gt, _prepared = self.render_prompt(raw_task)
        return {
            "correct": self.check_answer_correctness(answer, gt, raw_task),
            "predicted_answer": answer,
            "scorer_provenance": f"{self.dataset_name}.native_scorer.v1",
        }

    def score_answer(self, answer: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
        return self.score(answer, raw_task)

    def _display_id_for_task(self, raw_task: dict[str, Any], orig_idx: int) -> str | int | None:
        value = raw_task.get("id")
        return orig_idx if value is None else value

    def load_items(self, path: Path, *, registry_meta: dict[str, Any]) -> DatasetLoadResult:
        rows = _read_jsonl(path)
        dataset_revision = _derive_dataset_revision(
            dataset=self.dataset_name,
            rows=rows,
            registry_meta=registry_meta,
        )
        items: list[TaskItem] = []
        for orig_idx, raw_row in enumerate(rows):
            prompt_metadata = self.task_prompt_metadata(raw_row)
            try:
                _question, answer_key, _prepared = self.parse_question_answer(raw_row)
            except (KeyError, TypeError, ValueError) as exc:
                display_id = self._display_id_for_task(raw_row, orig_idx)
                raise ValueError(
                    f"{self.dataset_name} row {orig_idx} (display_id={display_id!r}) failed parse_question_answer: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            items.append(
                TaskItem(
                    item_uid=make_item_uid(
                        dataset=self.dataset_name,
                        raw_task=raw_row,
                        dataset_revision=dataset_revision,
                    ),
                    display_id=self._display_id_for_task(raw_row, orig_idx),
                    dataset=self.dataset_name,
                    raw_task=raw_row,
                    family=self.task_family(raw_row),
                    prompt_metadata=prompt_metadata,
                    answer_key=answer_key,
                    dataset_revision=dataset_revision,
                )
            )
        return DatasetLoadResult(
            items=items,
            dataset_revision=dataset_revision,
            registry_meta=dict(registry_meta),
            source_path=str(path),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _derive_dataset_revision(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    registry_meta: dict[str, Any],
) -> str | None:
    source_revision = registry_meta.get("source_dataset_revision")
    if source_revision is not None:
        text = str(source_revision).strip()
        if text:
            return text

    fingerprint = {
        "dataset": dataset,
        "source_dataset_id": registry_meta.get("source_dataset_id"),
        "source_dataset_config": registry_meta.get("source_dataset_config"),
        "source_dataset_split": registry_meta.get("source_dataset_split"),
        "rows": sorted(
            json.dumps(
                normalize_task_payload(dataset=dataset, raw_task=row),
                ensure_ascii=True,
                sort_keys=True,
            )
            for row in rows
        ),
    }
    digest = hashlib.sha1(json.dumps(fingerprint, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def build_standard_debate_message(
    *,
    intro: str,
    updates: list[str],
    outro: str,
    update_template: str = "\n\n One agent update: ```{answer}```",
) -> dict[str, str]:
    parts = [str(intro)]
    parts.extend(str(update_template).format(answer=answer) for answer in updates)
    parts.append(str(outro))
    return {"role": "user", "content": "".join(parts)}
