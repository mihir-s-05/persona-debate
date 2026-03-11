from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import PersonaArtifact


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_value(value[k]) for k in sorted(value)}
    return value


def normalize_task_payload(*, dataset: str, raw_task: dict[str, Any]) -> dict[str, Any]:
    if dataset == "gpqa":
        question_keys = [
            "question",
            "problem",
            "query",
            "Question",
            "Extra Revised Question",
            "Pre-Revision Question",
        ]
        correct_keys = [
            "correct_answer",
            "answer",
            "solution",
            "Correct Answer",
            "Extra Revised Correct Answer",
            "Pre-Revision Correct Answer",
        ]
        incorrect_keys = [
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "Extra Revised Incorrect Answer 1",
            "Extra Revised Incorrect Answer 2",
            "Extra Revised Incorrect Answer 3",
            "Pre-Revision Incorrect Answer 1",
            "Pre-Revision Incorrect Answer 2",
            "Pre-Revision Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ]
        question = next((raw_task[k] for k in question_keys if raw_task.get(k) is not None), None)
        correct = next((raw_task[k] for k in correct_keys if raw_task.get(k) is not None), None)
        incorrect = sorted(
            " ".join(str(raw_task[k]).split())
            for k in incorrect_keys
            if raw_task.get(k) is not None
        )
        return {
            "question": _normalize_value(question),
            "correct_answer": _normalize_value(correct),
            "incorrect_answers": incorrect,
        }
    normalized = _normalize_value(raw_task)
    if dataset == "aime25":
        return normalized
    return normalized


def dataset_revision_from_path(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def trusted_native_item_id(*, dataset: str, raw_task: dict[str, Any]) -> str | None:
    candidate_keys = ["id", "uid", "uuid", "question_id", "record_id"]
    if dataset not in {"gpqa", "hle"}:
        return None
    for key in candidate_keys:
        value = raw_task.get(key)
        if value is None:
            continue
        s = str(value).strip()
        if s:
            return s
    return None


def make_item_uid(
    *,
    dataset: str,
    raw_task: dict[str, Any],
    dataset_revision: str | None,
) -> str:
    native = trusted_native_item_id(dataset=dataset, raw_task=raw_task)
    if native:
        return f"{dataset}:{native}"
    normalized = normalize_task_payload(dataset=dataset, raw_task=raw_task)
    payload = {
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "task": normalized,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:h:{digest}"


def legacy_artifact_path(*, artifacts_dir: Path, dataset: str, item_uid: str) -> Path:
    safe_name = item_uid.replace(":", "__")
    return artifacts_dir / dataset / f"{safe_name}.json"


def artifact_config_key(*, dataset_revision: str | None, generation_settings: dict[str, Any] | None) -> str | None:
    normalized_settings = _normalize_value(generation_settings or {})
    if not normalized_settings:
        return None
    payload = {
        "dataset_revision": dataset_revision,
        "generation_settings": normalized_settings,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]


def artifact_path(
    *,
    artifacts_dir: Path,
    dataset: str,
    item_uid: str,
    dataset_revision: str | None = None,
    generation_settings: dict[str, Any] | None = None,
) -> Path:
    safe_name = item_uid.replace(":", "__")
    config_key = artifact_config_key(
        dataset_revision=dataset_revision,
        generation_settings=generation_settings,
    )
    if not config_key:
        return legacy_artifact_path(artifacts_dir=artifacts_dir, dataset=dataset, item_uid=item_uid)
    return artifacts_dir / dataset / f"{safe_name}--cfg-{config_key}.json"


def save_artifact(*, artifacts_dir: Path, artifact: PersonaArtifact) -> Path:
    path = artifact_path(
        artifacts_dir=artifacts_dir,
        dataset=artifact.dataset,
        item_uid=artifact.item_uid,
        dataset_revision=artifact.dataset_revision,
        generation_settings=artifact.generation_settings,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_artifact(*, path: Path) -> PersonaArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PersonaArtifact.from_dict(data)
