from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from debate_v_majority.datasets import get_dataset_adapter, get_registry_entry, list_dataset_adapters
from debate_v_majority.datasets.registry import DatasetRegistryEntry, ModuleDatasetAdapter


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_registry_entry_loads_yaml_metadata():
    gpqa = get_registry_entry("gpqa")
    hle = get_registry_entry("hle")
    assert gpqa.source_dataset_id == "Idavidrein/gpqa"
    assert gpqa.source_dataset_config == "gpqa_diamond"
    assert hle.source_dataset_id == "skylenage/HLE-Verified"
    assert hle.prompt_metadata_fields == (
        "source_variant",
        "source_dataset_config",
        "source_dataset_split",
        "source_dataset_revision",
        "answer_format_type",
        "domain_family",
    )


def test_list_dataset_adapters_includes_expected_datasets():
    names = [adapter.dataset_name for adapter in list_dataset_adapters()]
    assert {"aime25", "gpqa", "hle"}.issubset(set(names))


def test_adapter_load_items_produces_stable_task_items(tmp_path: Path):
    path = tmp_path / "gpqa.jsonl"
    _write_jsonl(
        path,
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "blue",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-reg-1",
            }
        ],
    )
    adapter = get_dataset_adapter("gpqa")
    result = adapter.load_items(path, registry_meta={"source_dataset_id": "Idavidrein/gpqa"})
    item = result.items[0]
    assert item.dataset == "gpqa"
    assert item.item_uid == "gpqa:gpqa-reg-1"
    assert item.display_id == "gpqa-reg-1"
    assert item.answer_key in {"A", "B", "C", "D"}
    assert result.registry_meta["source_dataset_id"] == "Idavidrein/gpqa"


def test_adapter_load_items_raises_on_malformed_row(tmp_path: Path):
    path = tmp_path / "gpqa_bad.jsonl"
    _write_jsonl(
        path,
        [
            {
                "Question": "Which option is correct?",
                "Incorrect Answer 1": "red",
                "Incorrect Answer 2": "green",
                "Incorrect Answer 3": "yellow",
                "id": "gpqa-bad-1",
            }
        ],
    )
    adapter = get_dataset_adapter("gpqa")
    with pytest.raises(ValueError, match="gpqa row 0 .*gpqa-bad-1.*failed parse_question_answer"):
        adapter.load_items(path, registry_meta={"source_dataset_id": "Idavidrein/gpqa"})


def test_adapter_score_raises_when_render_prompt_fails():
    adapter = get_dataset_adapter("aime25")
    with pytest.raises(KeyError, match="problem"):
        adapter.score("2", {"answer": "2"})


def test_dataset_adapter_does_not_mask_scorer_type_errors(monkeypatch: pytest.MonkeyPatch):
    def _broken_scorer(answer, gt, raw_task):
        del answer, gt, raw_task
        raise TypeError("bug inside scorer")

    fake_module = types.SimpleNamespace(check_answer_correctness=_broken_scorer)
    monkeypatch.setattr("debate_v_majority.datasets.registry.importlib.import_module", lambda _name: fake_module)

    adapter = ModuleDatasetAdapter(
        DatasetRegistryEntry(
            dataset="fake",
            module="fake.module",
            source_dataset_id=None,
            source_dataset_config=None,
            source_dataset_split=None,
        )
    )

    with pytest.raises(TypeError, match="bug inside scorer"):
        adapter.check_answer_correctness("A", "B", {"id": "fake"})


def test_hle_adapter_load_items_populates_family_and_prompt_metadata(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "hle-reg-1",
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "blue",
                "answer_type": "multipleChoice",
                "category": "Physics",
                "Verified_Classes": "Gold subset",
                "source_variant": "verified",
            }
        ],
    )
    adapter = get_dataset_adapter("hle")
    result = adapter.load_items(
        path,
        registry_meta={
            "source_dataset_id": "skylenage/HLE-Verified",
            "source_dataset_config": "default",
            "source_dataset_split": "train",
        },
    )
    item = result.items[0]
    assert item.item_uid == "hle:hle-reg-1"
    assert item.family == "physical_sciences"
    assert item.prompt_metadata["source_variant"] == "verified"
    assert item.prompt_metadata["answer_format_type"] == "multiple_choice"
    assert item.prompt_metadata["domain_family"] == "physical_sciences"


def test_gpqa_materialize_uses_pinned_registry_config(tmp_path: Path, monkeypatch):
    recorded: list[tuple[object, object, object]] = []

    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda dataset_id, config=None, split=None, **kwargs: recorded.append((dataset_id, config, split)) or []
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    adapter = get_dataset_adapter("gpqa")
    adapter.materialize(tmp_path / "gpqa.jsonl")

    assert recorded == [("Idavidrein/gpqa", "gpqa_diamond", "test")]


def test_gpqa_materialize_preserves_underlying_exception(tmp_path: Path, monkeypatch):
    def _boom(*args, **kwargs):
        raise ConnectionError("hub timeout")

    fake_datasets = types.SimpleNamespace(load_dataset=_boom)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    adapter = get_dataset_adapter("gpqa")
    with pytest.raises(ConnectionError, match="hub timeout"):
        adapter.materialize(tmp_path / "gpqa.jsonl")
