from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from debate_v_majority.cli.args import _build_arg_parser
from debate_v_majority.cli import engine_runtime as cli_engine_runtime
from debate_v_majority.cli import main_impl as cli_main_impl
from debate_v_majority.cli import subset as cli_subset
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.cli.main_impl import (
    FINAL_OUTPUT_SCHEMA_VERSION,
    SubsetItem,
    _augment_output_rows,
    _ensure_dataset_test_jsonl,
    _select_one_item,
    _write_or_validate_final_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_cli_has_no_duplicate_long_flags():
    parser = _build_arg_parser()
    flags = [
        opt
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    ]
    assert len(flags) == len(set(flags))


def test_make_dataset_subset_with_explicit_ids(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": f"Which option is correct for item {i}?",
            "Correct Answer": f"correct-{i}",
            "Incorrect Answer 1": f"wrong-{i}-1",
            "Incorrect Answer 2": f"wrong-{i}-2",
            "Incorrect Answer 3": f"wrong-{i}-3",
            "id": i,
        }
        for i in range(5)
    ]
    _write_jsonl(path, rows)
    items, meta = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=2,
        seed=123,
        ids=[0, 3],
        range_str=None,
    )
    assert [it.orig_id for it in items] == [0, 3]
    assert meta["subset_size"] == 2
    assert meta["dataset_config"] == "gpqa_diamond"
    assert meta["source_dataset_id"] == "Idavidrein/gpqa"


def test_make_dataset_subset_negative_id_maps_to_last_row(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": f"Which option is correct for item {i}?",
            "Correct Answer": f"correct-{i}",
            "Incorrect Answer 1": f"wrong-{i}-1",
            "Incorrect Answer 2": f"wrong-{i}-2",
            "Incorrect Answer 3": f"wrong-{i}-3",
            "id": i,
        }
        for i in range(4)
    ]
    _write_jsonl(path, rows)
    items, _ = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=1,
        seed=123,
        ids=[-1],
        range_str=None,
    )
    assert items[0].orig_id == -1
    assert items[0].raw_task["id"] == 3


def test_make_dataset_subset_out_of_range_id_raises(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": f"Which option is correct for item {i}?",
            "Correct Answer": f"correct-{i}",
            "Incorrect Answer 1": f"wrong-{i}-1",
            "Incorrect Answer 2": f"wrong-{i}-2",
            "Incorrect Answer 3": f"wrong-{i}-3",
            "id": i,
        }
        for i in range(2)
    ]
    _write_jsonl(path, rows)
    with pytest.raises(IndexError):
        _make_dataset_subset(
            dataset="gpqa",
            test_path=path,
            n=1,
            seed=123,
            ids=[99],
            range_str=None,
        )


def test_default_dataset_test_path_prefers_first_existing_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    p1 = tmp_path / "repo" / "data" / "gpqa" / "test.jsonl"
    p2 = tmp_path / "pkg" / "data" / "gpqa" / "test.jsonl"
    p3 = tmp_path / "legacy" / "data" / "gpqa" / "test.jsonl"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cli_main_impl,
        "_dataset_test_path_candidates",
        lambda dataset, source_file=None, hle_variant=None, dataset_local_mirror=None: [p1, p2, p3],
    )

    selected = cli_main_impl._default_dataset_test_path("gpqa")
    assert selected == p2


def test_dataset_test_path_candidates_include_repo_package_and_legacy(tmp_path: Path):
    source_file = tmp_path / "repo" / "src" / "debate_v_majority" / "cli" / "main_impl.py"
    cands = cli_main_impl._dataset_test_path_candidates("aime25", source_file=source_file)

    assert cands[0] == tmp_path / "repo" / "data" / "aime25" / "test.jsonl"
    assert cands[1] == tmp_path / "repo" / "src" / "debate_v_majority" / "data" / "aime25" / "test.jsonl"
    assert cands[2] == tmp_path / "repo" / "src" / "debate_v_majority" / "cli" / "data" / "aime25" / "test.jsonl"


def test_default_dataset_test_path_hle_uses_variant_and_local_mirror_root(tmp_path: Path):
    selected = cli_main_impl._default_dataset_test_path(
        "hle",
        hle_variant="verified_full",
        dataset_local_mirror=tmp_path / "mirror",
    )
    assert selected == tmp_path / "mirror" / "hle" / "verified_full.jsonl"


def test_default_dataset_test_path_hle_accepts_explicit_jsonl_mirror(tmp_path: Path):
    selected = cli_main_impl._default_dataset_test_path(
        "hle",
        hle_variant="verified",
        dataset_local_mirror=tmp_path / "custom_hle.jsonl",
    )
    assert selected == tmp_path / "custom_hle.jsonl"


def test_load_exclude_id_map_supports_text_and_json(tmp_path: Path):
    text_path = tmp_path / "exclude.txt"
    text_path.write_text("# comment\nhle-item-1\n42\n", encoding="utf-8")
    json_path = tmp_path / "exclude.json"
    json_path.write_text(json.dumps({"hle-item-2": "bad item", "7": None}), encoding="utf-8")

    text_map = cli_main_impl._load_exclude_id_map(text_path)
    json_map = cli_main_impl._load_exclude_id_map(json_path)

    assert text_map == {"hle-item-1": None, "42": None}
    assert json_map == {"hle-item-2": "bad item", "7": None}


def test_make_dataset_subset_excludes_hle_items_and_keeps_metadata(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    rows = [
        {
            "id": "hle-item-1",
            "question": "Which option is correct?\nA) red\nB) blue\nC) green",
            "answer": "blue",
            "answer_type": "multipleChoice",
            "category": "physics",
            "Verified_Classes": "Gold subset",
            "source_variant": "verified",
            "source_subset_label": "Gold subset",
        },
        {
            "id": "hle-item-2",
            "question": "Which option is correct?\nA) cat\nB) dog\nC) bird",
            "answer": "dog",
            "answer_type": "multipleChoice",
            "category": "computer science/ai",
            "Verified_Classes": "Gold subset",
            "source_variant": "verified",
            "source_subset_label": "Gold subset",
        },
    ]
    _write_jsonl(path, rows)
    exclude_path = tmp_path / "exclude.txt"
    exclude_path.write_text("hle-item-1\n", encoding="utf-8")

    items, meta = _make_dataset_subset(
        dataset="hle",
        test_path=path,
        n=2,
        seed=123,
        ids=None,
        range_str="all",
        hle_variant="verified",
        exclude_id_map={"hle-item-1": None},
        exclude_ids_path=exclude_path,
    )

    assert [item.item_display_id for item in items] == ["hle-item-2"]
    assert meta["dataset_variant"] == "verified"
    assert meta["available_after_exclusion"] == 1
    assert meta["excluded_count"] == 1
    assert meta["excluded_ids"] == ["hle-item-1"]
    assert meta["exclude_ids_path"] == str(exclude_path)
    assert items[0].dataset_meta["exclude_ids_path"] == str(exclude_path)
    assert items[0].dataset_meta["subset_size"] == 1


def test_make_dataset_subset_exclude_ids_prefers_canonical_numeric_id_over_orig_id(tmp_path: Path):
    path = tmp_path / "hle_numeric_ids.jsonl"
    rows = [
        {
            "id": "item-0",
            "question": "Which option is correct?\nA) red\nB) blue",
            "answer": "blue",
            "answer_type": "multipleChoice",
            "category": "physics",
            "Verified_Classes": "Gold subset",
            "source_variant": "verified",
        },
        {
            "id": 0,
            "question": "Which option is correct?\nA) cat\nB) dog",
            "answer": "dog",
            "answer_type": "multipleChoice",
            "category": "physics",
            "Verified_Classes": "Gold subset",
            "source_variant": "verified",
        },
    ]
    _write_jsonl(path, rows)

    items, meta = _make_dataset_subset(
        dataset="hle",
        test_path=path,
        n=2,
        seed=123,
        ids=None,
        range_str="all",
        hle_variant="verified",
        exclude_id_map={"0": None},
    )

    assert [item.item_display_id for item in items] == ["item-0"]
    assert meta["excluded_count"] == 1
    assert meta["excluded_ids"] == ["0"]


def test_make_dataset_subset_assigns_contiguous_subset_ids_after_late_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "Question": f"Which option is correct for item {i}?",
            "Correct Answer": f"correct-{i}",
            "Incorrect Answer 1": f"wrong-{i}-1",
            "Incorrect Answer 2": f"wrong-{i}-2",
            "Incorrect Answer 3": f"wrong-{i}-3",
            "id": i,
        }
        for i in range(3)
    ]
    _write_jsonl(path, rows)

    original_identifiers = cli_subset._canonical_item_identifiers
    late_exclusion_hits = {"count": 0}

    def _fake_canonical_item_identifiers(task_item):
        identifiers = original_identifiers(task_item)
        raw_id = str(getattr(task_item, "raw_task", {}).get("id"))
        if raw_id != "1":
            return identifiers
        late_exclusion_hits["count"] += 1
        if late_exclusion_hits["count"] == 1:
            return identifiers
        return ["exclude-me", *identifiers]

    monkeypatch.setattr(cli_subset, "_canonical_item_identifiers", _fake_canonical_item_identifiers)

    items, meta = _make_dataset_subset(
        dataset="gpqa",
        test_path=path,
        n=3,
        seed=123,
        ids=None,
        range_str="all",
        exclude_id_map={"exclude-me": "late exclusion"},
    )

    assert [item.orig_id for item in items] == [0, 2]
    assert [item.subset_id for item in items] == [0, 1]
    assert late_exclusion_hits["count"] >= 2
    assert meta["subset_size"] == 2


def test_ensure_dataset_test_jsonl_hle_materializes_filtered_variant_with_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_rows = [
        {
            "id": "gold-1",
            "question": "Q1\nA) x\nB) y",
            "answer": "B",
            "answer_type": "multipleChoice",
            "category": "Physics",
            "Verified_Classes": "Gold subset",
        },
        {
            "id": "rev-1",
            "question": "Q2",
            "answer": "42",
            "answer_type": "exactMatch",
            "category": "Math",
            "Verified_Classes": "Revision subset",
        },
    ]
    fake_module = type(
        "FakeDatasets",
        (),
        {
            "load_dataset": staticmethod(lambda dataset_id, split, revision: fake_rows),
        },
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    out_path = tmp_path / "verified.jsonl"
    _ensure_dataset_test_jsonl("hle", out_path, hle_variant="verified")

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "gold-1"
    assert rows[0]["source_variant"] == "verified"
    assert rows[0]["source_subset_label"] == "Gold subset"
    assert rows[0]["source_dataset_id"] == "skylenage/HLE-Verified"
    assert rows[0]["source_dataset_revision"] == "0bc83643672d4f68a5f89998617a639d85e7318b"


def test_select_one_item_matches_supported_identifiers():
    item = SubsetItem(
        subset_id=7,
        orig_id=22,
        item_uid="gpqa:item-22",
        dataset_revision="rev-1",
        item_display_id="item-22",
        raw_task={"id": "item-22"},
        dataset_meta={"subset_size": 1},
    )
    items = [item]
    assert _select_one_item(items, "gpqa:item-22") == items
    assert _select_one_item(items, "item-22") == items
    assert _select_one_item(items, "22") == items
    assert _select_one_item(items, "7") == items


def test_select_one_item_prefers_canonical_numeric_id_over_row_indexes():
    items = [
        SubsetItem(
            subset_id=0,
            orig_id=0,
            item_uid="gpqa:item-a",
            dataset_revision="rev-1",
            item_display_id="item-a",
            raw_task={"id": "item-a"},
            dataset_meta={"subset_size": 2},
        ),
        SubsetItem(
            subset_id=1,
            orig_id=9,
            item_uid="gpqa:item-b",
            dataset_revision="rev-1",
            item_display_id=0,
            raw_task={"id": 0},
            dataset_meta={"subset_size": 2},
        ),
    ]

    selected = _select_one_item(items, "0")
    assert selected == [items[1]]


def test_augment_output_rows_adds_phase7_logical_blocks():
    rows = [
        {
            "dataset": "gpqa",
            "item_uid": "gpqa:item-1",
            "dataset_revision": "rev-1",
            "item_display_id": "item-1",
            "subset_id": 0,
            "orig_id": 11,
            "question": "Which option is correct?",
            "raw_task": {"id": "item-1"},
            "final_answer": "A",
            "final_correct": 1,
            "round1_majority_answer": "B",
            "final_round_majority_answer": "A",
            "final_judge_answer": "A",
            "judge_summary": {"judge_id": "judge-1", "judge_family": "science"},
            "persona_summaries": [{"persona_id": "p1", "title": "Verifier"}],
            "agent_round_outputs": [[{"final_answer": "A"}]],
            "judge_trace": {"judge_raw_response": "\\boxed{A}"},
        }
    ]
    augmented = _augment_output_rows(
        rows,
        run_meta={
            "run_tag": "tag-1",
            "dataset": "gpqa",
            "dataset_meta": {"subset_size": 1},
            "output_schema_version": FINAL_OUTPUT_SCHEMA_VERSION,
            "emit_trace_level": "full",
            "final_manifest_path": None,
            "output_path": "out.jsonl",
        },
        mode="debate",
        use_personas=True,
        public_rationale_max_tokens=96,
        emit_trace_level="full",
    )
    row = augmented[0]
    assert row["run_meta"]["output_schema_version"] == FINAL_OUTPUT_SCHEMA_VERSION
    assert row["task"]["item_uid"] == "gpqa:item-1"
    assert row["strategy"]["mode"] == "debate"
    assert row["persona_meta"]["persona_summaries"][0]["persona_id"] == "p1"
    assert row["judge_meta"]["judge_summary"]["judge_id"] == "judge-1"
    assert row["results"]["judge_final_answer"] == "A"
    assert row["display"]["judge_final_answer"] == "A"
    assert row["trace"]["judge_trace"]["judge_raw_response"] == "\\boxed{A}"


def test_write_or_validate_final_manifest_rejects_mismatch(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_a = {"locked_config": {"dataset": "gpqa", "subset_size": 1}}
    manifest_b = {"locked_config": {"dataset": "gpqa", "subset_size": 2}}
    _write_or_validate_final_manifest(
        manifest_path=manifest_path,
        manifest=manifest_a,
        final_run=False,
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        _write_or_validate_final_manifest(
            manifest_path=manifest_path,
            manifest=manifest_b,
            final_run=True,
        )


def test_main_single_supports_one_output_and_final_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _FakeEngine:
        model_name = "gemini-3-flash"

        def shutdown(self):
            return None

    selected_items: list[str] = []

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        items = [
            SimpleNamespace(
                subset_id=0,
                orig_id=100,
                item_uid="aime25:item-1",
                item_display_id="item-1",
                raw_task={"problem": "What is 1+1?", "answer": "2"},
            ),
            SimpleNamespace(
                subset_id=1,
                orig_id=200,
                item_uid="aime25:item-2",
                item_display_id="item-2",
                raw_task={"problem": "What is 2+2?", "answer": "4"},
            ),
        ]
        meta = {
            "subset_size": 2,
            "seed": 7,
            "dataset_revision": "aime25.test.rev",
            "dataset_variant": None,
            "source_path": str(tmp_path / "aime25.jsonl"),
        }
        return items, meta

    def _fake_run_sampled(**kwargs):
        items = kwargs["items"]
        selected_items[:] = [item.item_uid for item in items]
        assert len(items) == 1
        item = items[0]
        return [
            {
                "schema_version": "phase2.single.v1",
                "mode": "single",
                "dataset": "aime25",
                "item_uid": item.item_uid,
                "dataset_revision": "aime25.test.rev",
                "item_display_id": item.item_display_id,
                "subset_id": item.subset_id,
                "orig_id": item.orig_id,
                "question": "What is 2+2?",
                "raw_task": {"problem": "What is 2+2?", "answer": "4"},
                "final_answer": "4",
                "final_correct": 1,
                "token_usage_summary": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "n_calls": 1,
                },
                "sample_call_metadata": [
                    {
                        "token_counts": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    }
                ],
            }
        ]

    monkeypatch.setattr(cli_engine_runtime, "create_inference_engine", lambda **kwargs: _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(
        cli_main_impl,
        "build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=1.0, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "run_sampled", _fake_run_sampled)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260309_120000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "single",
            "--model_name",
            "gemini-3-flash",
            "--provider",
            "gemini",
            "--one",
            "aime25:item-2",
            "--output",
            str(tmp_path / "single-output.jsonl"),
            "--final_manifest",
            str(tmp_path / "final-manifest.json"),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main_impl.main()

    assert selected_items == ["aime25:item-2"]
    rows = [json.loads(line) for line in (tmp_path / "single-output.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_meta"]["output_schema_version"] == "phase7.logical.v1"
    assert row["run_meta"]["output_path"] == str(tmp_path / "single-output.jsonl")
    assert row["run_meta"]["final_manifest_path"] == str(tmp_path / "final-manifest.json")
    assert row["task"]["item_uid"] == "aime25:item-2"
    assert row["strategy"]["mode"] == "single"
    assert row["results"]["final_answer"] == "4"
    assert row["display"]["question_short"] == "What is 2+2?"
    assert row["trace"]["token_usage_summary"]["total_tokens"] == 15
    assert "persona_meta" in row and row["persona_meta"] is None
    assert "judge_meta" in row and row["judge_meta"] is None

    manifest = json.loads((tmp_path / "final-manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "phase7.final_manifest.v1"
    assert manifest["locked_config"]["item_uids"] == ["aime25:item-2"]
    assert manifest["locked_config"]["output_schema_version"] == "phase7.logical.v1"


def test_write_or_validate_final_manifest_rejects_mismatch_for_final_run(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    cli_main_impl._write_or_validate_final_manifest(
        manifest_path=manifest_path,
        manifest={"manifest_version": "phase7.final_manifest.v1", "locked_config": {"dataset": "aime25", "model": "gemini-3-flash"}},
        final_run=False,
    )

    with pytest.raises(ValueError, match="manifest mismatch"):
        cli_main_impl._write_or_validate_final_manifest(
            manifest_path=manifest_path,
            manifest={"manifest_version": "phase7.final_manifest.v1", "locked_config": {"dataset": "aime25", "model": "wrong-model"}},
            final_run=True,
        )


def test_augment_output_rows_adds_phase7_logical_blocks_for_debate_rows():
    row = {
        "dataset": "gpqa",
        "item_uid": "gpqa:item-1",
        "dataset_revision": "gpqa.rev",
        "item_display_id": "item-1",
        "subset_id": 0,
        "orig_id": 11,
        "question": "Which option is correct?",
        "raw_task": {"question": "Which option is correct?"},
        "final_answer": "A",
        "final_correct": 1,
        "round1_majority_result": {"answer": "B", "correct": 0},
        "final_round_majority_result": {"answer": "A", "correct": 1},
        "judge_result": {"answer": "A", "correct": 1},
        "final_round_majority_answer": "A",
        "final_judge_answer": "A",
        "n_agents": 3,
        "n_rounds": 2,
        "persona_meta": {"artifact_path": "artifact.json", "persona_ids": ["p1", "p2", "p3"]},
        "judge_meta": {"judge_summary": {"judge_family": "science_selector"}},
        "persona_summaries": [{"persona_id": "p1", "title": "Verifier"}],
        "judge_summary": {"judge_family": "science_selector"},
        "debater_round_token_usage": [{"n_calls": 3, "total_tokens": 100}],
        "judge_round_token_usage": {"aggregate": {"n_calls": 1, "total_tokens": 20}},
        "token_usage_summary": {"all": {"n_calls": 4, "total_tokens": 120}},
        "agent_round_outputs": [[{"final_answer": "B"}], [{"final_answer": "A"}]],
        "judge_trace": {"judge_raw_response": "\\boxed{A}"},
        "agent_responses": [["r1"], ["r2"]],
    }
    rows = cli_main_impl._augment_output_rows(
        [row],
        run_meta={"run_tag": "run-1", "output_schema_version": "phase7.logical.v1"},
        mode="debate",
        use_personas=True,
        public_rationale_max_tokens=96,
        emit_trace_level="full",
    )

    assert len(rows) == 1
    updated = rows[0]
    assert updated["task"]["item_uid"] == "gpqa:item-1"
    assert updated["strategy"]["mode"] == "debate"
    assert updated["persona_meta"]["artifact_path"] == "artifact.json"
    assert updated["judge_meta"]["judge_summary"]["judge_family"] == "science_selector"
    assert updated["results"]["judge_final_answer"] == "A"
    assert updated["display"]["judge_final_answer"] == "A"
    assert updated["trace"]["engine_calls"]["judge_round_token_usage"]["aggregate"]["n_calls"] == 1
    assert updated["trace"]["judge_trace"]["judge_raw_response"] == "\\boxed{A}"
