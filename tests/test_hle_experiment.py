from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from debate_v_majority.cli.args import _build_arg_parser
from debate_v_majority.cli import main_impl as cli_main_impl
from debate_v_majority.cli.stage_state import make_stage_entry


ARM_NAMES = ("single", "debate_plain", "persona_debate")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fake_hle_items() -> tuple[list[SimpleNamespace], dict[str, Any]]:
    items = [
        SimpleNamespace(
            subset_id=0,
            orig_id=10,
            item_uid="hle:shared-1",
            item_display_id="shared-1",
            dataset_revision="hle.test.rev",
            raw_task={
                "id": "shared-1",
                "question": "Which option is correct?\nA) red\nB) blue\nC) green",
                "answer": "blue",
            },
        ),
        SimpleNamespace(
            subset_id=1,
            orig_id=11,
            item_uid="hle:shared-2",
            item_display_id="shared-2",
            dataset_revision="hle.test.rev",
            raw_task={
                "id": "shared-2",
                "question": "Which option is correct?\nA) cat\nB) dog\nC) bird",
                "answer": "dog",
            },
        ),
    ]
    meta = {
        "subset_size": len(items),
        "seed": 7,
        "dataset_revision": "hle.test.rev",
        "dataset_variant": "verified",
        "source_path": "C:/fake/hle/verified.jsonl",
    }
    return items, meta


def _single_row(item: SimpleNamespace) -> dict[str, Any]:
    return {
        "schema_version": "phase2.single.v1",
        "mode": "single",
        "dataset": "hle",
        "item_uid": item.item_uid,
        "dataset_revision": item.dataset_revision,
        "item_display_id": item.item_display_id,
        "subset_id": item.subset_id,
        "orig_id": item.orig_id,
        "question": item.raw_task["question"],
        "answer": "B",
        "raw_task": dict(item.raw_task),
        "source_variant": "verified",
        "source_subset_label": "Gold subset",
        "canonical_item_id": item.raw_task["id"],
        "answer_format_type": "multiple_choice",
        "domain_family": "test_family",
        "final_answer": "B",
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


def _debate_row(item: SimpleNamespace, *, use_personas: bool) -> dict[str, Any]:
    persona_summaries = (
        [
            {"persona_id": "p1", "title": "Verifier", "short_rule": "Check details."},
            {"persona_id": "p2", "title": "Challenger", "short_rule": "Probe weaknesses."},
        ]
        if use_personas
        else []
    )
    return {
        "schema_version": "phase4.hle.v1",
        "mode": "debate",
        "dataset": "hle",
        "item_uid": item.item_uid,
        "dataset_revision": item.dataset_revision,
        "item_display_id": item.item_display_id,
        "subset_id": item.subset_id,
        "orig_id": item.orig_id,
        "question": item.raw_task["question"],
        "answer": "B",
        "raw_task": dict(item.raw_task),
        "source_variant": "verified",
        "source_subset_label": "Gold subset",
        "canonical_item_id": item.raw_task["id"],
        "answer_format_type": "multiple_choice",
        "domain_family": "test_family",
        "persona_summaries": persona_summaries,
        "final_answer": "B",
        "final_correct": 1,
        "round1_majority_answer": "B",
        "round1_majority_correct": 1,
        "final_round_majority_answer": "B",
        "final_round_majority_correct": 1,
        "final_judge_answer": "B",
        "final_judge_correct": 1,
        "agent_round_outputs": [
            [
                {
                    "private_raw_response": "Agent one reasoning.\nConfidence: 0.80\n\\boxed{B}",
                    "public_rationale": "agent one rationale",
                    "final_answer": "B",
                    "confidence": 0.8,
                    "parse_success": True,
                    "extractor_trace": {
                        "extractor_provenance": "hle_verified.extractor.v1",
                        "answer_format_type": "multiple_choice",
                    },
                    "scoring_result": {"correct": 1},
                }
            ],
            [
                {
                    "private_raw_response": "Agent two reasoning.\nConfidence: 0.72\n\\boxed{B}",
                    "public_rationale": "agent two rationale",
                    "final_answer": "B",
                    "confidence": 0.72,
                    "parse_success": True,
                    "extractor_trace": {
                        "extractor_provenance": "hle_verified.extractor.v1",
                        "answer_format_type": "multiple_choice",
                    },
                    "scoring_result": {"correct": 1},
                }
            ],
        ],
        "judge_meta": {
            "judge_trace_mode": "visible_plus_thought_summary",
            "judge_persona_mode": "task_family_generated" if use_personas else "none",
        },
        "judge_trace": {
            "judge_raw_response": "Judge picks \\boxed{B}",
            "judge_extractor_trace": {
                "normalized_answer": "B",
                "parse_success": True,
                "extractor_provenance": "hle_verified.extractor.v1",
                "answer_format_type": "multiple_choice",
            },
            "judge_scoring_result": {"correct": 1},
        },
    }


def _find_experiment_root(base_dir: Path) -> Path:
    candidates = [
        path
        for path in [base_dir, *base_dir.rglob("*")]
        if path.is_dir() and all((path / arm).is_dir() for arm in ARM_NAMES)
    ]
    assert candidates, f"could not find experiment root under {base_dir}"
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _arm_jsonl_files(arm_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in arm_dir.rglob("*.jsonl")
        if "traces" not in {part.lower() for part in path.parts}
    )


def _arm_rows(arm_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _arm_jsonl_files(arm_dir):
        rows.extend(_read_jsonl(path))
    return rows


def _trace_files(arm_dir: Path) -> list[Path]:
    traces_dir = arm_dir / "traces"
    return sorted(
        path
        for path in traces_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )


def _all_manifest_keys(node: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(str(key))
            keys.update(_all_manifest_keys(value))
    elif isinstance(node, list):
        for value in node:
            keys.update(_all_manifest_keys(value))
    return keys


def _find_shared_manifest(root_dir: Path, expected_item_uids: list[str]) -> dict[str, Any]:
    for path in sorted(root_dir.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        serialized = json.dumps(manifest, sort_keys=True)
        if all(arm in serialized for arm in ARM_NAMES) and all(item_uid in serialized for item_uid in expected_item_uids):
            return manifest
    raise AssertionError(f"could not find shared manifest in {root_dir}")


def test_parser_accepts_hle_experiment_flag():
    parser = _build_arg_parser()
    args = parser.parse_args(["--dataset", "hle", "--hle_experiment", "--hle_modality", "text_only"])
    assert args.dataset == "hle"
    assert args.hle_experiment is True
    assert args.hle_modality == "text_only"


def test_main_hle_experiment_writes_arm_outputs_shared_manifest_and_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    items, meta = _fake_hle_items()
    expected_item_uids = [item.item_uid for item in items]
    sample_calls: list[dict[str, Any]] = []
    debate_calls: list[dict[str, Any]] = []
    persona_generation_calls: list[dict[str, Any]] = []

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self) -> None:
            return None

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return items, dict(meta)

    def _fake_role_engine(**kwargs):
        del kwargs
        return _FakeEngine()

    def _fake_run_sampled(**kwargs):
        sample_calls.append(
            {
                "mode_label": kwargs.get("mode_label"),
                "item_uids": [item.item_uid for item in kwargs["items"]],
            }
        )
        return [_single_row(item) for item in kwargs["items"]]

    def _fake_run_debate(**kwargs):
        debate_calls.append(
            {
                "use_personas": bool(kwargs.get("use_personas")),
                "item_uids": [item.item_uid for item in kwargs["items"]],
                "persona_artifacts_by_item": kwargs.get("persona_artifacts_by_item"),
            }
        )
        return {
            0: [
                _debate_row(item, use_personas=bool(kwargs.get("use_personas")))
                for item in kwargs["items"]
            ]
        }

    def _fake_run_persona_generation(**kwargs):
        persona_generation_calls.append(
            {"item_uids": [item.item_uid for item in kwargs["items"]]}
        )
        return [
            {
                "schema_version": "phase0.v1",
                "mode": "personas",
                "dataset": "hle",
                "item_uid": item.item_uid,
                "dataset_revision": item.dataset_revision,
                "item_display_id": item.item_display_id,
                "artifact_version": "test-artifact.v1",
                "persona_seed": 7,
                "generator_model": None,
                "judge_generator_model": None,
                "axes": {"mode": "hybrid", "axes": []},
                "sampled_points": [],
                "descriptors": [],
                "cards": [
                    {"persona_id": "p1", "title": "Verifier", "short_rule": "Check details."},
                    {"persona_id": "p2", "title": "Challenger", "short_rule": "Probe weaknesses."},
                ],
                "judge_card": {"judge_id": "judge-1", "judge_family": "test_family"},
                "prompt_versions": {},
                "created_at": "2026-03-13T00:00:00Z",
                "validator_metadata": {},
                "artifact_path": str(tmp_path / "persona_artifacts" / f"{item.item_display_id}.json"),
            }
            for item in kwargs["items"]
        ]

    def _fake_persona_artifacts_from_rows(rows):
        return {str(row["item_uid"]): {"artifact_path": row.get("artifact_path")} for row in rows}

    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(cli_main_impl, "build_sampling_config", lambda model_name: SimpleNamespace(max_tokens=128))
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "create_inference_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_reuse_or_create_role_engine", lambda **kwargs: kwargs.get("existing_engine") or _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "run_sampled", _fake_run_sampled)
    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(cli_main_impl, "run_persona_generation", _fake_run_persona_generation)
    monkeypatch.setattr(cli_main_impl, "run_persona_generation_staged", _fake_run_persona_generation)
    monkeypatch.setattr(cli_main_impl, "persona_artifacts_from_rows", _fake_persona_artifacts_from_rows)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260313_120000")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )

    cli_main_impl.main()
    captured = capfd.readouterr()

    experiment_root = next(tmp_path.glob("hle_experiment_*"))
    assert len(sample_calls) == 1
    assert sample_calls[0]["item_uids"] == expected_item_uids
    assert sample_calls[0]["mode_label"] in {"single", "single_baseline"}
    assert (
        persona_generation_calls
        or any(call.get("persona_artifacts_by_item") for call in debate_calls if call["use_personas"])
    )
    assert {call["use_personas"] for call in debate_calls} == {False, True}
    assert all(call["item_uids"] == expected_item_uids for call in debate_calls)

    arm_item_uids: dict[str, list[str]] = {}
    for arm_name in ARM_NAMES:
        arm_dir = experiment_root / arm_name
        assert arm_dir.is_dir(), arm_dir

        jsonl_files = _arm_jsonl_files(arm_dir)
        assert jsonl_files, f"no JSONL outputs written for {arm_name}"

        rows = _arm_rows(arm_dir)
        assert rows, f"no rows found for {arm_name}"
        arm_item_uids[arm_name] = sorted({str(row["item_uid"]) for row in rows})

        trace_files = _trace_files(arm_dir)
        assert len(trace_files) >= len(expected_item_uids), f"missing readable traces for {arm_name}"
        assert any("Mode:" in path.read_text(encoding="utf-8") for path in trace_files)

    assert arm_item_uids["single"] == expected_item_uids
    assert arm_item_uids["debate_plain"] == expected_item_uids
    assert arm_item_uids["persona_debate"] == expected_item_uids

    debate_plain_rows = _arm_rows(experiment_root / "debate_plain")
    assert debate_plain_rows
    assert all((row.get("strategy") or {}).get("use_personas") is False for row in debate_plain_rows)
    assert all(row.get("persona_meta") is None for row in debate_plain_rows)

    persona_debate_rows = _arm_rows(experiment_root / "persona_debate")
    assert persona_debate_rows
    assert all((row.get("strategy") or {}).get("use_personas") is True for row in persona_debate_rows)

    manifest = _find_shared_manifest(experiment_root, expected_item_uids)
    manifest_keys = {key.lower() for key in _all_manifest_keys(manifest)}
    assert isinstance(manifest, dict)
    assert any("arm" in key for key in manifest_keys)
    assert any("item_uid" in key for key in manifest_keys)
    manifest_text = json.dumps(manifest, sort_keys=True)
    assert '"hle"' in manifest_text
    for arm_name in ARM_NAMES:
        assert arm_name in manifest_text
    for item_uid in expected_item_uids:
        assert item_uid in manifest_text
    assert "[cost] single=" in captured.out
    assert "[cost] debate_plain=" in captured.out
    assert "[cost] persona_generation=" in captured.out
    assert "[cost] persona_debate=" in captured.out


def test_main_hle_experiment_resume_skips_completed_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    items, meta = _fake_hle_items()
    sample_calls: list[list[str]] = []
    debate_calls: list[bool] = []
    persona_generation_calls: list[list[str]] = []
    stage_state_file = tmp_path / "experiment_state.jsonl"

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self) -> None:
            return None

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return items, dict(meta)

    def _fake_role_engine(**kwargs):
        del kwargs
        return _FakeEngine()

    def _fake_run_sampled(**kwargs):
        sample_calls.append([item.item_uid for item in kwargs["items"]])
        return [_single_row(item) for item in kwargs["items"]]

    def _fake_run_debate(**kwargs):
        debate_calls.append(bool(kwargs.get("use_personas")))
        return {0: [_debate_row(item, use_personas=bool(kwargs.get("use_personas"))) for item in kwargs["items"]]}

    def _fake_run_persona_generation(**kwargs):
        persona_generation_calls.append([item.item_uid for item in kwargs["items"]])
        return [
            {
                "schema_version": "phase0.v1",
                "mode": "personas",
                "dataset": "hle",
                "item_uid": item.item_uid,
                "dataset_revision": item.dataset_revision,
                "item_display_id": item.item_display_id,
                "artifact_version": "test-artifact.v1",
                "persona_seed": 7,
                "generator_model": None,
                "judge_generator_model": None,
                "axes": {"mode": "hybrid", "axes": []},
                "sampled_points": [],
                "descriptors": [],
                "cards": [
                    {"persona_id": "p1", "title": "Verifier", "short_rule": "Check details."},
                    {"persona_id": "p2", "title": "Challenger", "short_rule": "Probe weaknesses."},
                ],
                "judge_card": {"judge_id": "judge-1", "judge_family": "test_family"},
                "prompt_versions": {},
                "created_at": "2026-03-13T00:00:00Z",
                "validator_metadata": {},
                "artifact_path": str(tmp_path / "persona_artifacts" / f"{item.item_display_id}.json"),
            }
            for item in kwargs["items"]
        ]

    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(cli_main_impl, "build_sampling_config", lambda model_name: SimpleNamespace(max_tokens=128))
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_reuse_or_create_role_engine", lambda **kwargs: kwargs.get("existing_engine") or _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "run_sampled", _fake_run_sampled)
    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(cli_main_impl, "run_persona_generation", _fake_run_persona_generation)
    monkeypatch.setattr(cli_main_impl, "persona_artifacts_from_rows", lambda rows: {str(row["item_uid"]): {"artifact_path": row.get("artifact_path")} for row in rows})
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260313_121500")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--hle_experiment_stop_after",
            "single",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )
    cli_main_impl.main()
    first_run = capfd.readouterr()

    experiment_root = next(tmp_path.glob("hle_experiment_*"))
    assert (experiment_root / "single").is_dir()
    assert not (experiment_root / "debate_plain").exists()
    assert not (experiment_root / "persona_debate").exists()
    assert len(sample_calls) == 1
    assert not debate_calls
    assert "[cost] single=" in first_run.out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )
    cli_main_impl.main()
    second_run = capfd.readouterr()

    assert len(sample_calls) == 1
    assert debate_calls == [False, True]
    assert len(persona_generation_calls) == 1
    assert (experiment_root / "debate_plain").is_dir()
    assert (experiment_root / "persona_debate").is_dir()
    assert "[cost] debate_plain=" in second_run.out
    assert "[cost] persona_generation=" in second_run.out
    assert "[cost] persona_debate=" in second_run.out


def test_main_hle_experiment_resume_rejects_changed_persona_and_judge_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    items, meta = _fake_hle_items()
    stage_state_file = tmp_path / "experiment_state.jsonl"

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self) -> None:
            return None

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return items, dict(meta)

    def _fake_role_engine(**kwargs):
        del kwargs
        return _FakeEngine()

    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(cli_main_impl, "build_sampling_config", lambda model_name: SimpleNamespace(max_tokens=128))
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_reuse_or_create_role_engine", lambda **kwargs: kwargs.get("existing_engine") or _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "run_sampled", lambda **kwargs: [_single_row(item) for item in kwargs["items"]])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260313_124500")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--hle_experiment_stop_after",
            "single",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--persona_seed",
            "7",
            "--judge_trace_mode",
            "visible_plus_thought_summary",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )
    cli_main_impl.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--persona_seed",
            "11",
            "--judge_trace_mode",
            "assistant_transcript",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )

    with pytest.raises(ValueError, match="HLE experiment state settings mismatch"):
        cli_main_impl.main()


def test_main_hle_experiment_interactive_staging_prompts_through_internal_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    items, meta = _fake_hle_items()
    persona_stage_calls: list[str] = []
    debate_stop_calls: list[tuple[bool, str | None, str | None]] = []
    arm_prompt_calls: list[str] = []

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self) -> None:
            return None

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return items, dict(meta)

    def _fake_role_engine(**kwargs):
        del kwargs
        return _FakeEngine()

    def _fake_run_sampled(**kwargs):
        return [_single_row(item) for item in kwargs["items"]]

    class _FakeDebateStopped(Exception):
        pass

    def _fake_run_debate(**kwargs):
        stop_after = kwargs.get("debate_stop_after")
        use_personas = bool(kwargs.get("use_personas"))
        state_file = kwargs.get("stage_state_file")
        completed_stage = stop_after or "round_0_judge"
        debate_stop_calls.append((use_personas, None if stop_after is None else str(stop_after), None if state_file is None else str(state_file)))
        if state_file is not None:
            entry = make_stage_entry(
                stage_type="debate",
                completed_stage=str(completed_stage),
                dataset="hle",
                items=[item.__dict__ for item in kwargs["items"]],
                debate_data={
                    "n_agents": kwargs["n_agents"],
                    "n_rounds": kwargs["n_rounds"],
                    "judge_rounds": kwargs["judge_rounds"],
                },
            )
            from debate_v_majority.cli.stage_state import append_stage_entry

            append_stage_entry(Path(state_file), entry)
        if stop_after == "round_0":
            raise _FakeDebateStopped()
        return {0: [_debate_row(item, use_personas=use_personas) for item in kwargs["items"]]}

    def _fake_run_persona_generation_staged(**kwargs):
        stage = str(kwargs["persona_stage"])
        persona_stage_calls.append(stage)
        entry = make_stage_entry(
            stage_type="persona",
            completed_stage=stage,
            dataset="hle",
            items=[item.__dict__ for item in kwargs["items"]],
            persona_data={
                item.item_uid: {
                    "artifact": {
                        "artifact_version": "test-artifact.v1",
                        "dataset": "hle",
                        "item_uid": item.item_uid,
                        "dataset_revision": item.dataset_revision,
                        "item_display_id": item.item_display_id,
                        "persona_seed": 7,
                        "generator_model": None,
                        "judge_generator_model": None,
                        "axes": {"mode": "hybrid", "axes": []},
                        "sampled_points": [],
                        "descriptors": [],
                        "cards": [
                            {"persona_id": "p1", "title": "Verifier", "short_rule": "Check details."},
                            {"persona_id": "p2", "title": "Challenger", "short_rule": "Probe weaknesses."},
                        ],
                        "judge_card": {"judge_id": "judge-1", "judge_family": "test_family"},
                        "prompt_versions": {},
                        "created_at": "2026-03-13T00:00:00Z",
                        "generation_settings": {},
                        "validator_metadata": {},
                    }
                }
                for item in kwargs["items"]
            },
        )
        from debate_v_majority.cli.stage_state import append_stage_entry

        append_stage_entry(Path(kwargs["stage_state_file"]), entry)
        return entry

    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(cli_main_impl, "build_sampling_config", lambda model_name: SimpleNamespace(max_tokens=128))
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_reuse_or_create_role_engine", lambda **kwargs: kwargs.get("existing_engine") or _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "run_sampled", _fake_run_sampled)
    monkeypatch.setattr(cli_main_impl, "run_persona_generation_staged", _fake_run_persona_generation_staged)
    monkeypatch.setattr(
        cli_main_impl,
        "persona_rows_from_stage_entry",
        lambda entry: [
            {
                "schema_version": "phase0.v1",
                "mode": "personas",
                "dataset": "hle",
                "item_uid": item.item_uid,
                "dataset_revision": item.dataset_revision,
                "item_display_id": item.item_display_id,
                "artifact_version": "test-artifact.v1",
                "persona_seed": 7,
                "generator_model": None,
                "judge_generator_model": None,
                "axes": {"mode": "hybrid", "axes": []},
                "sampled_points": [],
                "descriptors": [],
                "cards": [
                    {
                        "persona_id": "p1",
                        "title": "Verifier",
                        "core_reasoning_strategy": "Check details.",
                        "priorities": [],
                        "distrusts": [],
                        "decomposition_style": "structured",
                        "revision_policy": "revise on evidence",
                        "confidence_policy": "state confidence",
                        "failure_mode_to_avoid": "overconfidence",
                        "system_prompt": "Verifier prompt",
                        "card_version": "test-card.v1",
                    },
                    {
                        "persona_id": "p2",
                        "title": "Challenger",
                        "core_reasoning_strategy": "Probe weaknesses.",
                        "priorities": [],
                        "distrusts": [],
                        "decomposition_style": "adversarial",
                        "revision_policy": "revise on evidence",
                        "confidence_policy": "state confidence",
                        "failure_mode_to_avoid": "anchoring",
                        "system_prompt": "Challenger prompt",
                        "card_version": "test-card.v1",
                    },
                ],
                "judge_card": {"judge_id": "judge-1", "judge_family": "test_family"},
                "prompt_versions": {},
                "created_at": "2026-03-13T00:00:00Z",
                "validator_metadata": {},
                "artifact_path": str(tmp_path / "persona_artifacts" / f"{item.item_display_id}.json"),
            }
            for item in items
        ],
    )
    monkeypatch.setattr(
        cli_main_impl,
        "persona_artifacts_from_rows",
        lambda rows: {str(row["item_uid"]): {"artifact_path": row.get("artifact_path")} for row in rows},
    )
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_persona_stage", lambda **kwargs: True)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_to_debate", lambda **kwargs: True)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_debate_stage", lambda **kwargs: True)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_experiment_arm", lambda **kwargs: arm_prompt_calls.append(str(kwargs["next_arm"])) or True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260313_123000")

    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setitem(sys.modules, "debate_v_majority.cli.debate_runner", SimpleNamespace(_DebateStopped=_FakeDebateStopped))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--stage_state_file",
            str(tmp_path / "experiment_state.jsonl"),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )

    cli_main_impl.main()
    captured = capfd.readouterr()

    assert persona_stage_calls == ["axes", "descriptors", "cards", "judge_card"]
    assert arm_prompt_calls == ["debate_plain", "persona_debate"]
    assert [call[:2] for call in debate_stop_calls] == [
        (False, "round_0"),
        (False, "round_0_judge"),
        (True, "round_0"),
        (True, "round_0_judge"),
    ]
    assert "[cost] single=" in captured.out
    assert "[cost] debate_plain=" in captured.out
    assert "[cost] persona_generation=" in captured.out
    assert "[cost] persona_debate=" in captured.out


def test_main_hle_experiment_noninteractive_resume_reuses_mid_arm_stage_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    items, meta = _fake_hle_items()
    stage_state_file = tmp_path / "experiment_state.jsonl"
    sample_calls: list[list[str]] = []
    debate_calls: list[tuple[str | None, str | None]] = []

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self) -> None:
            return None

    class _FakeDebateStopped(Exception):
        pass

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return items, dict(meta)

    def _fake_role_engine(**kwargs):
        del kwargs
        return _FakeEngine()

    def _fake_run_sampled(**kwargs):
        sample_calls.append([item.item_uid for item in kwargs["items"]])
        return [_single_row(item) for item in kwargs["items"]]

    def _fake_run_debate(**kwargs):
        stop_after = kwargs.get("debate_stop_after")
        state_file = kwargs.get("stage_state_file")
        debate_calls.append(
            (
                None if stop_after is None else str(stop_after),
                None if state_file is None else str(state_file),
            )
        )
        if state_file is not None:
            from debate_v_majority.cli.stage_state import append_stage_entry

            append_stage_entry(
                Path(state_file),
                make_stage_entry(
                    stage_type="debate",
                    completed_stage=str(stop_after or "round_0_judge"),
                    dataset="hle",
                    items=[item.__dict__ for item in kwargs["items"]],
                    debate_data={
                        "n_agents": kwargs["n_agents"],
                        "n_rounds": kwargs["n_rounds"],
                        "judge_rounds": kwargs["judge_rounds"],
                    },
                ),
            )
        if stop_after == "round_0":
            raise _FakeDebateStopped()
        return {0: [_debate_row(item, use_personas=bool(kwargs.get("use_personas"))) for item in kwargs["items"]]}

    monkeypatch.setattr(cli_main_impl, "_make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr(cli_main_impl, "build_sampling_config", lambda model_name: SimpleNamespace(max_tokens=128))
    monkeypatch.setattr(cli_main_impl, "set_sampling_config", lambda cfg: None)
    monkeypatch.setattr(cli_main_impl, "_create_role_engine", _fake_role_engine)
    monkeypatch.setattr(cli_main_impl, "_reuse_or_create_role_engine", lambda **kwargs: kwargs.get("existing_engine") or _FakeEngine())
    monkeypatch.setattr(cli_main_impl, "run_sampled", _fake_run_sampled)
    monkeypatch.setattr(cli_main_impl, "run_debate", _fake_run_debate)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_experiment_arm", lambda **kwargs: True)
    monkeypatch.setattr(cli_main_impl, "_prompt_continue_debate_stage", lambda **kwargs: False)
    monkeypatch.setitem(sys.modules, "debate_v_majority.cli.debate_runner", SimpleNamespace(_DebateStopped=_FakeDebateStopped))
    monkeypatch.setattr(cli_main_impl, "_timestamp_tag", lambda: "20260313_130500")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )
    cli_main_impl.main()

    experiment_root = next(tmp_path.glob("hle_experiment_*"))
    debate_stage_file = experiment_root / "debate_plain" / "stage_state.jsonl"
    assert debate_stage_file.exists()
    assert sample_calls == [[item.item_uid for item in items]]
    assert debate_calls == [("round_0", str(debate_stage_file))]

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "hle",
            "--hle_experiment",
            "--hle_experiment_stop_after",
            "debate_plain",
            "--stage_state_file",
            str(stage_state_file),
            "--model_name",
            "gemini-3-flash-preview",
            "--provider",
            "gemini",
            "--subset_n",
            "2",
            "--subset_seed",
            "7",
            "--hle_variant",
            "verified",
            "--n_agents",
            "2",
            "--n_rounds",
            "0",
            "--persona_backend",
            "llm",
            "--out_dir",
            str(tmp_path),
            "--token_ledger_path",
            str(tmp_path / "token-ledger.jsonl"),
            "--quiet",
        ],
    )
    cli_main_impl.main()

    assert sample_calls == [[item.item_uid for item in items]]
    assert debate_calls == [
        ("round_0", str(debate_stage_file)),
        (None, str(debate_stage_file)),
    ]


