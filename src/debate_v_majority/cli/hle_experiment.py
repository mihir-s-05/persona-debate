from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tools.trace2txt import render_row_text


HLE_EXPERIMENT_MANIFEST_VERSION = "phase8.hle_experiment_manifest.v2"
HLE_EXPERIMENT_STAGE_TYPE = "hle_experiment"
HLE_EXPERIMENT_ARMS = ("single", "debate_plain", "persona_debate")
HLE_EXPERIMENT_STOP_AFTER_CHOICES = HLE_EXPERIMENT_ARMS


def default_experiment_root(*, out_dir: Path, run_tag: str) -> Path:
    return out_dir / f"hle_experiment_{run_tag}"


def default_manifest_path(*, experiment_root: Path) -> Path:
    return experiment_root / "manifest.json"


def arm_dir(*, experiment_root: Path, arm_name: str) -> Path:
    return experiment_root / str(arm_name)


def arm_results_path(*, experiment_root: Path, arm_name: str) -> Path:
    return arm_dir(experiment_root=experiment_root, arm_name=arm_name) / "results.jsonl"


def arm_traces_dir(*, experiment_root: Path, arm_name: str) -> Path:
    return arm_dir(experiment_root=experiment_root, arm_name=arm_name) / "traces"


def additional_debate_round_path(*, experiment_root: Path, arm_name: str, round_num: int) -> Path:
    return arm_dir(experiment_root=experiment_root, arm_name=arm_name) / f"round_{int(round_num)}_results.jsonl"


def _sanitize_filename_part(value: Any) -> str:
    cleaned = "".join(
        ch if str(ch).isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(value or "")
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_.")
    return cleaned or "item"


def trace_path_for_row(*, traces_dir: Path, row: dict[str, Any], row_index: int) -> Path:
    item_uid = row.get("item_uid")
    item_display_id = row.get("item_display_id")
    subset_id = row.get("subset_id")
    prefix = f"{int(subset_id):03d}_" if subset_id is not None else f"{int(row_index):03d}_"
    suffix_source = item_uid if item_uid is not None else item_display_id
    suffix = _sanitize_filename_part(suffix_source if suffix_source is not None else f"row_{row_index}")
    return traces_dir / f"{prefix}{suffix}.txt"


def write_readable_traces(*, rows: list[dict[str, Any]], traces_dir: Path) -> list[str]:
    traces_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for row_index, row in enumerate(rows):
        trace_path = trace_path_for_row(traces_dir=traces_dir, row=row, row_index=row_index)
        trace_path.write_text(render_row_text(row), encoding="utf-8")
        written_paths.append(str(trace_path))
    return written_paths


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(*, path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _arm_manifest_block(*, experiment_root: Path, arm_name: str, mode: str, use_personas: bool, runtime_judge_persona_enabled: bool) -> dict[str, Any]:
    return {
        "mode": mode,
        "use_personas": bool(use_personas),
        "runtime_judge_persona_enabled": bool(runtime_judge_persona_enabled),
        "directory": str(arm_dir(experiment_root=experiment_root, arm_name=arm_name)),
        "results_path": str(arm_results_path(experiment_root=experiment_root, arm_name=arm_name)),
        "traces_dir": str(arm_traces_dir(experiment_root=experiment_root, arm_name=arm_name)),
        "additional_output_paths": [],
        "trace_paths": [],
        "completed": False,
        "row_count": None,
        "correct": None,
        "accuracy": None,
    }


def build_manifest(
    *,
    manifest_path: Path,
    experiment_root: Path,
    run_tag: str,
    dataset: str,
    meta: dict[str, Any],
    item_uids: list[str],
    stage_state_file: Path | None,
    model_name: str | None,
    generator_model_name: str | None,
    judge_runtime_model_name: str | None,
    judge_generator_model_name: str | None,
    persona_backend: str,
    n_agents: int,
    n_rounds: int,
    emit_trace_level: str,
    hle_variant: str,
    hle_modality: str,
    resume_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "manifest_version": HLE_EXPERIMENT_MANIFEST_VERSION,
        "created_at": now,
        "updated_at": now,
        "status": "pending",
        "run_tag": run_tag,
        "dataset": dataset,
        "dataset_meta": dict(meta),
        "item_uids": list(item_uids),
        "manifest_path": str(manifest_path),
        "experiment_root": str(experiment_root),
        "stage_state_file": None if stage_state_file is None else str(stage_state_file),
        "experiment": {
            "type": "hle_quick_compare",
            "hle_variant": str(hle_variant),
            "hle_modality": str(hle_modality),
            "n_agents": int(n_agents),
            "n_rounds": int(n_rounds),
            "emit_trace_level": str(emit_trace_level),
            "persona_backend": str(persona_backend),
            "resume_settings": dict(resume_settings or {}),
            "models": {
                "debater": model_name,
                "judge_runtime": judge_runtime_model_name,
                "generator": generator_model_name,
                "judge_generator": judge_generator_model_name,
            },
        },
        "arms": {
            "single": _arm_manifest_block(
                experiment_root=experiment_root,
                arm_name="single",
                mode="single",
                use_personas=False,
                runtime_judge_persona_enabled=False,
            ),
            "debate_plain": _arm_manifest_block(
                experiment_root=experiment_root,
                arm_name="debate_plain",
                mode="debate",
                use_personas=False,
                runtime_judge_persona_enabled=False,
            ),
            "persona_debate": _arm_manifest_block(
                experiment_root=experiment_root,
                arm_name="persona_debate",
                mode="debate",
                use_personas=True,
                runtime_judge_persona_enabled=True,
            ),
        },
    }


def update_manifest_for_arm(
    *,
    manifest: dict[str, Any],
    arm_name: str,
    rows: list[dict[str, Any]],
    trace_paths: list[str],
    additional_output_paths: list[str] | None = None,
) -> dict[str, Any]:
    updated = dict(manifest)
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    arms = dict(updated.get("arms") or {})
    arm_block = dict(arms.get(arm_name) or {})
    correct = sum(int(row.get("final_correct") or 0) for row in rows)
    row_count = len(rows)
    arm_block.update(
        {
            "completed": True,
            "row_count": row_count,
            "correct": correct,
            "accuracy": 0.0 if row_count == 0 else correct / row_count,
            "trace_paths": list(trace_paths),
            "additional_output_paths": list(additional_output_paths or []),
        }
    )
    arms[arm_name] = arm_block
    updated["arms"] = arms
    updated["status"] = "completed" if all(dict(block).get("completed") for block in arms.values()) else "partial"
    return updated


def completed_arms(manifest: dict[str, Any]) -> list[str]:
    arms = dict(manifest.get("arms") or {})
    return [arm_name for arm_name in HLE_EXPERIMENT_ARMS if dict(arms.get(arm_name) or {}).get("completed")]
