from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import DatasetName
from ..engines import infer_provider_name
from ..personas import FIXED_AXIS_BANK
from ..personas.prompt_templates import (
    ARTIFACT_VERSION as PERSONA_ARTIFACT_VERSION,
    AXIS_PROMPT_VERSION,
    CARD_PROMPT_VERSION,
    DESCRIPTOR_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
)


FINAL_OUTPUT_SCHEMA_VERSION = "phase7.logical.v2"
FINAL_MANIFEST_VERSION = "phase7.final_manifest.v1"


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(r["final_correct"]) for r in rows) / len(rows)


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _model_tag(model_name: str) -> str:
    s = (model_name or "").strip()
    if not s:
        return "model"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "model"


def _dataset_tag(dataset: DatasetName) -> str:
    return "aime" if dataset == "aime25" else dataset


def _ids_tag(ids: list[int], *, max_ids: int = 5) -> str:
    if not ids:
        return ""
    if len(ids) <= max_ids:
        return "ids" + "-".join(str(i) for i in ids)
    head = "-".join(str(i) for i in ids[:max_ids])
    return f"ids{head}-plus{len(ids) - max_ids}"


def _range_tag(range_str: str | None) -> str:
    s = (range_str or "").strip()
    if not s:
        return ""
    return "range" + re.sub(r"[^0-9]+", "_", s).strip("_")


def _build_run_tag(*, tag: str | None, meta: dict[str, Any], subset_spec_tag: str, timestamp_tag: str) -> str:
    base = tag or f"n{meta['subset_size']}_seed{meta['seed']}"
    if subset_spec_tag:
        base = f"{base}_{subset_spec_tag}"
    return f"{base}_{timestamp_tag}"


def _default_out_dir(dataset: DatasetName) -> Path:
    return Path.cwd() / "results" / f"{dataset}_quick"


def _axis_bank_version() -> str:
    axis_ids = ",".join(axis.axis_id for axis in FIXED_AXIS_BANK)
    return f"fixed_bank.{len(FIXED_AXIS_BANK)}.{hashlib.sha256(axis_ids.encode('utf-8')).hexdigest()[:8]}"


def _short_question_text(question: Any, *, limit: int = 160) -> str | None:
    if question is None:
        return None
    text = " ".join(str(question).split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _display_block(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_short": _short_question_text(row.get("question")),
        "persona_summaries": row.get("persona_summaries") or row.get("sample_persona_summaries"),
        "judge_summary": row.get("judge_summary") or (row.get("judge_meta") or {}).get("judge_summary"),
        "round1_majority_answer": row.get("round1_majority_answer"),
        "final_round_majority_answer": row.get("final_round_majority_answer"),
        "judge_final_answer": row.get("final_judge_answer") or row.get("judge_final_answer"),
        "final_correctness_summary": {
            "final_answer": row.get("final_answer"),
            "final_correct": row.get("final_correct"),
        },
    }


def _trace_block(row: dict[str, Any], *, emit_trace_level: str) -> dict[str, Any]:
    full = str(emit_trace_level).strip().lower() == "full"
    trace: dict[str, Any] = {
        "token_usage_summary": row.get("token_usage_summary"),
    }
    if row.get("sample_call_metadata") is not None:
        trace["engine_calls"] = row.get("sample_call_metadata") if full else None
        trace["sample_extractions"] = row.get("sample_extractions") if full else None
    if row.get("agent_round_outputs") is not None:
        trace["engine_calls"] = {
            "debater_round_token_usage": row.get("debater_round_token_usage"),
            "judge_round_token_usage": row.get("judge_round_token_usage"),
        }
        if full:
            trace["per_round_agent_outputs"] = row.get("agent_round_outputs")
            trace["judge_trace"] = row.get("judge_trace")
            trace["agent_responses"] = row.get("agent_responses")
    if row.get("axes") is not None or row.get("cards") is not None:
        if full:
            trace["persona_generation"] = {
                "axes": row.get("axes"),
                "sampled_points": row.get("sampled_points"),
                "descriptors": row.get("descriptors"),
                "cards": row.get("cards"),
                "judge_card": row.get("judge_card"),
                "validator_metadata": row.get("validator_metadata"),
            }
    return {key: value for key, value in trace.items() if value is not None}


def _strategy_block(
    row: dict[str, Any],
    *,
    mode: str,
    use_personas: bool,
    judge_trace_mode: str | None = None,
    public_rationale_max_tokens: int | None,
    emit_trace_level: str,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "mode": mode,
        "use_personas": bool(use_personas),
        "judge_trace_mode": judge_trace_mode,
        "public_rationale_max_tokens": public_rationale_max_tokens,
        "emit_trace_level": emit_trace_level,
        "n_agents": row.get("n_agents"),
        "n_rounds": row.get("n_rounds"),
        "n_samples": row.get("n_samples"),
    }
    persona_meta = row.get("persona_meta")
    if isinstance(persona_meta, dict):
        n_plain = persona_meta.get("n_plain_agents", 0)
        if n_plain:
            block["persona_plain_agents"] = int(n_plain)
    return block


def _task_block(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "item_uid": row.get("item_uid"),
        "dataset_revision": row.get("dataset_revision"),
        "item_display_id": row.get("item_display_id"),
        "subset_id": row.get("subset_id"),
        "orig_id": row.get("orig_id"),
        "raw_task": row.get("raw_task"),
    }


def _results_block(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode") or "").strip().lower()
    final_answer_source = "single"
    if mode == "majority":
        final_answer_source = "majority"
    elif mode == "debate":
        final_answer_source = "judge"
    judge_final_answer = row.get("final_judge_answer")
    if judge_final_answer is None:
        judge_final_answer = row.get("judge_final_answer")
    judge_final_correct = row.get("final_judge_correct")
    if judge_final_correct is None:
        judge_final_correct = row.get("judge_final_correct")
    return {
        "final_answer": row.get("final_answer"),
        "final_correct": row.get("final_correct"),
        "final_answer_source": final_answer_source,
        "round1_majority_result": row.get("round1_majority_result"),
        "round1_majority_answer": row.get("round1_majority_answer"),
        "round1_majority_correct": row.get("round1_majority_correct"),
        "final_round_majority_result": row.get("final_round_majority_result"),
        "final_round_majority_answer": row.get("final_round_majority_answer"),
        "final_round_majority_correct": row.get("final_round_majority_correct"),
        "judge_result": row.get("judge_result"),
        "judge_final_correct": judge_final_correct,
        "majority_result": row.get("majority_result"),
        "vote_counts": row.get("vote_counts"),
        "final_majority_answer": row.get("final_majority_answer"),
        "judge_final_answer": judge_final_answer,
    }


def _persona_meta_block(row: dict[str, Any]) -> dict[str, Any] | None:
    persona_meta = row.get("persona_meta")
    if isinstance(persona_meta, dict):
        return dict(persona_meta)
    def _present(value: Any) -> bool:
        return value not in (None, "", [], {})
    if not any(
        _present(row.get(key))
        for key in (
            "artifact_path",
            "persona_seed",
            "sample_persona_ids",
            "sample_persona_summaries",
            "persona_ids",
            "persona_summaries",
            "axes",
            "cards",
        )
    ):
        return None
    return {
        "artifact_path": row.get("artifact_path"),
        "persona_seed": row.get("persona_seed"),
        "persona_ids": row.get("persona_ids") or row.get("sample_persona_ids"),
        "persona_summaries": row.get("persona_summaries") or row.get("sample_persona_summaries"),
        "axes_mode": row.get("axes_mode"),
        "backend": row.get("persona_backend"),
    }


def _judge_meta_block(row: dict[str, Any]) -> dict[str, Any] | None:
    judge_meta = row.get("judge_meta")
    if isinstance(judge_meta, dict):
        return dict(judge_meta)
    judge_summary = row.get("judge_summary")
    judge_card = row.get("judge_card")
    if judge_summary is None and judge_card is None and row.get("judge_trace") is None:
        return None
    return {
        "judge_summary": judge_summary,
        "judge_card": judge_card,
    }


def _augment_output_rows(
    rows: list[dict[str, Any]],
    *,
    run_meta: dict[str, Any],
    mode: str,
    use_personas: bool,
    judge_trace_mode: str | None = None,
    public_rationale_max_tokens: int | None,
    emit_trace_level: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["run_meta"] = dict(run_meta)
        updated["task"] = _task_block(updated)
        updated["strategy"] = _strategy_block(
            updated,
            mode=mode,
            use_personas=use_personas,
            judge_trace_mode=judge_trace_mode,
            public_rationale_max_tokens=public_rationale_max_tokens,
            emit_trace_level=emit_trace_level,
        )
        updated["persona_meta"] = _persona_meta_block(updated)
        updated["judge_meta"] = _judge_meta_block(updated)
        updated["results"] = _results_block(updated)
        updated["display"] = _display_block(updated)
        updated["trace"] = _trace_block(updated, emit_trace_level=emit_trace_level)
        out.append(updated)
    return out


def _run_meta_block(
    *,
    run_tag: str,
    dataset: DatasetName,
    meta: dict[str, Any],
    manifest_path: Path | None,
    output_path: Path | None,
    emit_trace_level: str,
) -> dict[str, Any]:
    return {
        "run_tag": run_tag,
        "dataset": dataset,
        "dataset_meta": dict(meta),
        "output_schema_version": FINAL_OUTPUT_SCHEMA_VERSION,
        "emit_trace_level": emit_trace_level,
        "final_manifest_path": None if manifest_path is None else str(manifest_path),
        "output_path": None if output_path is None else str(output_path),
    }


def _build_final_manifest(
    *,
    args: argparse.Namespace,
    dataset: DatasetName,
    meta: dict[str, Any],
    items: list[Any],
    provider_name: str | None,
    generator_provider_name: str | None,
    judge_provider_name: str | None,
    judge_generator_provider_name: str | None,
    generator_model_name: str | None,
    judge_generator_model_name: str | None,
    judge_runtime_model_name: str | None,
    persona_backend: str,
    modes: list[str],
    emit_trace_level: str,
) -> dict[str, Any]:
    def _role_binding(model_name: str | None, explicit_provider_name: str | None) -> dict[str, Any]:
        return {
            "model": model_name,
            "provider": None if model_name is None else (explicit_provider_name or infer_provider_name(model_name)),
        }

    return {
        "manifest_version": FINAL_MANIFEST_VERSION,
        "locked_config": {
            "dataset": dataset,
            "dataset_revision": meta.get("dataset_revision"),
            "dataset_variant": meta.get("dataset_variant"),
            "source_path": meta.get("source_path"),
            "subset_seed": meta.get("seed"),
            "subset_size": meta.get("subset_size"),
            "item_uids": [item.item_uid for item in items],
            "model_roles": {
                "debater": _role_binding(args.model_name, provider_name),
                "judge_runtime": _role_binding(judge_runtime_model_name or args.model_name, judge_provider_name),
                "generator": _role_binding(generator_model_name, generator_provider_name),
                "judge_generator": _role_binding(judge_generator_model_name, judge_generator_provider_name),
            },
            "persona": {
                "use_personas": bool(args.use_personas),
                "backend": persona_backend,
                "axis_mode": str(args.persona_axes_mode),
                "axis_bank_version": _axis_bank_version(),
                "artifact_version": PERSONA_ARTIFACT_VERSION,
                "prompt_versions": {
                    "axis": AXIS_PROMPT_VERSION,
                    "descriptor": DESCRIPTOR_PROMPT_VERSION,
                    "card": CARD_PROMPT_VERSION,
                    "judge": JUDGE_PROMPT_VERSION,
                },
                "judge_persona_mode": str(args.judge_persona_mode),
                "judge_bank_dir": None if args.judge_bank_dir is None else str(args.judge_bank_dir),
                "judge_bank_refresh": bool(args.judge_bank_refresh),
                "gpqa_family_cache_path": None if args.gpqa_family_cache_path is None else str(args.gpqa_family_cache_path),
            },
            "debate": {
                "judge_trace_mode": str(args.judge_trace_mode),
                "public_rationale_max_tokens": int(args.public_rationale_max_tokens),
                "n_agents": int(args.n_agents),
                "n_rounds": int(args.n_rounds),
            },
            "output_schema_version": FINAL_OUTPUT_SCHEMA_VERSION,
            "emit_trace_level": emit_trace_level,
            "modes": list(modes),
        },
    }


def _write_or_validate_final_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    final_run: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if final_run and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("locked_config") != manifest.get("locked_config"):
            raise ValueError(f"--final_run manifest mismatch: {manifest_path}")
        return
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "FINAL_MANIFEST_VERSION",
    "FINAL_OUTPUT_SCHEMA_VERSION",
    "_accuracy",
    "_augment_output_rows",
    "_build_final_manifest",
    "_build_run_tag",
    "_dataset_tag",
    "_default_out_dir",
    "_ids_tag",
    "_model_tag",
    "_range_tag",
    "_run_meta_block",
    "_timestamp_tag",
    "_write_or_validate_final_manifest",
]
