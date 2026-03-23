"""
CLI for debug_majority_debate package.

Provides simplified command-line interface for running inference/evaluation
on AIME25, GPQA, and HLE-Verified datasets with single, majority, or debate modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, TextIO, cast

from .. import DatasetName, Mode
from ..accounting import CostTracker, SpendLimitExceeded, active_cost_tracking, format_cost_summary
from ..datasets import get_dataset_adapter as resolve_dataset_adapter
from ..personas import FIXED_AXIS_BANK
from ..personas.prompt_templates import (
    ARTIFACT_VERSION as PERSONA_ARTIFACT_VERSION,
    AXIS_PROMPT_VERSION,
    CARD_PROMPT_VERSION,
    DESCRIPTOR_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
)


class _DoubleCtrlCHandler:
    """
    Require two Ctrl+C presses to interrupt.

    The first Ctrl+C prints a warning; a second Ctrl+C within `timeout` seconds exits (code 130).
    """

    def __init__(self, timeout: float = 2.0, output_file: TextIO | None = None) -> None:
        self.timeout = timeout
        self.output_file = output_file or sys.stderr
        self._last_sigint_time: float | None = None
        self._original_handler = None

    def _handler(self, signum: int, frame) -> None:
        now = time.monotonic()
        if self._last_sigint_time is not None and (now - self._last_sigint_time) < self.timeout:
            print("\nInterrupted.", file=self.output_file, flush=True)
            sys.exit(130)
        else:
            self._last_sigint_time = now
            print(
                f"\nPress Ctrl+C again within {self.timeout:.0f}s to cancel...",
                file=self.output_file,
                flush=True,
            )

    def __enter__(self) -> "_DoubleCtrlCHandler":
        self._last_sigint_time = None
        try:
            self._original_handler = signal.signal(signal.SIGINT, self._handler)
        except ValueError:
            self._original_handler = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
        return False


from ..engines import (
    GEMINI_3_FLASH_MODEL,
    InferenceEngine,
    build_sampling_config,
    create_inference_engine,
    infer_provider_name,
    normalize_gemini_model_name,
    set_sampling_config,
)
from .args import _build_arg_parser, _parse_judge_rounds
from .dataset_eval import _parse_question_answer
from .engine_runtime import (
    _create_role_engine,
    _default_token_ledger_path,
    _merge_token_counts,
    _reuse_or_create_role_engine,
)
from .hle_experiment import (
    HLE_EXPERIMENT_ARMS,
    HLE_EXPERIMENT_STAGE_TYPE,
    additional_debate_round_path,
    arm_results_path,
    arm_traces_dir,
    build_manifest as _build_hle_experiment_manifest,
    completed_arms as _completed_hle_experiment_arms,
    default_experiment_root,
    default_manifest_path as _default_hle_experiment_manifest_path,
    load_manifest,
    update_manifest_for_arm,
    write_manifest as _write_hle_experiment_manifest,
    write_readable_traces,
)
from .output import (
    FINAL_OUTPUT_SCHEMA_VERSION,
    _accuracy,
    _augment_output_rows,
    _build_final_manifest,
    _build_run_tag,
    _dataset_tag,
    _default_out_dir,
    _ids_tag,
    _model_tag,
    _range_tag,
    _run_meta_block,
    _timestamp_tag,
    _write_or_validate_final_manifest,
)
from .persona_runtime import (
    _resolve_persona_artifact,
    persona_artifacts_from_rows,
    persona_rows_from_stage_entry,
    run_persona_generation,
    run_persona_generation_staged,
)
from .stage_state import (
    append_stage_entry,
    load_latest_stage_entry_of_type,
    make_stage_entry,
    path_setting,
    subset_item_resume_signature,
)
from . import subset as _subset
from .subset import (
    SubsetItem,
    _ensure_dataset_test_jsonl,
    _load_exclude_id_map,
    _make_dataset_subset,
    _select_one_item,
    _write_jsonl,
)

class _QuietOutput:
    """
    Silence stdout/stderr noise (e.g., provider/SDK logs) while still allowing
    explicitly chosen output (progress bars, summary) to be shown.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self.keep_stdout: TextIO = sys.stdout

    def __enter__(self) -> "_QuietOutput":
        if not self.enabled:
            self.keep_stdout = sys.stdout
            return self

        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        self.keep_stdout = os.fdopen(
            self._saved_stdout_fd,
            "w",
            buffering=1,
            encoding="utf-8",
            errors="replace",
        )

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
        finally:
            try:
                os.close(devnull_fd)
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False

        if self._saved_stdout_fd is not None:
            try:
                os.dup2(self._saved_stdout_fd, 1)
            except Exception:
                pass
        if self._saved_stderr_fd is not None:
            try:
                os.dup2(self._saved_stderr_fd, 2)
            except Exception:
                pass

        try:
            self.keep_stdout.flush()
        except Exception:
            pass
        try:
            self.keep_stdout.close()
        except Exception:
            pass
        if self._saved_stderr_fd is not None:
            try:
                os.close(self._saved_stderr_fd)
            except Exception:
                pass

        return False

def _effective_persona_backend(*, requested_backend: str, generator_model_name: str | None) -> str:
    _ = generator_model_name
    backend = str(requested_backend).strip().lower()
    if backend != "llm":
        raise ValueError(f"Unsupported persona backend: {requested_backend}")
    return "llm"


def run_sampled(*args, **kwargs):
    from .sample_runner import run_sampled as _run_sampled

    return _run_sampled(*args, **kwargs)


def run_debate(*args, **kwargs):
    from .debate_runner import run_debate as _run_debate

    return _run_debate(*args, **kwargs)


def _auto_debate_stage_pointer_path(*, out_dir: Path, dataset_tag: str) -> Path:
    return out_dir / f"stage_state_{dataset_tag}_debate.latest"


def _read_auto_debate_stage_pointer(pointer_path: Path) -> Path | None:
    if not pointer_path.exists():
        return None
    try:
        raw_value = pointer_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_value:
        return None
    return Path(raw_value)


def _write_auto_debate_stage_pointer(pointer_path: Path, stage_state_file: Path) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(stage_state_file), encoding="utf-8")


def _clear_auto_debate_stage_pointer(pointer_path: Path) -> None:
    try:
        pointer_path.unlink()
    except FileNotFoundError:
        return


def _is_debate_resume_settings_mismatch(exc: ValueError) -> bool:
    msg = str(exc)
    return (
        "Debate state runtime settings mismatch" in msg
        or "Debate state persona settings mismatch" in msg
    )


def _parse_debate_stage_name(stage_name: str) -> tuple[int, bool] | None:
    if not stage_name.startswith("round_"):
        return None
    suffix = stage_name[len("round_") :]
    is_judged = suffix.endswith("_judge")
    if is_judged:
        suffix = suffix[: -len("_judge")]
    try:
        return int(suffix), is_judged
    except ValueError:
        return None


def _auto_resume_debate_stage_file(
    *,
    pointer_path: Path,
    dataset: DatasetName,
    items: list[SubsetItem],
    n_agents: int,
    n_rounds: int,
    judge_rounds: list[int],
) -> Path | None:
    stage_state_file = _read_auto_debate_stage_pointer(pointer_path)
    if stage_state_file is None or not stage_state_file.exists():
        _clear_auto_debate_stage_pointer(pointer_path)
        return None
    try:
        entry = load_latest_stage_entry_of_type(stage_state_file, "debate")
    except (OSError, ValueError, json.JSONDecodeError):
        _clear_auto_debate_stage_pointer(pointer_path)
        return None
    if entry is None:
        _clear_auto_debate_stage_pointer(pointer_path)
        return None
    if entry.dataset != str(dataset):
        return None
    saved_items = [subset_item_resume_signature(row) for row in entry.items]
    current_items = [subset_item_resume_signature(item) for item in items]
    if saved_items != current_items:
        return None
    debate_meta = dict(entry.debate_data or {})
    if int(debate_meta.get("n_agents", n_agents)) != int(n_agents):
        return None
    if int(debate_meta.get("n_rounds", n_rounds)) != int(n_rounds):
        return None
    resume_judge_rounds = debate_meta.get("judge_rounds")
    if not isinstance(resume_judge_rounds, list):
        return None
    if [int(round_num) for round_num in resume_judge_rounds] != [int(round_num) for round_num in judge_rounds]:
        return None
    return stage_state_file


def _validate_debate_stop_after(*, debate_stop_after: str | None, n_rounds: int, judge_rounds: list[int]) -> None:
    if debate_stop_after is None:
        return
    parsed_stage = _parse_debate_stage_name(str(debate_stop_after))
    if parsed_stage is None:
        raise ValueError(
            f"Invalid --debate_stop_after stage: {debate_stop_after!r}. "
            "Expected values like 'round_0' or 'round_1_judge'."
        )
    round_num, is_judge_stage = parsed_stage
    if round_num < 0 or round_num > int(n_rounds):
        raise ValueError(
            f"--debate_stop_after stage {debate_stop_after!r} is outside the configured debate range "
            f"(valid rounds: 0..{int(n_rounds)})."
        )
    if is_judge_stage and round_num not in {int(x) for x in judge_rounds}:
        valid_judge_stages = ", ".join(f"round_{int(x)}_judge" for x in judge_rounds) or "none"
        raise ValueError(
            f"--debate_stop_after stage {debate_stop_after!r} is not a judged step for this run "
            f"(valid judge stages: {valid_judge_stages})."
        )


def _next_debate_stage(*, completed_stage: str, n_rounds: int, judge_rounds: list[int]) -> str | None:
    parsed_stage = _parse_debate_stage_name(completed_stage)
    if parsed_stage is None:
        return None
    round_num, is_judge_stage = parsed_stage
    normalized_judge_rounds = {int(x) for x in judge_rounds}
    final_round = max(0, int(n_rounds))
    if is_judge_stage:
        next_round = round_num + 1
        return f"round_{next_round}" if next_round <= final_round else None
    if round_num in normalized_judge_rounds:
        return f"round_{round_num}_judge"
    next_round = round_num + 1
    return f"round_{next_round}" if next_round <= final_round else None


_PERSONA_STAGE_ORDER = ("axes", "descriptors", "cards", "judge_card")


def _next_persona_stage(completed_stage: str) -> str | None:
    try:
        idx = _PERSONA_STAGE_ORDER.index(str(completed_stage))
    except ValueError:
        return None
    next_idx = idx + 1
    return _PERSONA_STAGE_ORDER[next_idx] if next_idx < len(_PERSONA_STAGE_ORDER) else None


def _prompt_continue_persona_stage(*, next_stage: str, status_file: TextIO) -> bool:
    prompt = (
        f"[staged] Press Enter to continue to '{next_stage}', "
        "or type 'q' to save and stop: "
    )
    previous_handler = None
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        previous_handler = None
    try:
        response = input(prompt)
    except KeyboardInterrupt:
        print("\n[staged] Interrupted. Progress has been saved.", file=status_file)
        return False
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except ValueError:
                pass
    return str(response).strip().lower() not in {"q", "quit"}


def _prompt_continue_debate_stage(*, next_stage: str, status_file: TextIO) -> bool:
    prompt = (
        f"[staged] Press Enter to continue to debate stage '{next_stage}', "
        "or type 'q' to save and stop: "
    )
    previous_handler = None
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        previous_handler = None
    try:
        response = input(prompt)
    except KeyboardInterrupt:
        print("\n[staged] Interrupted. Progress has been saved.", file=status_file)
        return False
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except ValueError:
                pass
    return str(response).strip().lower() not in {"q", "quit"}


def _prompt_continue_to_debate(*, status_file: TextIO) -> bool:
    prompt = (
        "[staged] Personas are complete. Press Enter to start debate, "
        "or type 'q' to save and stop: "
    )
    previous_handler = None
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        previous_handler = None
    try:
        response = input(prompt)
    except KeyboardInterrupt:
        print("\n[staged] Interrupted. Progress has been saved.", file=status_file)
        return False
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except ValueError:
                pass
    return str(response).strip().lower() not in {"q", "quit"}


def _prompt_continue_experiment_arm(*, next_arm: str, status_file: TextIO) -> bool:
    prompt = (
        f"[staged] Press Enter to continue to experiment arm '{next_arm}', "
        "or type 'q' to save and stop: "
    )
    previous_handler = None
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        previous_handler = None
    try:
        response = input(prompt)
    except KeyboardInterrupt:
        print("\n[staged] Interrupted. Progress has been saved.", file=status_file)
        return False
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except ValueError:
                pass
    return str(response).strip().lower() not in {"q", "quit"}


def _cost_summary_snapshot(cost_tracker: CostTracker | None) -> dict[str, Any] | None:
    if cost_tracker is None:
        return None
    return cost_tracker.summary()


def _format_remaining_budget(limit_usd: float | None, spent_usd: float) -> str:
    if limit_usd is None:
        return "unbounded"
    return f"${max(0.0, float(limit_usd) - float(spent_usd)):.6f}"


def _format_stage_cost_summary(
    *,
    label: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    cost_tracker: CostTracker | None,
) -> str | None:
    if before is None or after is None or cost_tracker is None:
        return None
    before_session = dict(before.get("session") or {})
    after_session = dict(after.get("session") or {})
    after_cumulative = dict(after.get("cumulative") or {})
    delta_calls = int(after_session.get("n_calls", 0)) - int(before_session.get("n_calls", 0))
    delta_input = int(after_session.get("input_tokens", 0)) - int(before_session.get("input_tokens", 0))
    delta_cached = int(after_session.get("cached_input_tokens", 0)) - int(before_session.get("cached_input_tokens", 0))
    delta_output = int(after_session.get("billable_output_tokens", 0)) - int(before_session.get("billable_output_tokens", 0))
    delta_cost = float(after_session.get("estimated_cost_usd", 0.0)) - float(before_session.get("estimated_cost_usd", 0.0))
    run_total = float(after_session.get("estimated_cost_usd", 0.0))
    cumulative_total = float(after_cumulative.get("estimated_cost_usd", 0.0))
    return (
        f"[cost] {label}=${delta_cost:.6f} "
        f"(calls={delta_calls}, input={delta_input}, cached={delta_cached}, output={delta_output}) | "
        f"run_total=${run_total:.6f} | "
        f"run_remaining={_format_remaining_budget(cost_tracker.max_run_cost_usd, run_total)} | "
        f"cumulative=${cumulative_total:.6f} | "
        f"cumulative_remaining={_format_remaining_budget(cost_tracker.max_total_cost_usd, cumulative_total)} | "
        f"ledger={cost_tracker.ledger_path}"
    )


def _print_stage_cost_summary(
    *,
    label: str,
    before: dict[str, Any] | None,
    cost_tracker: CostTracker | None,
    status_file: TextIO,
) -> None:
    summary = _format_stage_cost_summary(
        label=label,
        before=before,
        after=_cost_summary_snapshot(cost_tracker),
        cost_tracker=cost_tracker,
    )
    if summary:
        print(summary, file=status_file)


def _dataset_test_path_candidates(
    dataset: DatasetName,
    *,
    source_file: Path | None = None,
    hle_variant: str | None = None,
    dataset_local_mirror: Path | None = None,
) -> list[Path]:
    return _subset._dataset_test_path_candidates(
        dataset,
        source_file=source_file,
        hle_variant=hle_variant,
        dataset_local_mirror=dataset_local_mirror,
    )


def _default_dataset_test_path(
    dataset: DatasetName,
    *,
    hle_variant: str | None = None,
    dataset_local_mirror: Path | None = None,
) -> Path:
    candidates = _dataset_test_path_candidates(
        dataset,
        hle_variant=hle_variant,
        dataset_local_mirror=dataset_local_mirror,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _hle_experiment_resume_settings(
    *,
    dataset: DatasetName,
    items: list[SubsetItem],
    hle_variant: str,
    hle_modality: str,
    model_name: str | None,
    provider_name: str | None,
    generator_model_name: str | None,
    generator_provider_name: str | None,
    judge_runtime_model_name: str | None,
    judge_provider_name: str | None,
    judge_generator_model_name: str | None,
    judge_generator_provider_name: str | None,
    persona_backend: str,
    n_agents: int,
    n_rounds: int,
    judge_rounds: list[int],
    judge_block_size: int | None,
    judge_sampling_kwargs: dict[str, Any] | None,
    judge_strict_final_only: bool,
    judge_recovery_parse_enabled: bool,
    persona_seed: int,
    persona_axes_mode: str,
    persona_fixed_axis_count: int,
    persona_task_axis_count: int,
    persona_sampling_method: str,
    judge_persona_mode: str,
    persona_axes_file: Path | None,
    judge_bank_dir: Path | None,
    judge_bank_refresh: bool,
    gpqa_family_cache_path: Path | None,
    judge_trace_mode: str,
    public_rationale_max_tokens: int,
    persona_save_artifacts: bool,
    persona_replay: bool,
    emit_trace_level: str,
    debate_protocol: str,
    persona_plain_agents: int = 0,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "hle_variant": str(hle_variant),
        "hle_modality": str(hle_modality),
        "item_signatures": [subset_item_resume_signature(item) for item in items],
        "models": {
            "debater": model_name,
            "debater_provider": provider_name,
            "generator": generator_model_name,
            "generator_provider": generator_provider_name,
            "judge_runtime": judge_runtime_model_name,
            "judge_provider": judge_provider_name,
            "judge_generator": judge_generator_model_name,
            "judge_generator_provider": judge_generator_provider_name,
        },
        "n_agents": int(n_agents),
        "n_rounds": int(n_rounds),
        "debate_protocol": str(debate_protocol),
        "judge_rounds": [int(round_num) for round_num in judge_rounds],
        "persona_generation": {
            "n_personas": int(n_agents) - int(persona_plain_agents),
            "n_plain_agents": int(persona_plain_agents),
            "persona_seed": int(persona_seed),
            "axis_mode": str(persona_axes_mode),
            "fixed_axis_count": int(persona_fixed_axis_count),
            "task_axis_count": int(persona_task_axis_count),
            "sampling_method": str(persona_sampling_method),
            "judge_persona_mode": str(judge_persona_mode),
            "requested_backend": str(persona_backend),
            "effective_backend": str(persona_backend),
            "generator_model": generator_model_name,
            "judge_generator_model": judge_generator_model_name,
            "axes_file": path_setting(persona_axes_file),
            "save_artifacts": bool(persona_save_artifacts),
            "replay": bool(persona_replay),
        },
        "debate_plain": {
            "persona_settings": {
                "use_personas": False,
                "runtime_judge_persona_enabled": False,
                "persona_seed": int(persona_seed),
                "persona_axis_mode": str(persona_axes_mode),
                "persona_fixed_axis_count": int(persona_fixed_axis_count),
                "persona_task_axis_count": int(persona_task_axis_count),
                "persona_sampling_method": str(persona_sampling_method),
                "persona_judge_mode": str(judge_persona_mode),
                "persona_backend": str(persona_backend),
                "generator_model": generator_model_name,
                "judge_generator_model": judge_generator_model_name,
                "persona_axes_file": path_setting(persona_axes_file),
                "judge_bank_dir": path_setting(judge_bank_dir),
                "judge_bank_refresh": bool(judge_bank_refresh),
                "gpqa_family_cache_path": path_setting(gpqa_family_cache_path),
            },
            "runtime_settings": {
                "debater_model": model_name,
                "debater_backend": provider_name,
                "judge_model": judge_runtime_model_name,
                "judge_backend": judge_provider_name,
                "judge_block_size": None if judge_block_size is None else int(judge_block_size),
                "judge_sampling_kwargs": dict(judge_sampling_kwargs or {}),
                "judge_strict_final_only": bool(judge_strict_final_only),
                "judge_recovery_parse_enabled": bool(judge_recovery_parse_enabled),
                "judge_trace_mode": str(judge_trace_mode),
                "public_rationale_max_tokens": int(public_rationale_max_tokens),
                "debate_protocol": str(debate_protocol),
            },
        },
        "persona_debate": {
            "persona_settings": {
                "use_personas": True,
                "runtime_judge_persona_enabled": True,
                "persona_seed": int(persona_seed),
                "persona_axis_mode": str(persona_axes_mode),
                "persona_fixed_axis_count": int(persona_fixed_axis_count),
                "persona_task_axis_count": int(persona_task_axis_count),
                "persona_sampling_method": str(persona_sampling_method),
                "persona_judge_mode": str(judge_persona_mode),
                "persona_backend": str(persona_backend),
                "generator_model": generator_model_name,
                "judge_generator_model": judge_generator_model_name,
                "persona_axes_file": path_setting(persona_axes_file),
                "judge_bank_dir": path_setting(judge_bank_dir),
                "judge_bank_refresh": bool(judge_bank_refresh),
                "gpqa_family_cache_path": path_setting(gpqa_family_cache_path),
                "persona_plain_agents": int(persona_plain_agents),
            },
            "runtime_settings": {
                "debater_model": model_name,
                "debater_backend": provider_name,
                "judge_model": judge_runtime_model_name,
                "judge_backend": judge_provider_name,
                "judge_block_size": None if judge_block_size is None else int(judge_block_size),
                "judge_sampling_kwargs": dict(judge_sampling_kwargs or {}),
                "judge_strict_final_only": bool(judge_strict_final_only),
                "judge_recovery_parse_enabled": bool(judge_recovery_parse_enabled),
                "judge_trace_mode": str(judge_trace_mode),
                "public_rationale_max_tokens": int(public_rationale_max_tokens),
                "debate_protocol": str(debate_protocol),
            },
        },
        "persona_backend": str(persona_backend),
        "emit_trace_level": str(emit_trace_level),
    }


def _load_hle_experiment_resume_state(
    *,
    stage_state_file: Path | None,
    resume_settings: dict[str, Any],
) -> dict[str, Any] | None:
    if stage_state_file is None or not stage_state_file.exists():
        return None
    entry = load_latest_stage_entry_of_type(stage_state_file, HLE_EXPERIMENT_STAGE_TYPE)
    if entry is None:
        return None
    saved_settings = dict(entry.meta.get("resume_settings") or {})
    if saved_settings != resume_settings:
        raise ValueError("HLE experiment state settings mismatch")
    experiment_state = dict(entry.meta.get("experiment_state") or {})
    if not experiment_state:
        raise ValueError("HLE experiment state missing experiment_state payload")
    return experiment_state


def _append_hle_experiment_stage(
    *,
    stage_state_file: Path | None,
    dataset: DatasetName,
    items: list[SubsetItem],
    completed_stage: str,
    experiment_state: dict[str, Any],
    resume_settings: dict[str, Any],
) -> None:
    if stage_state_file is None:
        return
    entry = make_stage_entry(
        stage_type=HLE_EXPERIMENT_STAGE_TYPE,
        completed_stage=completed_stage,
        dataset=str(dataset),
        items=[subset_item_resume_signature(item) for item in items],
        meta={
            "resume_settings": dict(resume_settings),
            "experiment_state": dict(experiment_state),
        },
    )
    append_stage_entry(stage_state_file, entry)


def _annotate_hle_experiment_rows(
    rows: list[dict[str, Any]],
    *,
    arm_name: str,
    run_meta: dict[str, Any],
    experiment_root: Path,
    manifest_path: Path,
    runtime_judge_persona_enabled: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        run_meta_block = dict(updated.get("run_meta") or run_meta)
        run_meta_block["experiment_arm"] = arm_name
        updated["run_meta"] = run_meta_block
        updated["experiment_arm"] = arm_name
        updated["experiment_meta"] = {
            "experiment_root": str(experiment_root),
            "manifest_path": str(manifest_path),
            "runtime_judge_persona_enabled": bool(runtime_judge_persona_enabled),
        }
        out.append(updated)
    return out


def _generate_readable_transcripts(
    *,
    trace_paths: list[str],
    experiment_root: Path,
    arm_name: str,
) -> None:
    """Generate readable transcript files with persona card headers for each trace."""
    from ..tools.extract_transcripts import extract_from_trace_file, find_artifact_for_uid
    import re as _re

    transcript_dir = experiment_root / arm_name / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    artifact_search_dirs = [
        str(experiment_root / "persona_artifacts" / "hle"),
        str(experiment_root / arm_name / "artifacts_llm" / "hle"),
    ]

    for trace_path in trace_paths:
        uid_match = _re.search(r'hle_([a-f0-9]+)\.txt', trace_path)
        uid_short = uid_match.group(1) if uid_match else None
        art_path = find_artifact_for_uid(uid_short, artifact_search_dirs) if uid_short else None
        extract_from_trace_file(trace_path, art_path, str(transcript_dir))


def _write_hle_experiment_arm_outputs(
    *,
    records: list[dict[str, Any]],
    arm_name: str,
    experiment_root: Path,
    manifest_path: Path,
    run_tag: str,
    dataset: DatasetName,
    meta: dict[str, Any],
    mode: str,
    use_personas: bool,
    runtime_judge_persona_enabled: bool,
    emit_trace_level: str,
    judge_trace_mode: str | None,
    public_rationale_max_tokens: int | None,
    results_by_round: dict[int, list[dict[str, Any]]] | None = None,
    judge_rounds: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    out_path = arm_results_path(experiment_root=experiment_root, arm_name=arm_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = _run_meta_block(
        run_tag=run_tag,
        dataset=dataset,
        meta=meta,
        manifest_path=manifest_path,
        output_path=out_path,
        emit_trace_level=emit_trace_level,
    )
    run_meta["experiment_arm"] = arm_name
    augmented = _augment_output_rows(
        records,
        run_meta=run_meta,
        mode=mode,
        use_personas=use_personas,
        judge_trace_mode=judge_trace_mode,
        public_rationale_max_tokens=public_rationale_max_tokens,
        emit_trace_level=emit_trace_level,
    )
    augmented = _annotate_hle_experiment_rows(
        augmented,
        arm_name=arm_name,
        run_meta=run_meta,
        experiment_root=experiment_root,
        manifest_path=manifest_path,
        runtime_judge_persona_enabled=runtime_judge_persona_enabled,
    )
    additional_output_paths: list[str] = []
    if results_by_round is not None and judge_rounds:
        final_round = max(int(round_num) for round_num in judge_rounds)
        for round_num in sorted(int(x) for x in judge_rounds):
            if round_num == final_round:
                continue
            round_path = additional_debate_round_path(
                experiment_root=experiment_root,
                arm_name=arm_name,
                round_num=round_num,
            )
            round_run_meta = _run_meta_block(
                run_tag=run_tag,
                dataset=dataset,
                meta=meta,
                manifest_path=manifest_path,
                output_path=round_path,
                emit_trace_level=emit_trace_level,
            )
            round_run_meta["experiment_arm"] = arm_name
            round_rows = _augment_output_rows(
                results_by_round[round_num],
                run_meta=round_run_meta,
                mode=mode,
                use_personas=use_personas,
                judge_trace_mode=judge_trace_mode,
                public_rationale_max_tokens=public_rationale_max_tokens,
                emit_trace_level=emit_trace_level,
            )
            round_rows = _annotate_hle_experiment_rows(
                round_rows,
                arm_name=arm_name,
                run_meta=round_run_meta,
                experiment_root=experiment_root,
                manifest_path=manifest_path,
                runtime_judge_persona_enabled=runtime_judge_persona_enabled,
            )
            _write_jsonl(round_path, round_rows)
            additional_output_paths.append(str(round_path))
    _write_jsonl(out_path, augmented)
    traces_dir = arm_traces_dir(experiment_root=experiment_root, arm_name=arm_name)
    trace_paths = write_readable_traces(
        rows=augmented,
        traces_dir=traces_dir,
    )
    if mode == "debate" and trace_paths:
        _generate_readable_transcripts(
            trace_paths=trace_paths,
            experiment_root=experiment_root,
            arm_name=arm_name,
        )
    return augmented, additional_output_paths, trace_paths



def main() -> None:
    """Main CLI entry point."""
    ap = _build_arg_parser()
    args = ap.parse_args()

    with _QuietOutput(bool(args.quiet)) as q:
        progress_file: TextIO = q.keep_stdout if args.quiet else sys.stdout
        status_file: TextIO = q.keep_stdout if args.quiet else sys.stderr

        if args.quiet:
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        batch_size = int(args.batch_size) if args.batch_size is not None else 128

        experiment_enabled = bool(getattr(args, "hle_experiment", False))
        if experiment_enabled:
            modes = ["single", "debate"]
            infer_modes = list(modes)
            if str(args.dataset).strip().lower() != "hle":
                print("--hle_experiment currently supports only --dataset hle", file=status_file)
                sys.exit(2)
            if args.output:
                print("--output is not supported with --hle_experiment because it writes multiple arm outputs", file=status_file)
                sys.exit(2)
            if bool(args.use_personas):
                print("--use_personas is managed internally by --hle_experiment", file=status_file)
                sys.exit(2)
            if str(args.persona_stage) != "full":
                print("--persona_stage is not supported with --hle_experiment; use --hle_experiment_stop_after for staged runs", file=status_file)
                sys.exit(2)
            if getattr(args, "debate_stop_after", None) is not None:
                print("--debate_stop_after is not supported with --hle_experiment; use --hle_experiment_stop_after for staged runs", file=status_file)
                sys.exit(2)
            if getattr(args, "hle_experiment_stop_after", None) is not None and not args.stage_state_file:
                print("--hle_experiment_stop_after requires --stage_state_file so the experiment can be resumed", file=status_file)
                sys.exit(2)
            if bool(args.final_run):
                print("--final_run is not supported with --hle_experiment", file=status_file)
                sys.exit(2)
        else:
            modes = []
            for m in args.mode.split(","):
                m = m.strip().lower()
                if m in ("single", "majority", "debate", "personas"):
                    if m not in modes:
                        modes.append(m)
            if not modes:
                print("No valid modes specified", file=status_file)
                sys.exit(1)
            infer_modes = [m for m in modes if m in ("single", "majority", "debate")]
            if "personas" in modes and any(m in infer_modes for m in ("single", "majority")):
                print("personas mode can only be combined with debate", file=status_file)
                sys.exit(2)
            if bool(args.use_personas) and not any(m in modes for m in ("debate", "majority")):
                print("--use_personas currently applies to debate or majority mode only", file=status_file)
                sys.exit(2)
            if "personas" in modes and "debate" in modes:
                args.use_personas = True
                modes = [m for m in modes if m not in {"personas", "debate"}]
                modes = ["personas", "debate", *modes]
            if (bool(args.persona_replay) or bool(args.persona_save_artifacts)) and "personas" not in modes and not bool(args.use_personas):
                print("persona replay/save flags require either personas mode or --use_personas", file=status_file)
                sys.exit(2)
            if args.output and len(modes) != 1:
                print("--output requires exactly one mode", file=status_file)
                sys.exit(2)
        if infer_modes and not args.model_name:
            print("--model_name is required for single/majority/debate modes", file=status_file)
            sys.exit(2)
        if str(args.persona_axes_mode) == "replay" and not bool(args.persona_replay):
            print("--persona_axes_mode replay requires --persona_replay", file=status_file)
            sys.exit(2)
        if str(args.persona_axes_mode) == "file" and not args.persona_axes_file:
            print("--persona_axes_mode file requires --persona_axes_file", file=status_file)
            sys.exit(2)
        if bool(args.persona_replay) and str(args.persona_axes_mode) != "replay":
            args.persona_axes_mode = "replay"
        if args.persona_seed is None:
            args.persona_seed = int(args.subset_seed) if args.subset_seed is not None else 0
        if int(args.public_rationale_max_tokens) <= 0:
            print("--public_rationale_max_tokens must be > 0", file=status_file)
            sys.exit(2)
        if str(getattr(args, "debate_protocol", "legacy")) == "structured" and int(args.n_rounds) != 2:
            print("--debate_protocol structured currently requires --n_rounds 2 (for 3 total answer rounds)", file=status_file)
            sys.exit(2)
        persona_plain_agents = int(getattr(args, "persona_plain_agents", 0) or 0)
        persona_backed_debate_enabled = bool(experiment_enabled or ("debate" in modes and bool(args.use_personas)))
        if persona_plain_agents < 0:
            print("--persona_plain_agents must be >= 0", file=status_file)
            sys.exit(2)
        if persona_plain_agents > 0:
            if persona_plain_agents >= int(args.n_agents):
                print("--persona_plain_agents must be less than --n_agents so at least one persona slot remains", file=status_file)
                sys.exit(2)
            if not persona_backed_debate_enabled:
                print("--persona_plain_agents currently applies only to persona-backed debate or --hle_experiment", file=status_file)
                sys.exit(2)
        if bool(args.final_run) and not args.final_manifest:
            print("--final_run requires --final_manifest", file=status_file)
            sys.exit(2)
        raw_generator_model_name = str(getattr(args, "generator_model", "") or "").strip() or None
        raw_judge_generator_model_name = str(getattr(args, "judge_generator_model", "") or "").strip() or None
        raw_judge_runtime_model_name = str(getattr(args, "judge_runtime_model", "") or "").strip() or None
        args.model_name = normalize_gemini_model_name(getattr(args, "model_name", None))
        requested_generator_model_name = raw_generator_model_name or (args.model_name if infer_modes else None)
        persona_backend = _effective_persona_backend(
            requested_backend=str(args.persona_backend),
            generator_model_name=requested_generator_model_name,
        )
        generator_model_name = (
            normalize_gemini_model_name(raw_generator_model_name or args.model_name)
            if persona_backend == "llm"
            else None
        )
        judge_generator_model_name = (
            normalize_gemini_model_name(raw_judge_generator_model_name or generator_model_name)
            if persona_backend == "llm"
            else None
        )
        judge_runtime_model_name = (
            normalize_gemini_model_name(raw_judge_runtime_model_name or args.model_name)
            if "debate" in modes
            else None
        )
        args.generator_model = generator_model_name
        args.judge_generator_model = judge_generator_model_name
        args.judge_runtime_model = judge_runtime_model_name
        try:
            provider_name = infer_provider_name(args.model_name, args.provider) if args.model_name else "gemini"
            generator_provider_name = infer_provider_name(generator_model_name, args.generator_provider) if generator_model_name else "gemini"
            judge_provider_name = infer_provider_name(judge_runtime_model_name, args.judge_provider) if judge_runtime_model_name else "gemini"
            judge_generator_provider_name = infer_provider_name(judge_generator_model_name, args.judge_generator_provider) if judge_generator_model_name else "gemini"
        except ValueError as exc:
            print(str(exc), file=status_file)
            sys.exit(2)
        if persona_backend == "llm" and not generator_model_name:
            print("persona llm backend requires --generator_model or --model_name", file=status_file)
            sys.exit(2)

        dataset: DatasetName = cast(DatasetName, args.dataset)
        subset_seed = args.subset_seed if args.subset_seed is not None else random.SystemRandom().randint(0, 2**32 - 1)

        ids = None
        subset_range = args.subset_range
        if args.subset_ids:
            sids = str(args.subset_ids).strip()
            if sids.lower() in ("all", "*"):
                subset_range = "all"
            else:
                ids = [int(x.strip()) for x in sids.split(",") if x.strip()]

        if args.all or args.subset_n == "all":
            subset_range = "all"
            ids = None
        elif subset_range and str(subset_range).strip().lower() in ("all", "*"):
            subset_range = "all"
            ids = None
        if args.one:
            subset_range = "all"
            ids = None

        exclude_ids_path = Path(args.exclude_ids) if args.exclude_ids else None
        exclude_id_map = _load_exclude_id_map(exclude_ids_path)
        dataset_local_mirror = Path(args.dataset_local_mirror) if args.dataset_local_mirror else None
        test_path = _default_dataset_test_path(
            dataset,
            hle_variant=str(args.hle_variant),
            dataset_local_mirror=dataset_local_mirror,
        )

        items, meta = _make_dataset_subset(
            dataset=dataset,
            test_path=test_path,
            n=0 if args.subset_n == "all" else int(args.subset_n),
            seed=subset_seed,
            ids=ids,
            range_str=subset_range,
            hle_variant=str(args.hle_variant),
            hle_modality=str(args.hle_modality),
            exclude_id_map=exclude_id_map,
            exclude_ids_path=exclude_ids_path,
        )
        if not args.quiet:
            print(f"[data] Subset: {len(items)} items from {dataset}", file=sys.stderr)
        if args.one:
            items = _select_one_item(items, str(args.one))
            meta["subset_size"] = len(items)
            meta["selected_one"] = str(args.one)
            if not args.quiet:
                print(f"[data] Selected single item via --one: {args.one}", file=sys.stderr)

        sampling_config = None
        judge_runtime_sampling_config = None
        if infer_modes:
            assert args.model_name is not None
            sampling_config = build_sampling_config(args.model_name)
            set_sampling_config(sampling_config)
            if judge_runtime_model_name:
                judge_runtime_sampling_config = (
                    sampling_config
                    if judge_runtime_model_name == args.model_name
                    else build_sampling_config(judge_runtime_model_name)
                )

        engine: InferenceEngine | None = None
        judge_runtime_engine: InferenceEngine | None = None
        persona_generator_engine: InferenceEngine | None = None
        persona_judge_engine: InferenceEngine | None = None
        results: dict[str, list[dict[str, Any]]] = {}
        cost_tracker: CostTracker | None = None
        cost_limit_exc: SpendLimitExceeded | None = None

        with _DoubleCtrlCHandler(timeout=2.0, output_file=status_file):
            try:
                out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(dataset)
                out_dir.mkdir(parents=True, exist_ok=True)
                dataset_tag = _dataset_tag(dataset)
                model_tag = _model_tag(args.model_name or "persona")
                persona_axes_path = Path(args.persona_axes_file) if args.persona_axes_file else None
                judge_bank_path = Path(args.judge_bank_dir) if args.judge_bank_dir else None
                gpqa_family_cache_path = Path(args.gpqa_family_cache_path) if args.gpqa_family_cache_path else None
                cli_stage_state_path = Path(args.stage_state_file) if args.stage_state_file else None
                experiment_judge_rounds = _parse_judge_rounds(args.debate_judge_rounds, args.n_rounds)
                experiment_judge_sampling_kwargs: dict[str, Any] | None = None
                if args.judge_max_tokens is not None or args.judge_temperature is not None:
                    main_sampling = judge_runtime_sampling_config or sampling_config
                    experiment_judge_sampling_kwargs = {
                        "max_tokens": int(args.judge_max_tokens) if args.judge_max_tokens is not None else (
                            int(main_sampling.max_tokens) if main_sampling and main_sampling.max_tokens is not None else None
                        ),
                    }
                    if args.judge_temperature is not None:
                        experiment_judge_sampling_kwargs["temperature"] = float(args.judge_temperature)
                if subset_range and str(subset_range).strip().lower() == "all":
                    subset_spec_tag = "all"
                else:
                    subset_spec_tag = _ids_tag(ids) if ids else _range_tag(subset_range)
                resume_settings = _hle_experiment_resume_settings(
                    dataset=dataset,
                    items=items,
                    hle_variant=str(args.hle_variant),
                    hle_modality=str(args.hle_modality),
                    model_name=args.model_name,
                    provider_name=provider_name,
                    generator_model_name=generator_model_name,
                    generator_provider_name=generator_provider_name,
                    judge_runtime_model_name=judge_runtime_model_name,
                    judge_provider_name=judge_provider_name,
                    judge_generator_model_name=judge_generator_model_name,
                    judge_generator_provider_name=judge_generator_provider_name,
                    persona_backend=persona_backend,
                    n_agents=int(args.n_agents),
                    n_rounds=int(args.n_rounds),
                    judge_rounds=experiment_judge_rounds,
                    judge_block_size=args.judge_block_size,
                    judge_sampling_kwargs=experiment_judge_sampling_kwargs,
                    judge_strict_final_only=bool(args.judge_strict_final_only),
                    judge_recovery_parse_enabled=bool(args.judge_recovery_parse_enabled),
                    persona_seed=int(args.persona_seed),
                    persona_axes_mode=str(args.persona_axes_mode),
                    persona_fixed_axis_count=int(args.persona_fixed_axis_count),
                    persona_task_axis_count=int(args.persona_task_axis_count),
                    persona_sampling_method=str(args.persona_sampling_method),
                    judge_persona_mode=str(args.judge_persona_mode),
                    persona_axes_file=persona_axes_path,
                    judge_bank_dir=judge_bank_path,
                    judge_bank_refresh=bool(args.judge_bank_refresh),
                    gpqa_family_cache_path=gpqa_family_cache_path,
                    judge_trace_mode=str(args.judge_trace_mode),
                    public_rationale_max_tokens=int(args.public_rationale_max_tokens),
                    persona_save_artifacts=bool(args.persona_save_artifacts),
                    persona_replay=bool(args.persona_replay),
                    emit_trace_level=str(args.emit_trace_level),
                    debate_protocol=str(args.debate_protocol),
                    persona_plain_agents=persona_plain_agents,
                ) if experiment_enabled else None
                experiment_resume_state = _load_hle_experiment_resume_state(
                    stage_state_file=cli_stage_state_path,
                    resume_settings=cast(dict[str, Any], resume_settings) if resume_settings is not None else {},
                ) if experiment_enabled else None
                if experiment_resume_state is not None:
                    run_tag = str(experiment_resume_state["run_tag"])
                    experiment_root = Path(str(experiment_resume_state["experiment_root"]))
                    manifest_path = Path(str(experiment_resume_state["manifest_path"]))
                    manifest = (
                        _build_hle_experiment_manifest(
                            manifest_path=manifest_path,
                            experiment_root=experiment_root,
                            run_tag=run_tag,
                            dataset=str(dataset),
                            meta=meta,
                            item_uids=[item.item_uid for item in items],
                            stage_state_file=cli_stage_state_path,
                            model_name=args.model_name,
                            generator_model_name=generator_model_name,
                            judge_runtime_model_name=judge_runtime_model_name,
                                judge_generator_model_name=judge_generator_model_name,
                                persona_backend=persona_backend,
                            n_agents=int(args.n_agents),
                            n_rounds=int(args.n_rounds),
                            emit_trace_level=str(args.emit_trace_level),
                            hle_variant=str(args.hle_variant),
                            hle_modality=str(args.hle_modality),
                            resume_settings=cast(dict[str, Any], resume_settings) if resume_settings is not None else None,
                        )
                        if not manifest_path.exists()
                        else None
                    )
                else:
                    ts = _timestamp_tag()
                    run_tag = _build_run_tag(tag=args.tag, meta=meta, subset_spec_tag=subset_spec_tag, timestamp_tag=ts)
                    experiment_root = default_experiment_root(out_dir=out_dir, run_tag=run_tag) if experiment_enabled else out_dir
                    manifest_path = (
                        Path(args.final_manifest)
                        if args.final_manifest
                        else (_default_hle_experiment_manifest_path(experiment_root=experiment_root) if experiment_enabled else None)
                    )
                    manifest = None
                experiment_root.mkdir(parents=True, exist_ok=True)
                if experiment_enabled:
                    persona_artifacts_dir = (
                        Path(args.persona_artifacts_dir)
                        if args.persona_artifacts_dir
                        else (experiment_root / "persona_artifacts")
                    )
                    if manifest is None:
                        manifest = (
                            load_manifest(manifest_path)
                            if manifest_path.exists()
                            else _build_hle_experiment_manifest(
                                manifest_path=manifest_path,
                                experiment_root=experiment_root,
                                run_tag=run_tag,
                                dataset=str(dataset),
                                meta=meta,
                                item_uids=[item.item_uid for item in items],
                                stage_state_file=cli_stage_state_path,
                                model_name=args.model_name,
                                generator_model_name=generator_model_name,
                                judge_runtime_model_name=judge_runtime_model_name,
                                judge_generator_model_name=judge_generator_model_name,
                                persona_backend=persona_backend,
                                n_agents=int(args.n_agents),
                                n_rounds=int(args.n_rounds),
                                emit_trace_level=str(args.emit_trace_level),
                                hle_variant=str(args.hle_variant),
                                hle_modality=str(args.hle_modality),
                                resume_settings=cast(dict[str, Any], resume_settings) if resume_settings is not None else None,
                            )
                        )
                    _write_hle_experiment_manifest(path=manifest_path, manifest=manifest)
                else:
                    persona_artifacts_dir = Path(args.persona_artifacts_dir) if args.persona_artifacts_dir else (out_dir / "persona_artifacts")
                    manifest_path = Path(args.final_manifest) if args.final_manifest else None
                    manifest = _build_final_manifest(
                        args=args,
                        dataset=dataset,
                        meta=meta,
                        items=items,
                        provider_name=provider_name,
                        generator_provider_name=generator_provider_name,
                        judge_provider_name=judge_provider_name,
                        judge_generator_provider_name=judge_generator_provider_name,
                        generator_model_name=generator_model_name,
                        judge_generator_model_name=judge_generator_model_name,
                        judge_runtime_model_name=judge_runtime_model_name,
                        persona_backend=persona_backend,
                        modes=modes,
                        emit_trace_level=str(args.emit_trace_level),
                    )
                    if manifest_path is not None:
                        _write_or_validate_final_manifest(
                            manifest_path=manifest_path,
                            manifest=manifest,
                            final_run=bool(args.final_run),
                        )
                ledger_path = Path(args.token_ledger_path) if args.token_ledger_path else _default_token_ledger_path()
                cost_tracker = CostTracker(
                    ledger_path=ledger_path,
                    session_name=run_tag,
                    session_meta={
                        "cli_name": "debate-v-majority",
                        "dataset": dataset,
                        "modes": ["hle_experiment"] if experiment_enabled else list(modes),
                        "run_tag": run_tag,
                        "debater_model": args.model_name,
                        "debater_provider": provider_name,
                        "judge_runtime_model": judge_runtime_model_name,
                        "judge_provider": judge_provider_name,
                        "generator_model": generator_model_name,
                        "generator_provider": generator_provider_name,
                        "judge_generator_model": judge_generator_model_name,
                        "judge_generator_provider": judge_generator_provider_name,
                    },
                    max_run_cost_usd=args.max_run_cost_usd,
                    max_total_cost_usd=args.max_total_cost_usd,
                )
                cost_tracking_ctx = active_cost_tracking(cost_tracker)
                cost_tracking_ctx.__enter__()
                try:

                    if infer_modes:
                        if not args.quiet:
                            print(f"[engine] Creating inference engine for {GEMINI_3_FLASH_MODEL}...", file=sys.stderr)
                        assert args.model_name is not None
                        engine = _create_role_engine(
                            model_name=args.model_name,
                            provider=provider_name,
                            model_role="debater",
                            context_len=args.context_len,
                        )
                        if judge_runtime_model_name:
                            judge_runtime_engine = _reuse_or_create_role_engine(
                                existing_engine=engine,
                                existing_model_name=getattr(engine, "model_name", None),
                                existing_provider_name=provider_name,
                                target_model_name=judge_runtime_model_name,
                                target_provider_name=judge_provider_name,
                                model_role="judge",
                                context_len=args.context_len,
                                quiet=bool(args.quiet),
                                status_label="judge runtime",
                            )
                    if (experiment_enabled or "personas" in modes or bool(args.use_personas)) and persona_backend == "llm":
                        assert generator_model_name is not None
                        persona_generator_engine = _reuse_or_create_role_engine(
                            existing_engine=engine,
                            existing_model_name=getattr(engine, "model_name", None) if engine is not None else None,
                            existing_provider_name=provider_name if engine is not None else None,
                            target_model_name=generator_model_name,
                            target_provider_name=generator_provider_name,
                            model_role="generator",
                            context_len=args.context_len,
                            quiet=bool(args.quiet),
                            status_label="generator",
                        )
                        if (
                            judge_generator_model_name == generator_model_name
                            and persona_generator_engine is not None
                        ):
                            persona_judge_engine = persona_generator_engine
                        else:
                            persona_judge_engine = _reuse_or_create_role_engine(
                                existing_engine=engine,
                                existing_model_name=getattr(engine, "model_name", None) if engine is not None else None,
                                existing_provider_name=provider_name if engine is not None else None,
                                target_model_name=cast(str, judge_generator_model_name),
                                target_provider_name=judge_generator_provider_name,
                                model_role="judge_generator",
                                context_len=args.context_len,
                                quiet=bool(args.quiet),
                                status_label="judge generator",
                            )

                    persona_generation_n = (
                        int(args.n_agents) - persona_plain_agents
                        if ("personas" in modes and "debate" in modes)
                        else int(args.persona_n)
                    )
                    persona_generation_kwargs = {
                        "dataset": dataset,
                        "items": items,
                        "artifacts_dir": persona_artifacts_dir,
                        "n_personas": persona_generation_n,
                        "persona_seed": int(args.persona_seed),
                        "axis_mode": str(args.persona_axes_mode),
                        "fixed_axis_count": int(args.persona_fixed_axis_count),
                        "task_axis_count": int(args.persona_task_axis_count),
                        "sampling_method": str(args.persona_sampling_method),
                        "judge_persona_mode": str(args.judge_persona_mode),
                        "backend": persona_backend,
                        "generator_model": generator_model_name,
                        "judge_generator_model": judge_generator_model_name,
                        "generator_engine": persona_generator_engine,
                        "judge_engine": persona_judge_engine,
                        "axes_file": persona_axes_path,
                        "judge_bank_dir": judge_bank_path,
                        "judge_bank_refresh": bool(args.judge_bank_refresh),
                        "gpqa_family_cache_path": gpqa_family_cache_path,
                        "n_plain_agents": persona_plain_agents,
                    }
                    experiment_persona_generation_kwargs = {
                        **persona_generation_kwargs,
                        "n_personas": int(args.n_agents) - persona_plain_agents,
                        "n_plain_agents": persona_plain_agents,
                    }

                    if experiment_enabled:
                        assert manifest_path is not None
                        experiment_manifest = cast(dict[str, Any], manifest)
                        completed_arm_names = set(_completed_hle_experiment_arms(experiment_manifest))
                        if completed_arm_names == set(HLE_EXPERIMENT_ARMS):
                            print("[experiment] All experiment arms are already complete.", file=status_file)
                            return
                        judge_sampling_kwargs = experiment_judge_sampling_kwargs
                        interactive_experiment = bool(sys.stdin.isatty()) and getattr(args, "hle_experiment_stop_after", None) is None

                        def _experiment_arm_stage_state_file(arm_name: str) -> Path:
                            return arm_results_path(experiment_root=experiment_root, arm_name=arm_name).parent / "stage_state.jsonl"

                        def _run_experiment_debate_arm(
                            *,
                            arm_name: str,
                            use_personas: bool,
                            runtime_judge_persona_enabled: bool,
                            persona_artifacts_by_item: dict[str, Any] | None = None,
                        ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
                            from .debate_runner import _DebateStopped

                            arm_cost_before = _cost_summary_snapshot(cost_tracker)
                            arm_stage_state_file = _experiment_arm_stage_state_file(arm_name)
                            prev_debate_entry = (
                                load_latest_stage_entry_of_type(arm_stage_state_file, "debate")
                                if arm_stage_state_file.exists()
                                else None
                            )
                            debate_stage_state_file = (
                                arm_stage_state_file
                                if (interactive_experiment or arm_stage_state_file.exists())
                                else None
                            )
                            current_debate_stop_after: str | None = None
                            if interactive_experiment:
                                if prev_debate_entry is None:
                                    current_debate_stop_after = "round_0"
                                else:
                                    current_debate_stop_after = _next_debate_stage(
                                        completed_stage=str(prev_debate_entry.completed_stage),
                                        n_rounds=int(args.n_rounds),
                                        judge_rounds=experiment_judge_rounds,
                                    )

                            while True:
                                debate_cost_before = _cost_summary_snapshot(cost_tracker)
                                try:
                                    arm_results_by_round = run_debate(
                                        dataset=dataset,
                                        items=items,
                                        engine=cast(InferenceEngine, engine),
                                        n_agents=args.n_agents,
                                        n_rounds=args.n_rounds,
                                        judge_rounds=experiment_judge_rounds,
                                        batch_size=batch_size,
                                        judge_block_size=args.judge_block_size,
                                        judge_sampling_kwargs=judge_sampling_kwargs,
                                        judge_strict_final_only=bool(args.judge_strict_final_only),
                                        judge_recovery_parse_enabled=bool(args.judge_recovery_parse_enabled),
                                        judge_engine=judge_runtime_engine,
                                        use_personas=use_personas,
                                        artifacts_dir=persona_artifacts_dir if use_personas else None,
                                        persona_seed=int(args.persona_seed),
                                        persona_axis_mode=str(args.persona_axes_mode),
                                        persona_fixed_axis_count=int(args.persona_fixed_axis_count),
                                        persona_task_axis_count=int(args.persona_task_axis_count),
                                        persona_sampling_method=str(args.persona_sampling_method),
                                        persona_judge_mode=str(args.judge_persona_mode),
                                        persona_backend=persona_backend,
                                        generator_model=generator_model_name,
                                        judge_generator_model=judge_generator_model_name,
                                        persona_generator_engine=persona_generator_engine,
                                        persona_judge_engine=persona_judge_engine,
                                        persona_axes_file=persona_axes_path,
                                        persona_save_artifacts=bool(args.persona_save_artifacts) if use_personas else False,
                                        persona_replay=bool(args.persona_replay) if use_personas else False,
                                        judge_trace_mode=str(args.judge_trace_mode),
                                        judge_bank_dir=judge_bank_path,
                                        judge_bank_refresh=bool(args.judge_bank_refresh),
                                        gpqa_family_cache_path=gpqa_family_cache_path,
                                        public_rationale_max_tokens=int(args.public_rationale_max_tokens),
                                    debate_protocol=str(args.debate_protocol),
                                        enable_runtime_judge_persona=runtime_judge_persona_enabled,
                                        persona_artifacts_by_item=persona_artifacts_by_item,
                                        progress_file=progress_file,
                                        debate_stop_after=current_debate_stop_after,
                                        stage_state_file=debate_stage_state_file,
                                        persona_plain_agents=persona_plain_agents if use_personas else 0,
                                    )
                                except _DebateStopped:
                                    completed_stage = str(current_debate_stop_after)
                                    _print_stage_cost_summary(
                                        label=f"{arm_name}:{completed_stage}",
                                        before=debate_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                    print(
                                        f"[staged] Debate stopped after '{completed_stage}'. State: {arm_stage_state_file}",
                                        file=status_file,
                                    )
                                    next_stage = _next_debate_stage(
                                        completed_stage=completed_stage,
                                        n_rounds=int(args.n_rounds),
                                        judge_rounds=experiment_judge_rounds,
                                    )
                                    if next_stage is None:
                                        return [], [], []
                                    if not _prompt_continue_debate_stage(next_stage=next_stage, status_file=status_file):
                                        print(
                                            f"[staged] Debate staging stopped after '{completed_stage}'. State: {arm_stage_state_file}",
                                            file=status_file,
                                        )
                                        return [], [], []
                                    current_debate_stop_after = next_stage
                                    continue
                                break

                            _print_stage_cost_summary(
                                label=arm_name,
                                before=arm_cost_before,
                                cost_tracker=cost_tracker,
                                status_file=status_file,
                            )
                            final_round = int(args.n_rounds) if args.debate_judge_rounds is None else max(experiment_judge_rounds)
                            arm_records = arm_results_by_round[final_round]
                            return _write_hle_experiment_arm_outputs(
                                records=arm_records,
                                arm_name=arm_name,
                                experiment_root=experiment_root,
                                manifest_path=manifest_path,
                                run_tag=run_tag,
                                dataset=dataset,
                                meta=meta,
                                mode="debate",
                                use_personas=use_personas,
                                runtime_judge_persona_enabled=runtime_judge_persona_enabled,
                                emit_trace_level=str(args.emit_trace_level),
                                judge_trace_mode=str(args.judge_trace_mode),
                                public_rationale_max_tokens=int(args.public_rationale_max_tokens),
                                results_by_round=arm_results_by_round,
                                judge_rounds=experiment_judge_rounds,
                            )

                        for arm_name in HLE_EXPERIMENT_ARMS:
                            if arm_name in completed_arm_names:
                                continue
                            if not args.quiet:
                                print(f"\n[experiment] Running {arm_name}...", file=sys.stderr)
                            if arm_name == "single":
                                single_cost_before = _cost_summary_snapshot(cost_tracker)
                                arm_records = run_sampled(
                                    dataset=dataset,
                                    items=items,
                                    engine=cast(InferenceEngine, engine),
                                    n_samples=1,
                                    batch_size=batch_size,
                                    mode_label="single",
                                    progress_file=progress_file,
                                )
                                output_rows, additional_paths, trace_paths = _write_hle_experiment_arm_outputs(
                                    records=arm_records,
                                    arm_name=arm_name,
                                    experiment_root=experiment_root,
                                    manifest_path=manifest_path,
                                    run_tag=run_tag,
                                    dataset=dataset,
                                    meta=meta,
                                    mode="single",
                                    use_personas=False,
                                    runtime_judge_persona_enabled=False,
                                    emit_trace_level=str(args.emit_trace_level),
                                    judge_trace_mode=None,
                                    public_rationale_max_tokens=None,
                                )
                                _print_stage_cost_summary(
                                    label="single",
                                    before=single_cost_before,
                                    cost_tracker=cost_tracker,
                                    status_file=status_file,
                                )
                            elif arm_name == "debate_plain":
                                output_rows, additional_paths, trace_paths = _run_experiment_debate_arm(
                                    arm_name=arm_name,
                                    use_personas=False,
                                    runtime_judge_persona_enabled=False,
                                )
                                if not output_rows and interactive_experiment:
                                    return
                            else:
                                arm_stage_state_file = _experiment_arm_stage_state_file(arm_name)
                                persona_rows: list[dict[str, Any]] | None = None
                                persona_generation_cost_before = _cost_summary_snapshot(cost_tracker)
                                persona_generation_executed = False
                                prev_persona_entry = (
                                    load_latest_stage_entry_of_type(arm_stage_state_file, "persona")
                                    if arm_stage_state_file.exists()
                                    else None
                                )
                                current_stage: str | None = None
                                if prev_persona_entry is None:
                                    if interactive_experiment:
                                        current_stage = "axes"
                                else:
                                    current_stage = _next_persona_stage(str(prev_persona_entry.completed_stage))
                                    if current_stage is None:
                                        persona_rows = persona_rows_from_stage_entry(prev_persona_entry)
                                if current_stage is not None:
                                    persona_generation_executed = True
                                    while current_stage is not None:
                                        stage_cost_before = _cost_summary_snapshot(cost_tracker)
                                        entry = run_persona_generation_staged(
                                            **experiment_persona_generation_kwargs,
                                            stage_state_file=arm_stage_state_file,
                                            persona_stage=current_stage,
                                            save_artifacts=bool(args.persona_save_artifacts),
                                            replay=bool(args.persona_replay),
                                            dump_cards=bool(args.persona_dump_cards),
                                            summary_file=status_file,
                                        )
                                        completed_stage = str(entry.completed_stage)
                                        print(
                                            f"[staged] Persona stage '{completed_stage}' complete. State: {arm_stage_state_file}",
                                            file=status_file,
                                        )
                                        _print_stage_cost_summary(
                                            label=f"{arm_name}:persona:{completed_stage}",
                                            before=stage_cost_before,
                                            cost_tracker=cost_tracker,
                                            status_file=status_file,
                                        )
                                        next_stage = _next_persona_stage(completed_stage)
                                        if next_stage is None:
                                            persona_rows = persona_rows_from_stage_entry(entry)
                                            break
                                        if interactive_experiment and not _prompt_continue_persona_stage(next_stage=next_stage, status_file=status_file):
                                            print(
                                                f"[staged] Persona staging stopped after '{completed_stage}'. State: {arm_stage_state_file}",
                                                file=status_file,
                                            )
                                            return
                                        current_stage = next_stage
                                elif persona_rows is None:
                                    persona_generation_executed = True
                                    persona_rows = run_persona_generation(
                                        **experiment_persona_generation_kwargs,
                                        save_artifacts=bool(args.persona_save_artifacts),
                                        replay=bool(args.persona_replay),
                                        dump_cards=bool(args.persona_dump_cards),
                                        summary_file=status_file,
                                    )
                                assert persona_rows is not None
                                if persona_generation_executed:
                                    _print_stage_cost_summary(
                                        label="persona_generation",
                                        before=persona_generation_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                generated_persona_artifacts = persona_artifacts_from_rows(persona_rows)
                                if interactive_experiment:
                                    prev_debate_entry = (
                                        load_latest_stage_entry_of_type(arm_stage_state_file, "debate")
                                        if arm_stage_state_file.exists()
                                        else None
                                    )
                                    if prev_debate_entry is None:
                                        if not _prompt_continue_to_debate(status_file=status_file):
                                            print("[staged] Stopped after personas. Progress has been saved.", file=status_file)
                                            return
                                output_rows, additional_paths, trace_paths = _run_experiment_debate_arm(
                                    arm_name=arm_name,
                                    use_personas=True,
                                    runtime_judge_persona_enabled=True,
                                    persona_artifacts_by_item=generated_persona_artifacts,
                                )
                                if not output_rows and interactive_experiment:
                                    return
                            results[arm_name] = output_rows
                            if not args.quiet:
                                acc = _accuracy(output_rows)
                                print(
                                    f"[result] {arm_name}: {acc*100:.1f}% ({sum(r['final_correct'] for r in output_rows)}/{len(output_rows)})",
                                    file=sys.stderr,
                                )
                            print(f"[output] Written to {arm_results_path(experiment_root=experiment_root, arm_name=arm_name)}", file=status_file)
                            experiment_manifest = update_manifest_for_arm(
                                manifest=experiment_manifest,
                                arm_name=arm_name,
                                rows=output_rows,
                                trace_paths=trace_paths,
                                additional_output_paths=additional_paths,
                            )
                            _write_hle_experiment_manifest(path=manifest_path, manifest=experiment_manifest)
                            completed_arm_names.add(arm_name)
                            _append_hle_experiment_stage(
                                stage_state_file=cli_stage_state_path,
                                dataset=dataset,
                                items=items,
                                completed_stage=arm_name,
                                experiment_state={
                                    "run_tag": run_tag,
                                    "experiment_root": str(experiment_root),
                                    "manifest_path": str(manifest_path),
                                    "completed_arms": sorted(completed_arm_names),
                                },
                                resume_settings=cast(dict[str, Any], resume_settings),
                            )
                            if str(getattr(args, "hle_experiment_stop_after", "") or "") == arm_name:
                                print(f"[experiment] Stopped after {arm_name}. Resume with the same --stage_state_file.", file=status_file)
                                return
                            if interactive_experiment:
                                remaining_arms = [name for name in HLE_EXPERIMENT_ARMS if name not in completed_arm_names]
                                if remaining_arms:
                                    next_arm = remaining_arms[0]
                                    if not _prompt_continue_experiment_arm(next_arm=next_arm, status_file=status_file):
                                        print(
                                            f"[staged] Experiment staging stopped after '{arm_name}'. Resume with the same --stage_state_file.",
                                            file=status_file,
                                        )
                                        return
                        return

                    ran_staged_personas = False
                    generated_persona_artifacts: dict[str, Any] | None = None
                    for mode in modes:
                        results_by_round: dict[int, list[dict[str, Any]]] | None = None
                        judge_rounds: list[int] | None = None
                        if not args.quiet:
                            print(f"\n[run] Running {mode} mode...", file=sys.stderr)

                        if mode == "personas":
                            persona_stage = str(args.persona_stage)
                            stage_state_path = cli_stage_state_path
                            if persona_stage != "full":
                                if stage_state_path is None:
                                    stage_state_path = out_dir / f"stage_state_{dataset_tag}_{run_tag}.jsonl"
                                current_stage = persona_stage
                                records = None
                                while True:
                                    stage_cost_before = _cost_summary_snapshot(cost_tracker)
                                    entry = run_persona_generation_staged(
                                        **persona_generation_kwargs,
                                        stage_state_file=stage_state_path,
                                        persona_stage=current_stage,
                                        save_artifacts=bool(args.persona_save_artifacts),
                                        replay=bool(args.persona_replay),
                                        dump_cards=bool(args.persona_dump_cards),
                                        summary_file=status_file,
                                    )
                                    completed_stage = str(entry.completed_stage)
                                    print(
                                        f"[staged] Persona stage '{completed_stage}' complete. State: {stage_state_path}",
                                        file=status_file,
                                    )
                                    _print_stage_cost_summary(
                                        label=f"persona:{completed_stage}",
                                        before=stage_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                    next_stage = _next_persona_stage(completed_stage)
                                    if next_stage is None:
                                        records = persona_rows_from_stage_entry(entry)
                                        break
                                    if not sys.stdin.isatty():
                                        break
                                    if not _prompt_continue_persona_stage(next_stage=next_stage, status_file=status_file):
                                        print(
                                            f"[staged] Persona staging stopped after '{completed_stage}'. State: {stage_state_path}",
                                            file=status_file,
                                        )
                                        break
                                    current_stage = next_stage
                                if records is None:
                                    return
                                ran_staged_personas = True
                            else:
                                if stage_state_path is None:
                                    stage_state_path = out_dir / f"stage_state_{dataset_tag}_{run_tag}.jsonl"
                                current_stage = _PERSONA_STAGE_ORDER[0]
                                records = None
                                while True:
                                    stage_cost_before = _cost_summary_snapshot(cost_tracker)
                                    entry = run_persona_generation_staged(
                                        **persona_generation_kwargs,
                                        stage_state_file=stage_state_path,
                                        persona_stage=current_stage,
                                        save_artifacts=bool(args.persona_save_artifacts),
                                        replay=bool(args.persona_replay),
                                        dump_cards=bool(args.persona_dump_cards),
                                        summary_file=status_file,
                                    )
                                    completed_stage = str(entry.completed_stage)
                                    _print_stage_cost_summary(
                                        label=f"persona:{completed_stage}",
                                        before=stage_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                    next_stage = _next_persona_stage(completed_stage)
                                    if next_stage is None:
                                        records = persona_rows_from_stage_entry(entry)
                                        break
                                    current_stage = next_stage
                            generated_persona_artifacts = persona_artifacts_from_rows(records)
                        elif mode == "single":
                            records = run_sampled(
                                dataset=dataset,
                                items=items,
                                engine=engine,
                                n_samples=1,
                                batch_size=batch_size,
                                mode_label="single",
                                progress_file=progress_file,
                            )
                        elif mode == "majority":
                            records = run_sampled(
                                dataset=dataset,
                                items=items,
                                engine=engine,
                                n_samples=args.majority_samples,
                                batch_size=batch_size,
                                mode_label="majority",
                                use_personas=bool(args.use_personas),
                                artifacts_dir=persona_artifacts_dir,
                                persona_seed=int(args.persona_seed),
                                persona_axis_mode=str(args.persona_axes_mode),
                                persona_fixed_axis_count=int(args.persona_fixed_axis_count),
                                persona_task_axis_count=int(args.persona_task_axis_count),
                                persona_sampling_method=str(args.persona_sampling_method),
                                persona_judge_mode=str(args.judge_persona_mode),
                                persona_backend=persona_backend,
                                generator_model=generator_model_name,
                                judge_generator_model=judge_generator_model_name,
                                persona_generator_engine=persona_generator_engine,
                                persona_judge_engine=persona_judge_engine,
                                persona_axes_file=persona_axes_path,
                                persona_save_artifacts=bool(args.persona_save_artifacts),
                                persona_replay=bool(args.persona_replay),
                                judge_bank_dir=judge_bank_path,
                                judge_bank_refresh=bool(args.judge_bank_refresh),
                                gpqa_family_cache_path=gpqa_family_cache_path,
                                progress_file=progress_file,
                            )
                        elif mode == "debate":
                            from .debate_runner import _DebateStopped

                            judge_rounds = _parse_judge_rounds(args.debate_judge_rounds, args.n_rounds)
                            try:
                                _validate_debate_stop_after(
                                    debate_stop_after=getattr(args, "debate_stop_after", None),
                                    n_rounds=int(args.n_rounds),
                                    judge_rounds=judge_rounds,
                                )
                            except ValueError as exc:
                                print(str(exc), file=status_file)
                                sys.exit(2)

                            judge_sampling_kwargs: dict[str, Any] | None = None
                            if (
                                args.judge_max_tokens is not None
                                or args.judge_temperature is not None
                            ):
                                main_sampling = judge_runtime_sampling_config or sampling_config
                                judge_sampling_kwargs = {
                                    "max_tokens": int(args.judge_max_tokens) if args.judge_max_tokens is not None else (
                                        int(main_sampling.max_tokens) if main_sampling.max_tokens is not None else None
                                    ),
                                }
                                if args.judge_temperature is not None:
                                    judge_sampling_kwargs["temperature"] = float(args.judge_temperature)

                            debate_stop_after = getattr(args, "debate_stop_after", None)
                            interactive_debate = bool(sys.stdin.isatty()) and (
                                debate_stop_after is not None or ran_staged_personas
                            )
                            if ran_staged_personas and bool(sys.stdin.isatty()):
                                if not _prompt_continue_to_debate(status_file=status_file):
                                    print(
                                        "[staged] Stopped after personas. Progress has been saved.",
                                        file=status_file,
                                    )
                                    return
                            current_debate_stop_after = (
                                debate_stop_after
                                if debate_stop_after is not None
                                else ("round_0" if interactive_debate else None)
                            )
                            auto_debate_stage_pointer = _auto_debate_stage_pointer_path(
                                out_dir=out_dir,
                                dataset_tag=dataset_tag,
                            )
                            using_auto_debate_stage_file = False
                            if cli_stage_state_path is not None:
                                debate_stage_state_file = cli_stage_state_path
                            else:
                                debate_stage_state_file = _auto_resume_debate_stage_file(
                                    pointer_path=auto_debate_stage_pointer,
                                    dataset=dataset,
                                    items=items,
                                    n_agents=int(args.n_agents),
                                    n_rounds=int(args.n_rounds),
                                    judge_rounds=judge_rounds,
                                )
                                if debate_stage_state_file is not None:
                                    using_auto_debate_stage_file = True
                                elif current_debate_stop_after:
                                    debate_stage_state_file = out_dir / f"stage_state_{dataset_tag}_{run_tag}.jsonl"
                                    _write_auto_debate_stage_pointer(auto_debate_stage_pointer, debate_stage_state_file)
                                    using_auto_debate_stage_file = True

                            def _run_debate_once(stage_state_file: Path | None, *, stop_after: str | None):
                                return run_debate(
                                    dataset=dataset,
                                    items=items,
                                    engine=engine,
                                    n_agents=args.n_agents,
                                    n_rounds=args.n_rounds,
                                    judge_rounds=judge_rounds,
                                    batch_size=batch_size,
                                    judge_block_size=args.judge_block_size,
                                    judge_sampling_kwargs=judge_sampling_kwargs,
                                    judge_strict_final_only=bool(args.judge_strict_final_only),
                                    judge_recovery_parse_enabled=bool(args.judge_recovery_parse_enabled),
                                    judge_engine=judge_runtime_engine,
                                    use_personas=bool(args.use_personas),
                                    artifacts_dir=persona_artifacts_dir,
                                    persona_seed=int(args.persona_seed),
                                    persona_axis_mode=str(args.persona_axes_mode),
                                    persona_fixed_axis_count=int(args.persona_fixed_axis_count),
                                    persona_task_axis_count=int(args.persona_task_axis_count),
                                    persona_sampling_method=str(args.persona_sampling_method),
                                    persona_judge_mode=str(args.judge_persona_mode),
                                    persona_backend=persona_backend,
                                    generator_model=generator_model_name,
                                    judge_generator_model=judge_generator_model_name,
                                    persona_generator_engine=persona_generator_engine,
                                    persona_judge_engine=persona_judge_engine,
                                    persona_axes_file=persona_axes_path,
                                    persona_save_artifacts=bool(args.persona_save_artifacts),
                                    persona_replay=bool(args.persona_replay),
                                    judge_trace_mode=str(args.judge_trace_mode),
                                    judge_bank_dir=judge_bank_path,
                                    judge_bank_refresh=bool(args.judge_bank_refresh),
                                    gpqa_family_cache_path=gpqa_family_cache_path,
                                    public_rationale_max_tokens=int(args.public_rationale_max_tokens),
                                    debate_protocol=str(args.debate_protocol),
                                    enable_runtime_judge_persona=bool(args.use_personas),
                                    persona_artifacts_by_item=generated_persona_artifacts,
                                    progress_file=progress_file,
                                    debate_stop_after=stop_after,
                                    stage_state_file=stage_state_file,
                                    persona_plain_agents=persona_plain_agents,
                                )

                            while True:
                                debate_cost_before = _cost_summary_snapshot(cost_tracker)
                                try:
                                    results_by_round = _run_debate_once(
                                        debate_stage_state_file,
                                        stop_after=current_debate_stop_after,
                                    )
                                except ValueError as exc:
                                    if not using_auto_debate_stage_file or not _is_debate_resume_settings_mismatch(exc):
                                        raise
                                    _clear_auto_debate_stage_pointer(auto_debate_stage_pointer)
                                    using_auto_debate_stage_file = False
                                    debate_stage_state_file = None
                                    if current_debate_stop_after:
                                        debate_stage_state_file = out_dir / f"stage_state_{dataset_tag}_{run_tag}.jsonl"
                                        _write_auto_debate_stage_pointer(auto_debate_stage_pointer, debate_stage_state_file)
                                        using_auto_debate_stage_file = True
                                    results_by_round = _run_debate_once(
                                        debate_stage_state_file,
                                        stop_after=current_debate_stop_after,
                                    )
                                    _print_stage_cost_summary(
                                        label="debate" if current_debate_stop_after is None else f"debate:{current_debate_stop_after}",
                                        before=debate_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                except _DebateStopped:
                                    completed_stage = str(current_debate_stop_after)
                                    _print_stage_cost_summary(
                                        label=f"debate:{completed_stage}",
                                        before=debate_cost_before,
                                        cost_tracker=cost_tracker,
                                        status_file=status_file,
                                    )
                                    print(
                                        f"[staged] Debate stopped after '{completed_stage}'. "
                                        f"State: {debate_stage_state_file}",
                                        file=status_file,
                                    )
                                    if not interactive_debate:
                                        return
                                    next_stage = _next_debate_stage(
                                        completed_stage=completed_stage,
                                        n_rounds=int(args.n_rounds),
                                        judge_rounds=judge_rounds,
                                    )
                                    if next_stage is None:
                                        return
                                    if not _prompt_continue_debate_stage(
                                        next_stage=next_stage,
                                        status_file=status_file,
                                    ):
                                        print(
                                            f"[staged] Debate staging stopped after '{completed_stage}'. "
                                            f"State: {debate_stage_state_file}",
                                            file=status_file,
                                        )
                                        return
                                    current_debate_stop_after = next_stage
                                    continue
                                break
                            if using_auto_debate_stage_file:
                                _clear_auto_debate_stage_pointer(auto_debate_stage_pointer)
                            max_round = max(judge_rounds) if judge_rounds else args.n_rounds
                            records = results_by_round.get(max_round, [])
                        else:
                            continue

                        if mode == "single":
                            default_out_path = out_dir / f"single_{dataset_tag}_{run_tag}_{model_tag}.jsonl"
                        elif mode == "majority":
                            default_out_path = out_dir / f"majority_{dataset_tag}_samples{args.majority_samples}_{run_tag}_{model_tag}.jsonl"
                        elif mode == "personas":
                            default_out_path = out_dir / f"personas_{dataset_tag}_{run_tag}.jsonl"
                        else:
                            assert results_by_round is not None
                            assert judge_rounds is not None
                            final_r = args.n_rounds if args.debate_judge_rounds is None else max(judge_rounds)
                            if args.debate_judge_rounds is not None or (args.judge_block_size is not None and args.judge_block_size > 0):
                                for r in sorted(judge_rounds):
                                    if r == final_r:
                                        continue
                                    default_out_path_r = out_dir / f"debate_{dataset_tag}_agents{args.n_agents}_r{r}_{run_tag}_{model_tag}.jsonl"
                                    run_meta_r = _run_meta_block(
                                        run_tag=run_tag,
                                        dataset=dataset,
                                        meta=meta,
                                        manifest_path=manifest_path,
                                        output_path=default_out_path_r,
                                        emit_trace_level=str(args.emit_trace_level),
                                    )
                                    round_rows = _augment_output_rows(
                                        results_by_round[r],
                                        run_meta=run_meta_r,
                                        mode=mode,
                                        use_personas=bool(args.use_personas),
                                        judge_trace_mode=str(args.judge_trace_mode),
                                        public_rationale_max_tokens=int(args.public_rationale_max_tokens),
                                        emit_trace_level=str(args.emit_trace_level),
                                    )
                                    _write_jsonl(default_out_path_r, round_rows)
                                    print(f"[output] Written to {default_out_path_r}", file=status_file)
                            default_out_path = out_dir / f"debate_{dataset_tag}_agents{args.n_agents}_r{final_r}_{run_tag}_{model_tag}.jsonl"

                        out_path = Path(args.output) if args.output else default_out_path
                        run_meta = _run_meta_block(
                            run_tag=run_tag,
                            dataset=dataset,
                            meta=meta,
                            manifest_path=manifest_path,
                            output_path=out_path,
                            emit_trace_level=str(args.emit_trace_level),
                        )
                        records = _augment_output_rows(
                            records,
                            run_meta=run_meta,
                            mode=mode,
                            use_personas=bool(args.use_personas) or mode == "personas",
                            judge_trace_mode=None if mode == "personas" else str(args.judge_trace_mode),
                            public_rationale_max_tokens=None if mode == "personas" else int(args.public_rationale_max_tokens),
                            emit_trace_level=str(args.emit_trace_level),
                        )
                        results[mode] = records
                        if not args.quiet:
                            if mode == "personas":
                                print(f"[result] {mode}: generated {len(records)} artifacts", file=sys.stderr)
                            else:
                                acc = _accuracy(records)
                                print(
                                    f"[result] {mode}: {acc*100:.1f}% ({sum(r['final_correct'] for r in records)}/{len(records)})",
                                    file=sys.stderr,
                                )
                        _write_jsonl(out_path, records)
                        print(f"[output] Written to {out_path}", file=status_file)
                except SpendLimitExceeded as exc:
                    cost_limit_exc = exc
                    print(f"[cost] {exc}", file=status_file)
                finally:
                    cost_tracking_ctx.__exit__(None, None, None)

            finally:
                if engine is not None:
                    engine.shutdown()
                if judge_runtime_engine is not None and judge_runtime_engine is not engine:
                    judge_runtime_engine.shutdown()
                if persona_generator_engine is not None and persona_generator_engine is not engine:
                    persona_generator_engine.shutdown()
                if (
                    persona_judge_engine is not None
                    and persona_judge_engine is not engine
                    and persona_judge_engine is not persona_generator_engine
                ):
                    persona_judge_engine.shutdown()
                if cost_tracker is not None:
                    print(
                        format_cost_summary(
                            cost_tracker.summary(),
                            ledger_path=cost_tracker.ledger_path,
                            max_run_cost_usd=cost_tracker.max_run_cost_usd,
                            max_total_cost_usd=cost_tracker.max_total_cost_usd,
                        ),
                        file=status_file,
                    )
                if cost_limit_exc is not None:
                    sys.exit(3)

        print("\n=== Summary ===", file=status_file)
        for mode, records in results.items():
            if mode == "personas":
                print(f"{mode}: {len(records)} artifacts", file=status_file)
            else:
                acc = _accuracy(records)
                print(f"{mode}: {acc*100:.1f}%", file=status_file)


if __name__ == "__main__":
    main()
