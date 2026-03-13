from __future__ import annotations

import argparse

from ..engines import GEMINI_3_FLASH_MODEL
from .judge_common import _parse_judge_rounds
from .subset import _parse_subset_n_arg


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run multi-agent debate, majority voting, single-response inference, or persona generation on AIME25/GPQA/HLE-Verified."
    )

    ap.add_argument(
        "--model_name",
        "--debater_model",
        dest="model_name",
        type=str,
        default=GEMINI_3_FLASH_MODEL,
        help=f"Primary debater model ID. The runtime is hardcoded to {GEMINI_3_FLASH_MODEL}.",
    )
    ap.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "gemini"],
        help="Inference provider for the primary debater model. Gemini remains the only supported runtime.",
    )
    ap.add_argument(
        "--judge_runtime_model",
        type=str,
        default=None,
        help=f"Optional judge model override. Ignored; the runtime always uses {GEMINI_3_FLASH_MODEL}.",
    )
    ap.add_argument(
        "--judge_provider",
        type=str,
        default="auto",
        choices=["auto", "gemini"],
        help="Inference provider for the judge runtime model.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="aime25",
        choices=["aime25", "gpqa", "hle"],
        help="Dataset to evaluate on.",
    )
    ap.add_argument(
        "--one",
        type=str,
        default=None,
        help="Convenience selector for a single item by item_uid, item_display_id, orig_id, or subset_id.",
    )
    ap.add_argument(
        "--hle_variant",
        type=str,
        default="verified",
        choices=["verified", "revised", "verified_full"],
        help="HLE-Verified slice to materialize/use when --dataset hle is selected.",
    )
    ap.add_argument(
        "--exclude_ids",
        type=str,
        default=None,
        help="Optional path to a newline/JSON list/map of canonical ids, item_uids, or orig_ids to exclude from sampling.",
    )
    ap.add_argument(
        "--dataset_local_mirror",
        type=str,
        default=None,
        help="Optional local dataset mirror root or explicit JSONL path. For HLE this bypasses download and pinned materialization.",
    )

    ap.add_argument(
        "--all",
        action="store_true",
        help="Run on the full dataset (overrides --subset_n/--subset_ids/--subset_range).",
    )
    ap.add_argument("--subset_n", type=_parse_subset_n_arg, default=20, help="Number of random samples (or 'all').")
    ap.add_argument("--subset_ids", type=str, default=None, help="Comma-separated specific indices.")
    ap.add_argument("--subset_range", type=str, default=None, help="Range like '0:10' or '0-9'.")
    ap.add_argument("--subset_seed", type=int, default=None, help="Random seed for subset sampling.")

    ap.add_argument("--mode", type=str, default="single,debate", help="Modes to run: single, majority, debate, personas (comma-separated).")

    ap.add_argument("--n_agents", type=int, default=5, help="Number of agents for debate.")
    ap.add_argument(
        "--n_rounds",
        type=int,
        default=3,
        help="Number of debate update rounds after the initial independent solve.",
    )
    ap.add_argument("--majority_samples", type=int, default=5, help="Samples for majority voting.")
    ap.add_argument(
        "--debate_judge_rounds",
        type=str,
        default=None,
        help="Comma-separated debate-update round numbers to judge (e.g., '1,2,3'). If not specified, only judges the final debate round.",
    )
    ap.add_argument(
        "--judge_block_size",
        type=int,
        default=None,
        help="Force a fixed judge block size (N rounds per judge prompt). If omitted, auto-selects the largest window that fits.",
    )
    ap.add_argument("--judge_max_tokens", type=int, default=None, help="Optional max new tokens for judge output.")
    ap.add_argument("--judge_temperature", type=float, default=None, help="Optional judge temperature.")
    ap.add_argument(
        "--judge_strict_final_only",
        dest="judge_strict_final_only",
        action="store_true",
        default=True,
        help="Require strict boxed final-answer parse before accepting judge output.",
    )
    ap.add_argument(
        "--no_judge_strict_final_only",
        dest="judge_strict_final_only",
        action="store_false",
        help="Disable strict-only acceptance of judge output.",
    )
    ap.add_argument(
        "--judge_recovery_parse_enabled",
        dest="judge_recovery_parse_enabled",
        action="store_true",
        default=True,
        help="Enable conservative recovery parse before judge retry.",
    )
    ap.add_argument(
        "--no_judge_recovery_parse_enabled",
        dest="judge_recovery_parse_enabled",
        action="store_false",
        help="Disable conservative recovery parse before judge retry.",
    )

    ap.add_argument(
        "--context_len",
        type=int,
        default=None,
        help="Fixed context length (applies to both main generation and judge). If omitted, uses adaptive context.",
    )
    ap.add_argument(
        "--max_model_len",
        dest="context_len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--batch_size", type=int, default=None, help="Max batch size for inference (default: auto).")

    ap.add_argument(
        "--quiet",
        "--silent",
        action="store_true",
        help="Silence provider/SDK logs; keep only progress bars, output paths, and final summary.",
    )

    ap.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    ap.add_argument("--output", type=str, default=None, help="Optional explicit output JSONL path for the final records file.")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for output files.")
    ap.add_argument(
        "--persona_stage",
        type=str,
        default="full",
        choices=["axes", "descriptors", "cards", "judge_card", "full"],
        help="Run staged persona generation up to this stage; interactive terminals can continue through later stages without rerunning the command.",
    )
    ap.add_argument(
        "--stage_state_file",
        type=str,
        default=None,
        help="JSONL file for staged execution state. Each stage appends one line; resume reads the last.",
    )
    ap.add_argument(
        "--debate_stop_after",
        type=str,
        default=None,
        help="Stop debate after this step (e.g. 'round_0', 'round_1_judge'). Omit to run to completion.",
    )

    ap.add_argument("--use_personas", action="store_true", help="Enable persona-conditioned debate or majority generation/replay.")
    ap.add_argument("--persona_n", type=int, default=5, help="Number of personas to generate in persona mode.")
    ap.add_argument(
        "--persona_axes_mode",
        type=str,
        default="hybrid",
        choices=["fixed", "task", "hybrid", "file", "replay"],
        help="Axis selection mode for persona generation.",
    )
    ap.add_argument(
        "--persona_sampling_method",
        type=str,
        default="maximin",
        choices=["maximin", "halton"],
        help="Sampling method for persona axis points.",
    )
    ap.add_argument("--persona_fixed_axis_count", type=int, default=3, help="Number of fixed axes to use in hybrid/fixed persona generation.")
    ap.add_argument("--persona_task_axis_count", type=int, default=3, help="Number of task axes to use in hybrid/task persona generation.")
    ap.add_argument("--persona_axes_file", type=str, default=None, help="JSON file containing axes when --persona_axes_mode file is used.")
    ap.add_argument(
        "--persona_backend",
        type=str,
        default="llm",
        choices=["auto", "heuristic", "llm"],
        help="Persona generation backend. Defaults to llm. 'auto' uses llm when a generator model is available, else heuristic.",
    )
    ap.add_argument(
        "--generator_model",
        type=str,
        default=None,
        help=f"Optional persona generation model override. Ignored; the runtime always uses {GEMINI_3_FLASH_MODEL}.",
    )
    ap.add_argument(
        "--generator_provider",
        type=str,
        default="auto",
        choices=["auto", "gemini"],
        help="Inference provider for persona generation models.",
    )
    ap.add_argument(
        "--judge_persona_mode",
        type=str,
        default="benchmark_family_bank",
        choices=["neutral_baseline", "task_family_generated", "question_conditioned_generated", "benchmark_family_bank"],
        help="Judge persona generation mode for persona artifacts.",
    )
    ap.add_argument(
        "--judge_generator_model",
        "--judge_model",
        dest="judge_generator_model",
        type=str,
        default=None,
        help=f"Optional judge-card generation model override. Ignored; the runtime always uses {GEMINI_3_FLASH_MODEL}.",
    )
    ap.add_argument(
        "--judge_generator_provider",
        type=str,
        default="auto",
        choices=["auto", "gemini"],
        help="Inference provider for judge-card generation.",
    )
    ap.add_argument("--persona_artifacts_dir", type=str, default=None, help="Directory for persona artifact save/replay.")
    ap.add_argument("--judge_bank_dir", type=str, default=None, help="Directory for persistent benchmark-family judge-bank artifacts.")
    ap.add_argument("--judge_bank_refresh", action="store_true", help="Regenerate persistent benchmark-family judge-bank entries.")
    ap.add_argument("--gpqa_family_cache_path", type=str, default=None, help="Optional cache path for GPQA biology/chemistry/physics family assignments.")
    ap.add_argument("--persona_seed", type=int, default=0, help="Seed for persona sampling.")
    ap.add_argument("--persona_replay", action="store_true", help="Replay saved persona artifacts instead of generating new ones.")
    ap.add_argument("--persona_save_artifacts", action="store_true", help="Persist generated persona artifacts.")
    ap.add_argument("--persona_dump_cards", action="store_true", help="Print generated persona system prompts to stdout/stderr summary stream.")
    ap.add_argument(
        "--judge_trace_mode",
        type=str,
        default="visible_plus_thought_summary",
        choices=["assistant_transcript", "visible_plus_thought_summary"],
        help="Trace representation used when constructing judge prompts.",
    )
    ap.add_argument(
        "--public_rationale_max_tokens",
        type=int,
        default=96,
        help="Approximate token/word budget used when deriving bounded public rationales for debate sharing.",
    )
    ap.add_argument(
        "--emit_trace_level",
        type=str,
        default="full",
        choices=["minimal", "full"],
        help="Logical trace block detail level embedded in output rows.",
    )
    ap.add_argument("--final_manifest", type=str, default=None, help="Optional manifest path used to write or lock final-run config.")
    ap.add_argument("--final_run", action="store_true", help="Validate or create a final-run manifest that freezes key config.")
    ap.add_argument(
        "--token_ledger_path",
        type=str,
        default=None,
        help="Append-only JSONL ledger path for per-call token and cost tracking. Defaults to out/token_ledger.jsonl.",
    )
    ap.add_argument(
        "--max_run_cost_usd",
        type=float,
        default=None,
        help="Stop once the current run's estimated Gemini API spend exceeds this USD cap.",
    )
    ap.add_argument(
        "--max_total_cost_usd",
        type=float,
        default=None,
        help="Stop once the cumulative estimated spend recorded in the token ledger exceeds this USD cap.",
    )

    return ap


def build_parser():
    return _build_arg_parser()


__all__ = ["_build_arg_parser", "_parse_judge_rounds", "_parse_subset_n_arg", "build_parser"]
