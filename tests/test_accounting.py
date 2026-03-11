from __future__ import annotations

import json
from pathlib import Path

import pytest

from debate_v_majority.accounting import CostTracker, SpendLimitExceeded, estimate_result_cost
from debate_v_majority.engines.base import InferenceResult


def test_estimate_result_cost_for_gemini_3_flash_includes_cache_storage():
    result = InferenceResult(
        text="answer",
        usage={
            "prompt_token_count": 2000,
            "cached_content_token_count": 1200,
            "candidates_token_count": 300,
            "thoughts_token_count": 50,
            "total_token_count": 2350,
        },
        provider_meta={
            "explicit_cache_created": True,
            "explicit_cache_prefix_tokens": 1200,
            "explicit_cache_ttl_seconds": 3600,
        },
        model_role="debater",
        model_name="gemini-3-flash",
        provider_name="gemini",
    )
    estimate = estimate_result_cost(result)
    assert estimate["known_cost"] is True
    assert estimate["pricing_label"] == "gemini_3_flash_text"
    assert estimate["uncached_input_tokens"] == 800
    assert estimate["cached_input_tokens"] == 1200
    assert estimate["billable_output_tokens"] == 350
    assert estimate["components"]["cache_storage_cost_usd"] > 0.0
    assert estimate["estimated_cost_usd"] > estimate["components"]["output_cost_usd"]


def test_cost_tracker_appends_rows_and_enforces_run_cap(tmp_path: Path):
    ledger_path = tmp_path / "token_ledger.jsonl"
    tracker = CostTracker(
        ledger_path=ledger_path,
        session_name="test-run",
        session_meta={"dataset": "aime25"},
        max_run_cost_usd=0.0001,
    )
    cheap_result = InferenceResult(
        text="ok",
        usage={
            "prompt_token_count": 100,
            "candidates_token_count": 10,
            "total_token_count": 110,
        },
        provider_meta={"request_id": "req-1"},
        model_role="debater",
        model_name="gemini-3-flash",
        provider_name="gemini",
    )
    tracker.record_result(cheap_result)
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first = json.loads(lines[0])
    assert first["session_name"] == "test-run"
    assert first["dataset"] == "aime25"
    assert first["estimated_cost_usd"] is not None

    expensive_result = InferenceResult(
        text="still ok",
        usage={
            "prompt_token_count": 50_000,
            "candidates_token_count": 10_000,
            "total_token_count": 60_000,
        },
        provider_meta={"request_id": "req-2"},
        model_role="judge",
        model_name="gemini-3-flash",
        provider_name="gemini",
    )
    with pytest.raises(SpendLimitExceeded):
        tracker.record_result(expensive_result)
    summary = tracker.summary()
    assert summary["session"]["estimated_cost_usd"] > 0.0001
    assert summary["cumulative"]["n_calls"] == 2


def test_estimate_result_cost_raises_on_invalid_cache_metadata():
    result = InferenceResult(
        text="answer",
        usage={
            "prompt_token_count": 2000,
            "cached_content_token_count": 1200,
            "candidates_token_count": 300,
            "thoughts_token_count": 50,
            "total_token_count": 2350,
        },
        provider_meta={
            "explicit_cache_created": True,
            "explicit_cache_prefix_tokens": "bad-prefix",
            "explicit_cache_ttl_seconds": 3600,
        },
        model_role="debater",
        model_name="gemini-3-flash",
        provider_name="gemini",
    )
    with pytest.raises(ValueError, match="explicit_cache_prefix_tokens='bad-prefix'"):
        estimate_result_cost(result)


def test_estimate_result_cost_raises_on_invalid_usage_metadata():
    result = InferenceResult(
        text="answer",
        usage={
            "prompt_token_count": "bad-prompt",
            "candidates_token_count": 300,
            "total_token_count": 2350,
        },
        provider_meta={},
        model_role="debater",
        model_name="gemini-3-flash",
        provider_name="gemini",
    )
    with pytest.raises(ValueError, match="Invalid usage metadata for prompt_token_count"):
        estimate_result_cost(result)


def test_cost_tracker_raises_on_invalid_ledger_json(tmp_path: Path):
    ledger_path = tmp_path / "token_ledger.jsonl"
    ledger_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON in ledger .* line 1"):
        CostTracker(ledger_path=ledger_path, session_name="bad-ledger")


def test_cost_tracker_raises_on_invalid_ledger_cost_field(tmp_path: Path):
    ledger_path = tmp_path / "token_ledger.jsonl"
    ledger_path.write_text('{"estimated_cost_usd":"oops"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ledger cost .* estimated_cost_usd='oops'"):
        CostTracker(ledger_path=ledger_path, session_name="bad-ledger")
