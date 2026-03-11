from __future__ import annotations

import json
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .engines.base import InferenceResult


PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class GeminiTextPricing:
    input_per_million_tokens_usd: float
    cached_input_per_million_tokens_usd: float
    output_per_million_tokens_usd: float
    cache_storage_per_million_tokens_hour_usd: float
    prompt_threshold_tokens: int | None = None
    pricing_label: str = "standard"


def _gemini_text_pricing(model_name: str, prompt_tokens: int | None) -> GeminiTextPricing | None:
    del prompt_tokens
    model = str(model_name or "").strip().lower()
    if "gemini-3-flash-preview" not in model:
        return None
    return GeminiTextPricing(
        input_per_million_tokens_usd=0.50,
        cached_input_per_million_tokens_usd=0.05,
        output_per_million_tokens_usd=3.00,
        cache_storage_per_million_tokens_hour_usd=1.00,
        pricing_label="gemini_3_flash_text",
    )


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid usage metadata for {key}: {value!r}") from exc


def _billable_output_tokens(usage: dict[str, Any]) -> int | None:
    candidates = _usage_int(usage, "candidates_token_count")
    thoughts = _usage_int(usage, "thoughts_token_count")
    if candidates is not None:
        return int(candidates) + int(thoughts or 0)
    total = _usage_int(usage, "total_token_count")
    prompt = _usage_int(usage, "prompt_token_count")
    if total is not None and prompt is not None:
        return max(0, int(total) - int(prompt))
    return None


def estimate_result_cost(result: "InferenceResult") -> dict[str, Any]:
    provider_name = str(result.provider_name or "").strip().lower()
    usage = dict(result.usage or {})
    provider_meta = dict(result.provider_meta or {})
    prompt_tokens = _usage_int(usage, "prompt_token_count")
    cached_input_tokens = _usage_int(usage, "cached_content_token_count") or 0
    uncached_input_tokens = None
    if prompt_tokens is not None:
        uncached_input_tokens = max(0, int(prompt_tokens) - int(cached_input_tokens))
    billable_output_tokens = _billable_output_tokens(usage)

    if provider_name != "gemini":
        print(
            f"[cost] unknown provider {provider_name!r} for model {result.model_name!r}; cost estimation unavailable",
            file=sys.stderr,
        )
        return {
            "known_cost": False,
            "pricing_source": None,
            "pricing_label": None,
            "estimated_cost_usd": None,
            "components": {},
            "input_tokens": prompt_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "billable_output_tokens": billable_output_tokens,
        }

    pricing = _gemini_text_pricing(str(result.model_name or ""), prompt_tokens)
    if pricing is None or prompt_tokens is None or billable_output_tokens is None:
        missing = []
        if pricing is None:
            missing.append(f"no pricing for model {result.model_name!r}")
        if prompt_tokens is None:
            missing.append("missing prompt_token_count")
        if billable_output_tokens is None:
            missing.append("missing output token counts")
        print(f"[cost] incomplete cost data ({', '.join(missing)}); recording without cost estimate", file=sys.stderr)
        return {
            "known_cost": False,
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_label": None if pricing is None else pricing.pricing_label,
            "estimated_cost_usd": None,
            "components": {},
            "input_tokens": prompt_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "billable_output_tokens": billable_output_tokens,
        }

    input_cost = float(uncached_input_tokens or 0) / 1_000_000.0 * pricing.input_per_million_tokens_usd
    cached_input_cost = float(cached_input_tokens or 0) / 1_000_000.0 * pricing.cached_input_per_million_tokens_usd
    output_cost = float(billable_output_tokens or 0) / 1_000_000.0 * pricing.output_per_million_tokens_usd
    cache_storage_cost = 0.0
    if provider_meta.get("explicit_cache_created"):
        prefix_tokens = provider_meta.get("explicit_cache_prefix_tokens")
        ttl_seconds = provider_meta.get("explicit_cache_ttl_seconds")
        if prefix_tokens is not None and ttl_seconds is not None:
            try:
                cache_storage_cost = (
                    float(prefix_tokens) / 1_000_000.0
                    * pricing.cache_storage_per_million_tokens_hour_usd
                    * (float(ttl_seconds) / 3600.0)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid Gemini cache metadata for cost estimation: "
                    f"explicit_cache_prefix_tokens={prefix_tokens!r}, explicit_cache_ttl_seconds={ttl_seconds!r}"
                ) from exc

    total_cost = input_cost + cached_input_cost + output_cost + cache_storage_cost
    return {
        "known_cost": True,
        "pricing_source": PRICING_SOURCE_URL,
        "pricing_label": pricing.pricing_label,
        "estimated_cost_usd": total_cost,
        "components": {
            "uncached_input_cost_usd": input_cost,
            "cached_input_cost_usd": cached_input_cost,
            "output_cost_usd": output_cost,
            "cache_storage_cost_usd": cache_storage_cost,
        },
        "input_tokens": prompt_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "billable_output_tokens": billable_output_tokens,
    }


class SpendLimitExceeded(RuntimeError):
    def __init__(self, *, message: str, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


class CostTracker:
    def __init__(
        self,
        *,
        ledger_path: Path,
        session_name: str,
        session_meta: dict[str, Any] | None = None,
        max_run_cost_usd: float | None = None,
        max_total_cost_usd: float | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.session_name = str(session_name)
        self.session_id = f"{self.session_name}-{uuid.uuid4().hex[:12]}"
        self.session_meta = dict(session_meta or {})
        self.max_run_cost_usd = None if max_run_cost_usd is None else float(max_run_cost_usd)
        self.max_total_cost_usd = None if max_total_cost_usd is None else float(max_total_cost_usd)
        self._lock = threading.Lock()
        self._starting_summary = self._read_summary(self.ledger_path)
        self._session_summary = self._empty_summary()

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "n_calls": 0,
            "estimated_cost_usd": 0.0,
            "known_cost_calls": 0,
            "unknown_cost_calls": 0,
            "input_tokens": 0,
            "uncached_input_tokens": 0,
            "cached_input_tokens": 0,
            "billable_output_tokens": 0,
        }

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any]:
        summary = {
            "n_calls": 0,
            "estimated_cost_usd": 0.0,
            "known_cost_calls": 0,
            "unknown_cost_calls": 0,
            "input_tokens": 0,
            "uncached_input_tokens": 0,
            "cached_input_tokens": 0,
            "billable_output_tokens": 0,
        }
        if not path.exists():
            return summary
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in ledger {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Invalid ledger row in {path} at line {line_number}: expected object, got {type(row).__name__}"
                    )
                summary["n_calls"] += 1
                cost = row.get("estimated_cost_usd")
                if cost is None:
                    summary["unknown_cost_calls"] += 1
                elif isinstance(cost, (int, float)):
                    summary["estimated_cost_usd"] += float(cost)
                    summary["known_cost_calls"] += 1
                else:
                    raise ValueError(
                        f"Invalid ledger cost in {path} at line {line_number}: estimated_cost_usd={cost!r}"
                    )
                for key in ("input_tokens", "uncached_input_tokens", "cached_input_tokens", "billable_output_tokens"):
                    value = row.get(key)
                    if value is None:
                        continue
                    if isinstance(value, int):
                        summary[key] += value
                        continue
                    raise ValueError(
                        f"Invalid ledger token count in {path} at line {line_number}: {key}={value!r}"
                    )
        return summary

    def record_result(self, result: "InferenceResult") -> dict[str, Any]:
        estimate = estimate_result_cost(result)
        entry = {
            "timestamp_utc": _timestamp_utc(),
            "session_id": self.session_id,
            "session_name": self.session_name,
            **self.session_meta,
            "provider_name": result.provider_name,
            "model_name": result.model_name,
            "model_role": result.model_role,
            "request_id": result.provider_meta.get("request_id"),
            "response_id": result.provider_meta.get("response_id"),
            "latency_ms": result.latency_ms,
            "retries": result.retries,
            "error": None if result.error is None else dict(result.error),
            "input_tokens": estimate["input_tokens"],
            "uncached_input_tokens": estimate["uncached_input_tokens"],
            "cached_input_tokens": estimate["cached_input_tokens"],
            "billable_output_tokens": estimate["billable_output_tokens"],
            "estimated_cost_usd": estimate["estimated_cost_usd"],
            "cost_components": estimate["components"],
            "pricing_label": estimate["pricing_label"],
            "pricing_source": estimate["pricing_source"],
            "provider_meta": {
                key: result.provider_meta.get(key)
                for key in (
                    "explicit_cache_used",
                    "explicit_cache_created",
                    "explicit_cache_name",
                    "explicit_cache_scope",
                    "explicit_cache_prefix_tokens",
                    "explicit_cache_ttl_seconds",
                )
                if result.provider_meta.get(key) is not None
            },
        }

        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._session_summary["n_calls"] += 1
            if isinstance(estimate["input_tokens"], int):
                self._session_summary["input_tokens"] += int(estimate["input_tokens"])
            if isinstance(estimate["uncached_input_tokens"], int):
                self._session_summary["uncached_input_tokens"] += int(estimate["uncached_input_tokens"])
            if isinstance(estimate["cached_input_tokens"], int):
                self._session_summary["cached_input_tokens"] += int(estimate["cached_input_tokens"])
            if isinstance(estimate["billable_output_tokens"], int):
                self._session_summary["billable_output_tokens"] += int(estimate["billable_output_tokens"])
            if isinstance(estimate["estimated_cost_usd"], (int, float)):
                self._session_summary["estimated_cost_usd"] += float(estimate["estimated_cost_usd"])
                self._session_summary["known_cost_calls"] += 1
            else:
                self._session_summary["unknown_cost_calls"] += 1

            summary = self.summary()
            if self.max_run_cost_usd is not None and summary["session"]["estimated_cost_usd"] > self.max_run_cost_usd:
                raise SpendLimitExceeded(
                    message=(
                        f"Session cost cap exceeded: ${summary['session']['estimated_cost_usd']:.6f} > "
                        f"${self.max_run_cost_usd:.6f}"
                    ),
                    summary=summary,
                )
            if self.max_total_cost_usd is not None and summary["cumulative"]["estimated_cost_usd"] > self.max_total_cost_usd:
                raise SpendLimitExceeded(
                    message=(
                        f"Cumulative cost cap exceeded: ${summary['cumulative']['estimated_cost_usd']:.6f} > "
                        f"${self.max_total_cost_usd:.6f}"
                    ),
                    summary=summary,
                )
        return entry

    def summary(self) -> dict[str, Any]:
        session = dict(self._session_summary)
        previous = dict(self._starting_summary)
        cumulative = dict(previous)
        for key in ("n_calls", "known_cost_calls", "unknown_cost_calls", "input_tokens", "uncached_input_tokens", "cached_input_tokens", "billable_output_tokens"):
            cumulative[key] += int(session[key])
        cumulative["estimated_cost_usd"] += float(session["estimated_cost_usd"])
        return {"session": session, "previous": previous, "cumulative": cumulative}


_ACTIVE_COST_TRACKER: CostTracker | None = None


def get_active_cost_tracker() -> CostTracker | None:
    return _ACTIVE_COST_TRACKER


@contextmanager
def active_cost_tracking(tracker: CostTracker | None) -> Iterator[CostTracker | None]:
    global _ACTIVE_COST_TRACKER
    prev = _ACTIVE_COST_TRACKER
    _ACTIVE_COST_TRACKER = tracker
    try:
        yield tracker
    finally:
        _ACTIVE_COST_TRACKER = prev


def maybe_record_result(result: "InferenceResult") -> dict[str, Any] | None:
    tracker = get_active_cost_tracker()
    if tracker is None:
        return None
    return tracker.record_result(result)


def format_cost_summary(summary: dict[str, Any], *, ledger_path: Path) -> str:
    session = summary["session"]
    previous = summary["previous"]
    cumulative = summary["cumulative"]
    return (
        f"[cost] run=${session['estimated_cost_usd']:.6f} "
        f"(calls={session['n_calls']}, input={session['input_tokens']}, cached={session['cached_input_tokens']}, output={session['billable_output_tokens']}) | "
        f"previous=${previous['estimated_cost_usd']:.6f} | "
        f"cumulative=${cumulative['estimated_cost_usd']:.6f} | "
        f"ledger={ledger_path}"
    )
