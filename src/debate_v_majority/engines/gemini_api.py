from __future__ import annotations

import concurrent.futures
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import url2pathname, urlopen

from ..accounting import SpendLimitExceeded, maybe_record_result
from .base import BaseInferenceEngine, InferenceResult
from .engine_impl import GEMINI_3_FLASH_MODEL, normalize_gemini_model_name

GEMINI_DEFAULT_CONTEXT_LEN = 1_048_576
GEMINI_DEFAULT_TEMPERATURE = 1.0
GEMINI_DEFAULT_MAX_OUTPUT_TOKENS = 4096
GEMINI_DEFAULT_MAX_PARALLEL_REQUESTS = 8


_dotenv_cache: dict[str, tuple[float, str | None]] = {}
_DOTENV_CACHE_TTL = 30  # seconds


def _dotenv_gemini_api_key(start_dir: str) -> str | None:
    now = time.monotonic()
    cached = _dotenv_cache.get(start_dir)
    if cached is not None and (now - cached[0]) < _DOTENV_CACHE_TTL:
        return cached[1]

    result = _read_dotenv_gemini_key(start_dir)
    _dotenv_cache[start_dir] = (now, result)
    return result


def _read_dotenv_gemini_key(start_dir: str) -> str | None:
    current = Path(start_dir).resolve()
    for directory in (current, *current.parents):
        dotenv_path = directory / ".env"
        if not dotenv_path.is_file():
            continue
        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "GEMINI_API_KEY":
                    continue
                parsed = value.strip()
                if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
                    parsed = parsed[1:-1]
                return parsed or None
        except OSError:
            continue
    return None


def _resolve_gemini_api_key(api_key: str | None) -> str | None:
    dotenv_key = _dotenv_gemini_api_key(str(Path.cwd()))
    if dotenv_key:
        return dotenv_key
    return api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    for method_name in ("model_dump", "dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            try:
                return dict(method(exclude_none=True))
            except TypeError:
                return dict(method())
    out: dict[str, Any] = {}
    for key in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "tool_use_prompt_token_count",
        "traffic_type",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def _sdk_http_response_meta(response: Any) -> dict[str, Any]:
    sdk_http_response = getattr(response, "sdk_http_response", None)
    if sdk_http_response is None:
        return {}
    out: dict[str, Any] = {}
    status_code = getattr(sdk_http_response, "status_code", None)
    if status_code is not None:
        out["http_status_code"] = int(status_code)
    request_url = getattr(sdk_http_response, "url", None)
    if request_url is not None:
        out["request_url"] = str(request_url)
    headers = getattr(sdk_http_response, "headers", None)
    if headers is not None:
        request_id = headers.get("x-request-id") or headers.get("x-goog-request-id")
        if request_id:
            out["request_id"] = str(request_id)
    return out


def _response_text(response: Any) -> str:
    visible_text, _thought_summary, _thought_meta = _response_channels(response)
    if visible_text:
        return visible_text
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _response_channels(response: Any) -> tuple[str, str | None, dict[str, Any]]:
    pieces: list[str] = []
    thought_pieces: list[str] = []
    thought_part_count = 0
    thought_signature_count = 0
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        for part in parts or []:
            if getattr(part, "thought_signature", None) is not None:
                thought_signature_count += 1
            part_text = getattr(part, "text", None)
            if not isinstance(part_text, str) or not part_text:
                continue
            if bool(getattr(part, "thought", False)):
                thought_part_count += 1
                thought_pieces.append(part_text)
            else:
                pieces.append(part_text)
    visible_text = "\n".join(pieces).strip()
    thought_summary = "\n".join(thought_pieces).strip() or None
    return (
        visible_text,
        thought_summary,
        {
            "thought_summary_available": thought_summary is not None,
            "thought_part_count": thought_part_count,
            "thought_signature_count": thought_signature_count,
        },
    )


_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_MAX_RETRIES = 5
_TRANSIENT_BASE_DELAY = 4.0
_TRANSIENT_MAX_DELAY = 120.0


def _is_transient_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _TRANSIENT_STATUS_CODES:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("503", "unavailable", "overloaded", "resource exhausted", "429", "rate limit")
    )


def _guess_mime_type(ref: str | None) -> str:
    guessed, _encoding = mimetypes.guess_type(str(ref or ""))
    return guessed or "application/octet-stream"


def _bytes_from_local_ref(ref: str) -> bytes | None:
    parsed = urlparse(ref)
    path: Path | None = None
    if parsed.scheme == "file":
        uri_path = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            uri_path = f"//{parsed.netloc}{uri_path}"
        path = Path(uri_path)
    elif parsed.scheme == "" or (len(ref) >= 2 and ref[1] == ":"):
        candidate = Path(ref)
        if candidate.exists():
            path = candidate
    if path is None or not path.exists():
        return None
    return path.read_bytes()


def _bytes_from_remote_ref(ref: str) -> tuple[bytes, str | None] | None:
    parsed = urlparse(ref)
    if parsed.scheme not in {"http", "https"}:
        return None
    with urlopen(ref, timeout=30) as response:
        data = response.read()
        return data, response.headers.get_content_type()


class GeminiInferenceEngine(BaseInferenceEngine):
    provider_name = "gemini"

    def __init__(
        self,
        model_name: str,
        *,
        model_role: str | None = None,
        api_key: str | None = None,
        max_model_len: int | None = None,
        api_version: str | None = None,
    ) -> None:
        super().__init__(model_name=normalize_gemini_model_name(model_name), model_role=model_role)
        self._api_key = _resolve_gemini_api_key(api_key)
        self._api_version = api_version or os.environ.get("GOOGLE_GENAI_API_VERSION")
        self._max_model_len = int(max_model_len) if max_model_len is not None else GEMINI_DEFAULT_CONTEXT_LEN
        self._client = None

    def _api_model_name(self) -> str:
        return GEMINI_3_FLASH_MODEL

    def _api_model_resource_name(self) -> str:
        return f"models/{self._api_model_name()}"

    @property
    def context_len_tokens(self) -> int:
        return int(self._max_model_len)

    def initialize(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise ValueError(
                "Gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        from google import genai
        from google.genai import types

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._api_version:
            client_kwargs["http_options"] = types.HttpOptions(api_version=self._api_version)
        self._client = genai.Client(**client_kwargs)

    def shutdown(self) -> None:
        self._client = None

    def _max_parallel_requests(self, batch_size: int | None) -> int:
        if batch_size is not None:
            try:
                return max(1, int(batch_size))
            except (TypeError, ValueError):
                pass
        raw_value = os.environ.get("GEMINI_MAX_PARALLEL_REQUESTS")
        if raw_value:
            try:
                return max(1, int(raw_value))
            except (TypeError, ValueError):
                pass
        return GEMINI_DEFAULT_MAX_PARALLEL_REQUESTS

    def _build_contents(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
        from google.genai import types

        def _text_part(text: Any) -> Any:
            return types.Part.from_text(text=str(text or ""))

        def _part_from_spec(spec: Any) -> Any:
            if isinstance(spec, str):
                return _text_part(spec)
            if not isinstance(spec, dict):
                return _text_part(spec)

            part_type = str(spec.get("type") or "text").strip().lower()
            if part_type == "text":
                return _text_part(spec.get("text"))
            if part_type != "image":
                return _text_part(spec.get("text") or spec)

            image_ref = str(spec.get("image_uri") or spec.get("uri") or spec.get("path") or "").strip()
            fallback_text = str(
                spec.get("fallback_text")
                or spec.get("alt_text")
                or spec.get("text")
                or f"[Image unavailable: {image_ref or 'missing_image_reference'}]"
            ).strip()
            if not image_ref:
                return _text_part(fallback_text)
            mime_type = str(spec.get("mime_type") or _guess_mime_type(image_ref))
            image_label = str(spec.get("source_key") or "image").strip() or "image"

            local_bytes = _bytes_from_local_ref(image_ref)
            if local_bytes is not None:
                return types.Part.from_bytes(data=local_bytes, mime_type=mime_type)

            parsed_ref = urlparse(image_ref)
            if parsed_ref.scheme in {"http", "https"}:
                try:
                    remote = _bytes_from_remote_ref(image_ref)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to fetch explicit image '{image_label}' from {image_ref}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if remote is not None:
                    remote_bytes, remote_mime = remote
                    return types.Part.from_bytes(
                        data=remote_bytes,
                        mime_type=str(remote_mime or mime_type),
                    )
                raise RuntimeError(
                    f"Failed to fetch explicit image '{image_label}' from {image_ref}: "
                    "remote image retrieval returned no data."
                )

            if parsed_ref.scheme in {"", "file"} or (len(image_ref) >= 2 and image_ref[1] == ":"):
                return _text_part(fallback_text)

            return types.Part.from_uri(file_uri=image_ref, mime_type=mime_type)

        def _parts_from_content(content: Any) -> list[Any]:
            if isinstance(content, list):
                return [_part_from_spec(spec) for spec in content]
            return [_part_from_spec(content)]

        def _system_text(content: Any) -> str:
            if isinstance(content, list):
                text_parts = []
                for spec in content:
                    if isinstance(spec, dict) and str(spec.get("type") or "text").strip().lower() == "text":
                        text_parts.append(str(spec.get("text") or ""))
                    elif isinstance(spec, str):
                        text_parts.append(spec)
                return "\n".join(part for part in text_parts if part.strip())
            return str(content or "")

        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            content = message.get("content")
            if role == "cache_control":
                # Older callers may still emit the pseudo-role used for the
                # removed explicit cache path; ignore it rather than forwarding
                # an empty user message to Gemini.
                continue
            if role == "system":
                system_parts.append(_system_text(content))
                continue
            mapped_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=mapped_role,
                    parts=_parts_from_content(content),
                )
            )
        system_instruction = "\n\n".join(part for part in system_parts if part.strip()) or None
        return system_instruction, contents

    def _count_tokens(
        self,
        *,
        system_instruction: str | None,
        contents: list[Any],
    ) -> int:
        from google.genai import types

        assert self._client is not None
        count_contents = list(contents)
        if system_instruction:
            # Gemini's count_tokens endpoint does not accept system_instruction in
            # CountTokensConfig even though generate_content accepts it in
            # GenerateContentConfig. For accounting, prepend the system text as a
            # leading user content block so token counting still includes it.
            count_contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(system_instruction))],
                ),
                *count_contents,
            ]
        response = self._client.models.count_tokens(
            model=self._api_model_resource_name(),
            contents=count_contents,
        )
        total_tokens = getattr(response, "total_tokens", None)
        if total_tokens is None:
            raise RuntimeError(f"Gemini count_tokens returned no total_tokens for model {self.model_name}")
        return int(total_tokens)

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.initialize()
        system_instruction, contents = self._build_contents(messages)
        return self._count_tokens(
            system_instruction=system_instruction,
            contents=contents,
        )

    def _build_config(
        self,
        sampling_kwargs: dict[str, Any] | None,
        system_instruction: str | None,
    ) -> Any:
        from google.genai import types

        kwargs = dict(sampling_kwargs or {})
        include_thought_summaries = bool(kwargs.pop("include_thought_summaries", False))
        config_kwargs: dict[str, Any] = {
            "candidate_count": 1,
            "temperature": GEMINI_DEFAULT_TEMPERATURE,
            "max_output_tokens": int(kwargs["max_tokens"]) if kwargs.get("max_tokens") is not None else GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if kwargs.get("top_p") is not None:
            config_kwargs["top_p"] = float(kwargs["top_p"])
        if kwargs.get("top_k") is not None:
            config_kwargs["top_k"] = int(kwargs["top_k"])
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if include_thought_summaries:
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        return types.GenerateContentConfig(**config_kwargs)

    def generate_batch_results(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        result_callback: Callable[[int, InferenceResult], None] | None = None,
        model_role: str | None = None,
    ) -> list[InferenceResult]:
        self.initialize()
        assert self._client is not None
        max_workers = min(len(contexts), self._max_parallel_requests(batch_size))
        results: list[InferenceResult | None] = [None] * len(contexts)
        requested_max_tokens = (
            int(sampling_kwargs["max_tokens"])
            if sampling_kwargs is not None and sampling_kwargs.get("max_tokens") is not None
            else GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        )

        def _run_one(messages: list[dict[str, str]]) -> InferenceResult:
            transient_retries = 0
            for _ in range(1 + _TRANSIENT_MAX_RETRIES):
                system_instruction, contents = self._build_contents(messages)
                config = self._build_config(sampling_kwargs, system_instruction)
                start = time.perf_counter()
                try:
                    response = self._client.models.generate_content(
                        model=self._api_model_name(),
                        contents=contents,
                        config=config,
                    )
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    usage = _usage_to_dict(getattr(response, "usage_metadata", None))
                    visible_text, thought_summary, thought_meta = _response_channels(response)
                    provider_meta = {
                        "response_id": getattr(response, "response_id", None),
                        "model_version": getattr(response, "model_version", None),
                        **thought_meta,
                        **_sdk_http_response_meta(response),
                    }
                    provider_meta = {key: value for key, value in provider_meta.items() if value is not None}
                    result = InferenceResult(
                        text=visible_text or _response_text(response),
                        thought_summary=thought_summary,
                        thought_summary_available=bool(thought_summary is not None),
                        usage=usage,
                        latency_ms=latency_ms,
                        provider_meta=provider_meta,
                        retries=transient_retries,
                        error=None,
                        model_role=model_role or self.model_role,
                        model_name=self._api_model_name(),
                        provider_name=self.provider_name,
                        token_budget={
                            "context_len_tokens": self.context_len_tokens,
                            "requested_max_output_tokens": requested_max_tokens,
                            "prompt_token_count": usage.get("prompt_token_count"),
                            "total_token_count": usage.get("total_token_count"),
                        },
                    )
                    maybe_record_result(result)
                    return result
                except SpendLimitExceeded:
                    raise
                except Exception as exc:
                    if _is_transient_error(exc) and transient_retries < _TRANSIENT_MAX_RETRIES:
                        transient_retries += 1
                        delay = min(_TRANSIENT_BASE_DELAY * (2 ** (transient_retries - 1)), _TRANSIENT_MAX_DELAY)
                        delay *= 0.5 + random.random()
                        print(
                            f"[gemini] Transient error (attempt {transient_retries}/{_TRANSIENT_MAX_RETRIES}), "
                            f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"Gemini API request failed for model {self.model_name}: {type(exc).__name__}: {exc}"
                    ) from exc
            raise AssertionError("unreachable")

        if max_workers <= 1:
            for idx, messages in enumerate(contexts):
                results[idx] = _run_one(messages)
                if result_callback is not None:
                    result_callback(idx, results[idx])
                if progress_callback is not None:
                    progress_callback(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(_run_one, messages): idx
                    for idx, messages in enumerate(contexts)
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    if result_callback is not None:
                        result_callback(idx, results[idx])
                    if progress_callback is not None:
                        progress_callback(1)

        return [result for result in results if result is not None]

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        result_callback: Callable[[int, InferenceResult], None] | None = None,
        model_role: str | None = None,
    ) -> list[str]:
        return [
            result.text
            for result in self.generate_batch_results(
                contexts,
                batch_size=batch_size,
                sampling_kwargs=sampling_kwargs,
                progress_callback=progress_callback,
                result_callback=result_callback,
                model_role=model_role,
            )
        ]
