from __future__ import annotations

import hashlib
import json
import mimetypes
import os
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
GEMINI_CACHE_CONTROL_ROLE = "cache_control"
GEMINI_DEFAULT_CACHE_TTL_SECONDS = 3600


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


def _message_signature(messages: list[dict[str, Any]]) -> str:
    payload = [
        {
            "role": str(message.get("role") or ""),
            "content": message.get("content"),
        }
        for message in messages
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        self._cache_registry: dict[str, dict[str, Any]] = {}
        self._created_cache_names: set[str] = set()

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
        if self._client is not None:
            for cache_name in list(self._created_cache_names):
                try:
                    self._client.caches.delete(name=cache_name)
                except Exception:
                    pass
        self._cache_registry.clear()
        self._created_cache_names.clear()
        self._client = None

    def _split_cache_control(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not messages:
            return None, messages
        first = messages[0]
        if str(first.get("role") or "").strip().lower() != GEMINI_CACHE_CONTROL_ROLE:
            return None, messages
        payload_messages = list(messages[1:])
        raw_prefix_count = first.get("cache_prefix_message_count")
        try:
            prefix_count = int(raw_prefix_count)
        except (TypeError, ValueError):
            prefix_count = 0
        prefix_count = max(0, min(prefix_count, len(payload_messages)))
        raw_ttl = first.get("cache_ttl_seconds")
        try:
            ttl_seconds = int(raw_ttl)
        except (TypeError, ValueError):
            ttl_seconds = GEMINI_DEFAULT_CACHE_TTL_SECONDS
        ttl_seconds = max(1, ttl_seconds)
        control = {
            "prefix_count": prefix_count,
            "display_name": str(first.get("cache_display_name") or "").strip() or None,
            "ttl_seconds": ttl_seconds,
            "cache_scope": str(first.get("cache_scope") or "").strip() or None,
        }
        return control, payload_messages

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

    def _cache_min_tokens(self) -> int:
        return 1024

    def _count_tokens(
        self,
        *,
        system_instruction: str | None,
        contents: list[Any],
    ) -> int:
        from google.genai import types

        assert self._client is not None
        config = None
        if system_instruction:
            config = types.CountTokensConfig(system_instruction=system_instruction)
        response = self._client.models.count_tokens(
            model=self._api_model_resource_name(),
            contents=contents,
            config=config,
        )
        total_tokens = getattr(response, "total_tokens", None)
        if total_tokens is None:
            raise RuntimeError(f"Gemini count_tokens returned no total_tokens for model {self.model_name}")
        return int(total_tokens)

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.initialize()
        _cache_control, payload_messages = self._split_cache_control(messages)
        system_instruction, contents = self._build_contents(payload_messages)
        return self._count_tokens(
            system_instruction=system_instruction,
            contents=contents,
        )

    def _build_config(
        self,
        sampling_kwargs: dict[str, Any] | None,
        system_instruction: str | None,
        *,
        cached_content: str | None = None,
    ) -> Any:
        from google.genai import types

        kwargs = dict(sampling_kwargs or {})
        include_thought_summaries = bool(kwargs.pop("include_thought_summaries", False))
        config_kwargs: dict[str, Any] = {
            "candidate_count": 1,
            "temperature": float(kwargs["temperature"]) if kwargs.get("temperature") is not None else GEMINI_DEFAULT_TEMPERATURE,
            "max_output_tokens": int(kwargs["max_tokens"]) if kwargs.get("max_tokens") is not None else GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
        }
        if kwargs.get("top_p") is not None:
            config_kwargs["top_p"] = float(kwargs["top_p"])
        if kwargs.get("top_k") is not None:
            config_kwargs["top_k"] = int(kwargs["top_k"])
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if cached_content:
            config_kwargs["cached_content"] = cached_content
        if include_thought_summaries:
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        return types.GenerateContentConfig(**config_kwargs)

    def _get_or_create_cached_content(
        self,
        *,
        prefix_messages: list[dict[str, Any]],
        display_name: str | None,
        ttl_seconds: int,
    ) -> dict[str, Any]:
        from google.genai import types

        assert self._client is not None
        cache_key = f"{self.model_name}:{_message_signature(prefix_messages)}"
        cached_entry = self._cache_registry.get(cache_key)
        if cached_entry is not None:
            return {**cached_entry, "created": False}

        system_instruction, contents = self._build_contents(prefix_messages)
        prefix_token_count = self._count_tokens(
            system_instruction=system_instruction,
            contents=contents,
        )
        min_tokens = self._cache_min_tokens()
        if prefix_token_count is not None and prefix_token_count < min_tokens:
            return {
                "cache_key": cache_key,
                "cache_name": None,
                "display_name": display_name,
                "ttl_seconds": int(ttl_seconds),
                "created": False,
                "cache_skipped": True,
                "skip_reason": "below_min_tokens",
                "prefix_token_count": int(prefix_token_count),
                "min_prefix_tokens": int(min_tokens),
            }
        cached = self._client.caches.create(
            model=self._api_model_resource_name(),
            config=types.CreateCachedContentConfig(
                contents=contents,
                system_instruction=system_instruction,
                display_name=display_name,
                ttl=f"{int(ttl_seconds)}s",
            ),
        )
        cache_name = str(getattr(cached, "name", "") or "")
        entry = {
            "cache_key": cache_key,
            "cache_name": cache_name or None,
            "display_name": display_name,
            "ttl_seconds": int(ttl_seconds),
            "created": True,
            "prefix_signature": _message_signature(prefix_messages),
            "prefix_token_count": prefix_token_count,
            "min_prefix_tokens": int(min_tokens),
        }
        if not cache_name:
            return {
                "cache_key": cache_key,
                "cache_name": None,
                "display_name": display_name,
                "ttl_seconds": int(ttl_seconds),
                "created": False,
                "cache_skipped": True,
                "skip_reason": "missing_cache_name",
                "prefix_token_count": prefix_token_count,
                "min_prefix_tokens": int(min_tokens),
            }
        self._created_cache_names.add(cache_name)
        self._cache_registry[cache_key] = dict(entry)
        return entry

    def generate_batch_results(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[InferenceResult]:
        del batch_size
        self.initialize()
        assert self._client is not None
        results: list[InferenceResult] = []
        requested_max_tokens = (
            int(sampling_kwargs["max_tokens"])
            if sampling_kwargs is not None and sampling_kwargs.get("max_tokens") is not None
            else GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
        )
        for messages in contexts:
            cache_control, payload_messages = self._split_cache_control(messages)
            provider_cache_meta: dict[str, Any] = {}
            if cache_control is not None and cache_control["prefix_count"] > 0:
                prefix_messages = payload_messages[: int(cache_control["prefix_count"])]
                live_messages = payload_messages[int(cache_control["prefix_count"]) :]
                cache_entry = self._get_or_create_cached_content(
                    prefix_messages=prefix_messages,
                    display_name=cache_control["display_name"],
                    ttl_seconds=int(cache_control["ttl_seconds"]),
                )
                provider_cache_meta = {
                    "explicit_cache_requested": True,
                    "explicit_cache_used": bool(cache_entry.get("cache_name")),
                    "explicit_cache_name": cache_entry.get("cache_name"),
                    "explicit_cache_key": cache_entry.get("cache_key"),
                    "explicit_cache_created": bool(cache_entry.get("created")),
                    "explicit_cache_scope": cache_control.get("cache_scope"),
                    "explicit_cache_prefix_message_count": int(cache_control["prefix_count"]),
                    "explicit_cache_live_message_count": len(live_messages),
                    "explicit_cache_ttl_seconds": int(cache_control["ttl_seconds"]),
                    "explicit_cache_prefix_tokens": cache_entry.get("prefix_token_count"),
                    "explicit_cache_min_prefix_tokens": cache_entry.get("min_prefix_tokens"),
                    "explicit_cache_skip_reason": cache_entry.get("skip_reason"),
                }
                system_instruction, contents = self._build_contents(live_messages)
                config = self._build_config(
                    sampling_kwargs,
                    system_instruction,
                    cached_content=cache_entry.get("cache_name"),
                )
                if not contents or not cache_entry.get("cache_name"):
                    system_instruction, contents = self._build_contents(payload_messages)
                    config = self._build_config(sampling_kwargs, system_instruction)
            else:
                system_instruction, contents = self._build_contents(payload_messages)
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
                    **provider_cache_meta,
                }
                provider_meta = {key: value for key, value in provider_meta.items() if value is not None}
                result = InferenceResult(
                    text=visible_text or _response_text(response),
                    thought_summary=thought_summary,
                    thought_summary_available=bool(thought_summary is not None),
                    usage=usage,
                    latency_ms=latency_ms,
                    provider_meta=provider_meta,
                    retries=0,
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
                results.append(result)
            except SpendLimitExceeded:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini API request failed for model {self.model_name}: {type(exc).__name__}: {exc}"
                ) from exc
            if progress_callback is not None:
                progress_callback(1)
        return results

    def generate_batch(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
        model_role: str | None = None,
    ) -> list[str]:
        return [
            result.text
            for result in self.generate_batch_results(
                contexts,
                batch_size=batch_size,
                sampling_kwargs=sampling_kwargs,
                progress_callback=progress_callback,
                model_role=model_role,
            )
        ]
