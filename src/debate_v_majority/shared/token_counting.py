from __future__ import annotations

import hashlib
import re
import struct
import threading
from typing import Any


def _hash_obj_for_cache(hasher: "hashlib._Hash", obj: Any) -> None:
    """
    Update `hasher` with a stable representation of `obj`.

    This avoids keeping large prompt strings alive in cache keys while still
    incorporating all message fields that may affect `apply_chat_template()`.
    """

    def _u(b: bytes) -> None:
        hasher.update(b)
        hasher.update(b"\0")

    def _u_int(n: int) -> None:
        try:
            hasher.update(struct.pack("!q", int(n)))
            hasher.update(b"\0")
        except (OverflowError, TypeError, ValueError):
            _u(str(n).encode("utf-8"))

    if obj is None:
        _u(b"none")
        return
    if obj is True:
        _u(b"true")
        return
    if obj is False:
        _u(b"false")
        return

    if isinstance(obj, int):
        _u(b"int")
        _u_int(obj)
        return
    if isinstance(obj, float):
        _u(b"float")
        _u(repr(obj).encode("utf-8"))
        return
    if isinstance(obj, str):
        _u(b"str")
        _u_int(len(obj))
        _u_int(hash(obj))
        prefix = obj[:64]
        suffix = obj[-64:]
        _u_int(hash(prefix))
        _u_int(hash(suffix))
        return
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        _u(b"bytes")
        _u_int(len(b))
        _u_int(hash(b))
        return

    if isinstance(obj, (list, tuple)):
        _u(b"list" if isinstance(obj, list) else b"tuple")
        _u_int(len(obj))
        for item in obj:
            _hash_obj_for_cache(hasher, item)
        return

    if isinstance(obj, dict):
        _u(b"dict")
        _u_int(len(obj))
        items = list(obj.items())
        items.sort(key=lambda kv: (repr(kv[0]), type(kv[0]).__qualname__))
        for k, v in items:
            _hash_obj_for_cache(hasher, k)
            _hash_obj_for_cache(hasher, v)
        return

    _u(b"obj")
    _u(f"{type(obj).__module__}.{type(obj).__qualname__}".encode("utf-8"))
    r = object.__repr__(obj)
    if len(r) > 2048:
        r = r[:2048] + "...(truncated)"
    _u(r.encode("utf-8", errors="surrogatepass"))


def _messages_cache_key(messages: list[dict[str, Any]]) -> tuple[int, int, str]:
    """
    Create a compact, stable cache key for chat messages.

    Returns a tuple containing a version number, message count, and a digest.
    """
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(b"messages_cache_key_v2\0")
    _hash_obj_for_cache(hasher, messages)
    return (2, len(messages), hasher.hexdigest())


class PromptTokenCounter:
    """Token counter for chat messages with lazy tokenizer loading and caching."""

    def __init__(self, model_name: str, *, cache_max_size: int = 4096) -> None:
        self._model_name = model_name
        self._tokenizer = None
        self._token_cache: dict[tuple, int] = {}
        self._cache_max_size = cache_max_size
        self._cache_lock = threading.Lock()

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        class _FallbackTokenizer:
            @staticmethod
            def encode(text: str) -> list[str]:
                tokens = re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)
                return tokens or [str(text)]

            @staticmethod
            def decode(tokens: list[str], skip_special_tokens: bool = True) -> str:
                del skip_special_tokens
                return " ".join(str(token) for token in tokens)

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            self._tokenizer = _FallbackTokenizer()
        return self._tokenizer

    def _cache_key(self, messages: list[dict[str, Any]]) -> tuple:
        """Create a full cache key including model name."""
        add_generation_prompt = True
        return (self._model_name, add_generation_prompt, _messages_cache_key(messages))

    def _get_cached(self, key: tuple) -> int | None:
        """Get cached token count if available."""
        return self._token_cache.get(key)

    def _set_cached(self, key: tuple, count: int) -> None:
        """Cache token count with simple LRU-style eviction."""
        with self._cache_lock:
            if len(self._token_cache) >= self._cache_max_size:
                keys_to_remove = list(self._token_cache.keys())[: self._cache_max_size // 2]
                for old_key in keys_to_remove:
                    self._token_cache.pop(old_key, None)
            self._token_cache[key] = count

    def count_chat_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Exact token count for chat messages (cached)."""
        cache_key = self._cache_key(messages)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        tok = self._get_tokenizer()
        if hasattr(tok, "apply_chat_template"):
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            try:
                count = int(len(ids))
            except TypeError:
                count = int(ids.shape[-1])
        else:
            text = "\n".join(
                str(message.get("role", "")) + ": " + str(message.get("content", ""))
                for message in messages
            )
            count = int(len(tok.encode(text)))

        self._set_cached(cache_key, count)
        return count

    def estimate_prompt_tokens(self, messages: list[dict[str, Any]], *, exact_if_large: int) -> int:
        """
        Exact token count retained for API compatibility with older call sites.
        """
        del exact_if_large
        return self.count_chat_tokens(messages)


def truncate_chat_messages_to_fit(
    *,
    counter: PromptTokenCounter,
    messages: list[dict[str, str]],
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], bool]:
    """
    Truncate chat history to fit within token budget.
    Returns (truncated_messages, was_truncated).
    """
    if max_prompt_tokens <= 0 or not messages:
        return messages, False

    def _fits(candidate: list[dict[str, str]]) -> bool:
        return counter.count_chat_tokens(candidate) <= max_prompt_tokens

    if _fits(messages):
        return messages, False

    msgs = [dict(m) for m in messages]

    sys_prefix: list[dict[str, str]] = []
    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        sys_prefix.append(msgs[i])
        i += 1
    tail = msgs[i:]

    while len(tail) > 1 and not _fits(sys_prefix + tail):
        tail = tail[1:]

    if _fits(sys_prefix + tail):
        return sys_prefix + tail, True

    if not tail:
        return sys_prefix, True

    msg = dict(tail[0])
    content = msg.get("content") or ""
    if not content:
        return sys_prefix + [msg], True

    tok = counter._get_tokenizer()
    content_ids = tok.encode(content)

    lo, hi = 1, len(content_ids)
    best: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        truncated = tok.decode(content_ids[-mid:], skip_special_tokens=True)
        candidate = sys_prefix + [dict(msg, content=truncated)]
        if _fits(candidate):
            best = truncated
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return sys_prefix, True

    msg["content"] = best
    return sys_prefix + [msg], True


__all__ = ["PromptTokenCounter", "truncate_chat_messages_to_fit"]
