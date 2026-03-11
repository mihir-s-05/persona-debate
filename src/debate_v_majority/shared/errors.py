from __future__ import annotations

import os
import re
import signal


def exception_chain_contains(err: BaseException, needles: tuple[str, ...], max_depth: int = 5) -> bool:
    """Check if any exception in the chain contains any of the needle strings."""
    cur: BaseException | None = err
    for _ in range(max_depth):
        if cur is None:
            break
        msg = str(cur).lower()
        if any(n in msg for n in needles):
            return True
        cur = cur.__cause__
    return False


def is_cuda_oom(err: BaseException) -> bool:
    """Check if an exception is a CUDA OOM error."""
    try:
        import torch
        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(err).lower()
    return ("out of memory" in msg) and ("cuda" in msg or "cublas" in msg)


def is_cuda_device_side_assert(err: BaseException) -> bool:
    """Detect CUDA device-side asserts (unrecoverable)."""
    needles = (
        "device-side assert",
        "device side assert",
        "cudaerrorassert",
        "illegal memory access",
        "an illegal memory access was encountered",
        "unspecified launch failure",
        "misaligned address",
        "warp illegal address",
    )
    return exception_chain_contains(err, needles)


def is_prompt_too_long(err: BaseException) -> bool:
    """Check if error is due to prompt exceeding context length."""
    msg = str(err).lower()
    return (
        ("longer than the maximum model length" in msg)
        or ("maximum model length" in msg and "prompt" in msg)
        or ("context length" in msg and "maximum" in msg)
        or ("exceeds the context window" in msg)
    )


def is_flash_attn_import_error(err: BaseException) -> bool:
    """Check if an exception is a FlashAttention import error."""
    msg = str(err)
    return (
        isinstance(err, ImportError)
        and ("flash_attn" in msg or "flash-attn" in msg)
        and ("undefined symbol" in msg or "flash_attn_2_cuda" in msg)
    )


def extract_prompt_length_tokens(err: BaseException) -> int | None:
    """Extract prompt length from error message."""
    msg = str(err)
    patterns = [
        r"\(length\s+(\d+)\)",
        r"prompt\s*\(length\s*(\d+)\)",
        r"prompt\s+length\s+(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def kill_process_tree(pid: int) -> None:
    """Best-effort kill of a process and all its children (Linux)."""
    try:
        children: list[int] = []
        try:
            with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
                children = [int(c) for c in f.read().split()]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass

        for child_pid in children:
            kill_process_tree(child_pid)

        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    except Exception:
        pass


__all__ = [
    "exception_chain_contains",
    "extract_prompt_length_tokens",
    "is_cuda_device_side_assert",
    "is_cuda_oom",
    "is_flash_attn_import_error",
    "is_prompt_too_long",
    "kill_process_tree",
]
