"""
Debug Majority Debate Package

Run multi-agent debate, majority voting, or single-response inference
on AIME25, GPQA, and HLE-Verified datasets using Gemini API backends.
"""
from __future__ import annotations

from typing import Literal

# Type definitions
Mode = Literal["single", "majority", "debate", "personas"]
DatasetName = Literal["aime25", "gpqa", "hle"]
Backend = Literal["gemini"]
Parallelism = Literal["auto", "tp", "dp", "hybrid"]

__all__ = ["Mode", "DatasetName", "Backend", "Parallelism"]
