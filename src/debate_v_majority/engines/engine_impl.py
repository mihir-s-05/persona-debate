"""
Gemini-backed engine implementation.

The harness now uses the Gemini API as its only maintained inference runtime.
Sampling-config helpers remain here because the rest of the codebase already
depends on them for model-role defaults and manifest metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseInferenceEngine
from .providers import infer_provider_name

GEMINI_3_FLASH_MODEL = "gemini-3-flash-preview"


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""

    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }


_SAMPLING_CONFIG: SamplingConfig = SamplingConfig()


def load_generation_config(model_name: str) -> dict[str, Any]:
    del model_name
    return {}


def build_sampling_config(model_name: str) -> SamplingConfig:
    del model_name
    return SamplingConfig()


def get_sampling_config() -> SamplingConfig:
    return _SAMPLING_CONFIG


def set_sampling_config(config: SamplingConfig) -> None:
    global _SAMPLING_CONFIG
    _SAMPLING_CONFIG = config


def infer_native_context_len(model_name: str) -> int | None:
    """
    Infer the model context limit when possible.

    Gemini models default to the 1M-token API limit used elsewhere in the repo.
    """

    try:
        return 1_048_576 if infer_provider_name(model_name) == "gemini" else None
    except ValueError:
        return None


InferenceEngine = BaseInferenceEngine


def normalize_gemini_model_name(model_name: str | None = None) -> str:
    del model_name
    return GEMINI_3_FLASH_MODEL


def create_inference_engine(
    *,
    model_name: str,
    provider: str | None = None,
    model_role: str | None = None,
    max_model_len: int | None = None,
    gemini_api_key: str | None = None,
    gemini_api_version: str | None = None,
) -> InferenceEngine:
    provider_name = infer_provider_name(model_name, provider)
    if provider_name != "gemini":
        raise ValueError(f"Unsupported provider {provider_name!r}; only Gemini is supported.")

    from .gemini_api import GeminiInferenceEngine

    return GeminiInferenceEngine(
        model_name=normalize_gemini_model_name(model_name),
        model_role=model_role,
        api_key=gemini_api_key,
        max_model_len=max_model_len,
        api_version=gemini_api_version,
    )
