from __future__ import annotations

from types import SimpleNamespace

import pytest

from debate_v_majority.engines import create_inference_engine
from debate_v_majority.engines.base import InferenceResult, ensure_inference_results


def test_ensure_inference_results_wraps_legacy_text_engine():
    class _LegacyEngine:
        model_name = "legacy-model"

        def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
            if progress_callback is not None:
                progress_callback(len(contexts))
            return ["out-1", "out-2"]

    results = ensure_inference_results(
        _LegacyEngine(),
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        model_role="debater",
    )

    assert [result.text for result in results] == ["out-1", "out-2"]
    assert results[0].model_name == "legacy-model"
    assert results[0].provider_name == "legacy_text_only"
    assert results[0].model_role == "debater"


def test_ensure_inference_results_does_not_mask_legacy_engine_type_errors():
    class _BrokenLegacyEngine:
        model_name = "broken-legacy-model"

        def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
            del contexts, batch_size, sampling_kwargs, progress_callback
            raise TypeError("bug inside legacy engine")

    with pytest.raises(TypeError, match="bug inside legacy engine"):
        ensure_inference_results(
            _BrokenLegacyEngine(),
            [[{"role": "user", "content": "a"}]],
            model_role="debater",
        )


def test_create_inference_engine_routes_gemini(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, object] = {}

    class _FakeGeminiEngine:
        def __init__(
            self,
            model_name: str,
            *,
            model_role: str | None = None,
            api_key: str | None = None,
            max_model_len: int | None = None,
            api_version: str | None = None,
        ):
            created["model_name"] = model_name
            created["model_role"] = model_role
            created["api_key"] = api_key
            created["max_model_len"] = max_model_len
            created["api_version"] = api_version

    monkeypatch.setattr("debate_v_majority.engines.gemini_api.GeminiInferenceEngine", _FakeGeminiEngine)

    engine = create_inference_engine(
        model_name="gemini-3-flash",
        max_model_len=8192,
    )

    assert isinstance(engine, _FakeGeminiEngine)
    assert created == {
        "model_name": "gemini-3-flash",
        "model_role": None,
        "api_key": None,
        "max_model_len": 8192,
        "api_version": None,
    }


def test_create_inference_engine_rejects_non_gemini_models():
    with pytest.raises(ValueError, match="Gemini models only"):
        create_inference_engine(
            model_name="Qwen/Qwen3-8B",
        )
