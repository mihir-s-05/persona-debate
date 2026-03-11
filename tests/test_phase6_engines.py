from __future__ import annotations

from types import SimpleNamespace

import pytest

from debate_v_majority.cli.main_impl import SubsetItem, run_debate, run_sampled
from debate_v_majority.engines import gemini_api
from debate_v_majority.engines import (
    BaseInferenceEngine,
    GeminiInferenceEngine,
    InferenceResult,
    create_inference_engine,
    make_text_result,
)


class _TypedFakeEngine(BaseInferenceEngine):
    def __init__(self, *, model_name: str, provider_name: str, outputs_by_call: list[list[str]]) -> None:
        super().__init__(model_name=model_name)
        self.provider_name = provider_name
        self.outputs_by_call = list(outputs_by_call)
        self.calls: list[tuple[list[list[dict[str, str]]], str | None]] = []
        self._context_len_tokens = 8192

    @property
    def context_len_tokens(self) -> int:
        return self._context_len_tokens

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        del messages
        return 11

    def generate_batch_results(
        self,
        contexts: list[list[dict[str, str]]],
        batch_size: int | None = None,
        *,
        sampling_kwargs: dict[str, object] | None = None,
        progress_callback=None,
        model_role: str | None = None,
    ) -> list[InferenceResult]:
        del batch_size
        del sampling_kwargs
        self.calls.append((contexts, model_role))
        outputs = self.outputs_by_call.pop(0)
        assert len(outputs) == len(contexts)
        if progress_callback is not None:
            progress_callback(len(outputs))
        return [
            make_text_result(
                text,
                model_name=self.model_name,
                provider_name=self.provider_name,
                model_role=model_role,
                usage={"prompt_token_count": 11, "total_token_count": 17},
                latency_ms=13,
                provider_meta={"call_index": len(self.calls)},
                token_budget={"context_len_tokens": self.context_len_tokens, "requested_max_output_tokens": 4096},
            )
            for text in outputs
        ]


def test_create_inference_engine_auto_routes_gemini(monkeypatch):
    def _fake_initialize(self):
        self._client = object()

    monkeypatch.setattr(GeminiInferenceEngine, "initialize", _fake_initialize)
    engine = create_inference_engine(model_name="gemini-3-flash-preview")
    assert isinstance(engine, GeminiInferenceEngine)
    assert engine.provider_name == "gemini"


def test_gemini_inference_engine_prefers_dotenv_key_over_constructor_and_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text('GEMINI_API_KEY="dotenv-key"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")
    gemini_api._dotenv_cache.clear()

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview", api_key="constructor-key")

    assert engine._api_key == "dotenv-key"


def test_gemini_inference_engine_emits_typed_usage_and_provider_metadata():
    class _FakeUsage:
        def model_dump(self, exclude_none: bool = True):
            del exclude_none
            return {
                "prompt_token_count": 21,
                "candidates_token_count": 8,
                "total_token_count": 29,
                "traffic_type": "ON_DEMAND",
            }

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            assert model == "gemini-3-flash-preview"
            assert config.max_output_tokens == 256
            assert config.system_instruction == "system policy"
            assert config.temperature == 1.0
            assert (
                getattr(config.thinking_config, "include_thoughts", None) is True
                or getattr(config.thinking_config, "includeThoughts", None) is True
            )
            assert getattr(config, "top_p", None) == 0.2
            assert getattr(config, "top_k", None) == 4
            assert contents[0].role == "user"
            return SimpleNamespace(
                text=None,
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="thought summary", thought=True, thought_signature=b"abc"),
                                SimpleNamespace(text="final answer", thought=False, thought_signature=None),
                            ]
                        )
                    )
                ],
                usage_metadata=_FakeUsage(),
                response_id="resp-123",
                model_version="gemini-3-flash-preview-001",
                sdk_http_response=SimpleNamespace(
                    status_code=200,
                    url="https://example.test/generate",
                    headers={"x-goog-request-id": "req-456"},
                ),
            )

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview", api_key="test-key", max_model_len=16384)
    engine._client = SimpleNamespace(models=_FakeModels())
    results = engine.generate_batch_results(
        [[{"role": "system", "content": "system policy"}, {"role": "user", "content": "hello"}]],
        sampling_kwargs={"max_tokens": 256, "top_p": 0.2, "top_k": 4, "include_thought_summaries": True},
        model_role="debater",
    )
    assert len(results) == 1
    result = results[0]
    assert result.text == "final answer"
    assert result.thought_summary == "thought summary"
    assert result.thought_summary_available is True
    assert result.usage["prompt_token_count"] == 21
    assert result.provider_meta["response_id"] == "resp-123"
    assert result.provider_meta["request_id"] == "req-456"
    assert result.provider_meta["thought_summary_available"] is True
    assert result.provider_meta["thought_part_count"] == 1
    assert result.provider_meta["thought_signature_count"] == 1
    assert result.model_role == "debater"
    assert result.token_budget["requested_max_output_tokens"] == 256


def test_gemini_inference_engine_raises_provider_error_with_default_token_budget():
    progress_updates: list[int] = []

    class _BoomModels:
        def generate_content(self, *, model, contents, config):
            del model
            del contents
            assert config.max_output_tokens == 4096
            raise RuntimeError("provider unavailable")

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview", api_key="test-key", max_model_len=16384)
    engine._client = SimpleNamespace(models=_BoomModels())
    with pytest.raises(RuntimeError, match="Gemini API request failed.*provider unavailable"):
        engine.generate_batch_results(
            [[{"role": "user", "content": "hello"}]],
            sampling_kwargs=None,
            model_role="judge",
            progress_callback=progress_updates.append,
        )

    assert progress_updates == []


def test_gemini_inference_engine_uses_explicit_cached_content_for_controlled_prefix():
    class _FakeUsage:
        def model_dump(self, exclude_none: bool = True):
            del exclude_none
            return {
                "prompt_token_count": 52,
                "cached_content_token_count": 41,
                "candidates_token_count": 9,
                "total_token_count": 61,
            }

    class _FakeCaches:
        def __init__(self) -> None:
            self.created = []
            self.deleted = []

        def create(self, *, model, config):
            self.created.append((model, config))
            return SimpleNamespace(name="cachedContents/debate-prefix-1")

        def delete(self, *, name):
            self.deleted.append(name)
            return None

    class _FakeModels:
        def __init__(self) -> None:
            self.calls = []

        def count_tokens(self, *, model, contents, config=None):
            del model
            del contents
            del config
            return SimpleNamespace(total_tokens=2048)

        def generate_content(self, *, model, contents, config):
            self.calls.append((model, contents, config))
            assert config.cached_content == "cachedContents/debate-prefix-1"
            assert config.max_output_tokens == 128
            assert config.temperature == 1.0
            assert contents[0].role == "user"
            assert "peer critique" in contents[0].parts[0].text
            return SimpleNamespace(
                text="updated answer",
                usage_metadata=_FakeUsage(),
                response_id="resp-cache-1",
                model_version="gemini-3-flash-preview-001",
                sdk_http_response=SimpleNamespace(
                    status_code=200,
                    url="https://example.test/generate",
                    headers={"x-goog-request-id": "req-cache-1"},
                ),
            )

    fake_caches = _FakeCaches()
    fake_models = _FakeModels()
    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview", api_key="test-key", max_model_len=16384)
    engine._client = SimpleNamespace(models=fake_models, caches=fake_caches)
    contexts = [[
        {"role": "cache_control", "content": "", "cache_prefix_message_count": "3", "cache_display_name": "debate-prefix", "cache_ttl_seconds": "7200", "cache_scope": "debate_round_prefix"},
        {"role": "system", "content": "system policy"},
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "prior private reasoning"},
        {"role": "user", "content": "peer critique"},
    ]]
    results = engine.generate_batch_results(
        contexts,
        sampling_kwargs={"max_tokens": 128},
        model_role="debater",
    )
    assert len(results) == 1
    result = results[0]
    assert result.provider_meta["explicit_cache_used"] is True
    assert result.provider_meta["explicit_cache_name"] == "cachedContents/debate-prefix-1"
    assert result.provider_meta["explicit_cache_scope"] == "debate_round_prefix"
    assert result.provider_meta["explicit_cache_created"] is True
    assert result.usage["cached_content_token_count"] == 41
    assert len(fake_caches.created) == 1
    cache_model, cache_config = fake_caches.created[0]
    assert cache_model == "models/gemini-3-flash-preview"
    assert cache_config.system_instruction == "system policy"
    assert cache_config.ttl == "7200s"
    assert cache_config.contents[0].role == "user"
    assert cache_config.contents[1].role == "model"

    second = engine.generate_batch_results(
        contexts,
        sampling_kwargs={"max_tokens": 128},
        model_role="debater",
    )[0]
    assert second.provider_meta["explicit_cache_created"] is False
    assert len(fake_caches.created) == 1


def test_gemini_inference_engine_skips_explicit_cache_when_prefix_is_below_minimum():
    class _FakeUsage:
        def model_dump(self, exclude_none: bool = True):
            del exclude_none
            return {
                "prompt_token_count": 18,
                "candidates_token_count": 6,
                "total_token_count": 24,
            }

    class _FakeCaches:
        def __init__(self) -> None:
            self.created = []

        def create(self, *, model, config):
            self.created.append((model, config))
            raise AssertionError("cache creation should have been skipped")

    class _FakeModels:
        def count_tokens(self, *, model, contents, config=None):
            del model
            del contents
            del config
            return SimpleNamespace(total_tokens=120)

        def generate_content(self, *, model, contents, config):
            assert model == "gemini-3-flash-preview"
            assert getattr(config, "cached_content", None) is None
            assert len(contents) == 3
            return SimpleNamespace(
                text="uncached answer",
                usage_metadata=_FakeUsage(),
                response_id="resp-cache-skip",
                model_version="gemini-3-flash-preview-001",
                sdk_http_response=SimpleNamespace(
                    status_code=200,
                    url="https://example.test/generate",
                    headers={"x-goog-request-id": "req-cache-skip"},
                ),
            )

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview", api_key="test-key", max_model_len=16384)
    engine._client = SimpleNamespace(models=_FakeModels(), caches=_FakeCaches())
    context = [[
        {"role": "cache_control", "content": "", "cache_prefix_message_count": "3", "cache_display_name": "debate-prefix", "cache_ttl_seconds": "3600", "cache_scope": "debate_round_prefix"},
        {"role": "system", "content": "system policy"},
        {"role": "user", "content": "short question"},
        {"role": "assistant", "content": "short prior"},
        {"role": "user", "content": "peer critique"},
    ]]
    result = engine.generate_batch_results(
        context,
        sampling_kwargs={"max_tokens": 128},
        model_role="debater",
    )[0]
    assert result.provider_meta["explicit_cache_requested"] is True
    assert result.provider_meta["explicit_cache_used"] is False
    assert result.provider_meta["explicit_cache_skip_reason"] == "below_min_tokens"
    assert result.provider_meta["explicit_cache_prefix_tokens"] == 120


def test_run_sampled_emits_sample_call_metadata():
    item = SubsetItem(
        subset_id=0,
        orig_id=1,
        item_uid="aime25:item-1",
        dataset_revision="rev-test",
        item_display_id="aime25-item-1",
        raw_task={"problem": "What is 1 + 1?", "answer": "2"},
        dataset_meta={"dataset": "aime25"},
    )
    engine = _TypedFakeEngine(
        model_name="debater-model",
        provider_name="fake-provider",
        outputs_by_call=[["Reasoning...\n\\boxed{2}"]],
    )
    rows = run_sampled(
        dataset="aime25",
        items=[item],
        engine=engine,
        n_samples=1,
        batch_size=1,
        mode_label="single",
    )
    row = rows[0]
    assert row["sample_call_metadata"][0]["provider_name"] == "fake-provider"
    assert row["sample_call_metadata"][0]["model_role"] == "debater"
    assert row["sample_call_metadata"][0]["usage"]["prompt_token_count"] == 11
    assert row["sample_token_usage"][0]["input_tokens"] == 11
    assert row["sample_call_metadata"][0]["request_message_token_counts"] == {"prompt_tokens": 11}
    assert row["token_usage_summary"]["n_calls"] == 1


def test_run_debate_emits_separate_debater_and_judge_call_metadata():
    item = SubsetItem(
        subset_id=0,
        orig_id=1,
        item_uid="hle:item-1",
        dataset_revision="rev-test",
        item_display_id="hle-item-1",
        raw_task={
            "id": "hle-item-1",
            "question": "Which option is correct?\nA) red\nB) blue\nC) green",
            "answer": "blue",
            "answer_type": "multipleChoice",
            "category": "physics",
            "Verified_Classes": "Gold subset",
            "source_variant": "verified",
            "source_subset_label": "Gold subset",
        },
        dataset_meta={"dataset": "hle"},
    )
    debater_engine = _TypedFakeEngine(
        model_name="debater-model",
        provider_name="fake-debater",
        outputs_by_call=[["Confidence: 0.8\n\\boxed{B}", "Confidence: 0.7\n\\boxed{B}"]],
    )
    judge_engine = _TypedFakeEngine(
        model_name="judge-model",
        provider_name="fake-judge",
        outputs_by_call=[["Confidence: 0.9\n\\boxed{B}"]],
    )
    results = run_debate(
        dataset="hle",
        items=[item],
        engine=debater_engine,
        judge_engine=judge_engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=2,
        use_personas=False,
    )
    row = results[0][0]
    assert row["debater_backend"] == "fake-debater"
    assert row["agent_round_outputs"][0][0]["call_metadata"]["model_name"] == "debater-model"
    assert row["agent_round_outputs"][0][0]["call_metadata"]["model_role"] == "debater"
    assert row["agent_round_outputs"][0][0]["call_metadata"]["request_message_token_counts"] == {"prompt_tokens": 11}
    assert row["judge_trace"]["judge_backend"] == "fake-judge"
    assert row["judge_trace"]["judge_raw_call_metadata"]["model_name"] == "judge-model"
    assert row["judge_trace"]["judge_raw_call_metadata"]["model_role"] == "judge"
    assert row["judge_trace"]["judge_raw_call_metadata"]["request_message_token_counts"] == {"prompt_tokens": 11}
    assert row["debater_round_token_usage"][0]["n_calls"] == 2
    assert row["judge_round_token_usage"]["aggregate"]["n_calls"] == 1
