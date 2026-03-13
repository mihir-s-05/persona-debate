from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from debate_v_majority.cli.args import _build_arg_parser
from debate_v_majority.cli import engine_runtime as cli_engine_runtime
from debate_v_majority.cli import main_impl as cli_main_impl
from debate_v_majority.cli.debate_runner import run_debate
from debate_v_majority.cli.sample_runner import run_sampled
from debate_v_majority.cli.subset import _make_dataset_subset
from debate_v_majority.cli.main_impl import _merge_token_counts
from debate_v_majority.cli.main_impl import main as cli_main
from debate_v_majority.engines import GeminiInferenceEngine, InferenceResult, create_inference_engine, ensure_inference_results, infer_provider_name
from debate_v_majority.shared import PromptTokenCounter


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_png(path: Path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yX1cAAAAASUVORK5CYII="
        )
    )


class _LegacyTextEngine:
    def __init__(self) -> None:
        self.model_name = "legacy-model"
        self.provider_name = "legacy"

    def count_prompt_tokens(self, messages):
        del messages
        return 11

    def generate_batch(self, contexts, batch_size=None, sampling_kwargs=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback(len(contexts))
        return ["legacy-output" for _ in contexts]


class _TypedEngine:
    def __init__(self, outputs_by_call: list[list[str]], *, model_name: str = "gemini-3-flash-preview") -> None:
        self.outputs_by_call = list(outputs_by_call)
        self.model_name = model_name
        self.provider_name = "gemini"
        self.calls: list[dict[str, object]] = []

    def count_prompt_tokens(self, messages):
        del messages
        return 11

    def generate_batch_results(
        self,
        contexts,
        batch_size=None,
        *,
        sampling_kwargs=None,
        progress_callback=None,
        model_role=None,
    ):
        call_idx = len(self.calls)
        outputs = self.outputs_by_call[call_idx]
        assert len(outputs) == len(contexts)
        self.calls.append(
            {
                "contexts": contexts,
                "batch_size": batch_size,
                "sampling_kwargs": sampling_kwargs,
                "model_role": model_role,
            }
        )
        if progress_callback is not None:
            progress_callback(len(outputs))
        return [
            InferenceResult(
                text=output,
                usage={"prompt_token_count": 11, "candidates_token_count": 5, "total_token_count": 16},
                latency_ms=12,
                provider_meta={"mock_provider": True},
                retries=0,
                error=None,
                model_role=model_role,
                model_name=self.model_name,
                provider_name=self.provider_name,
                token_budget={"context_limit_tokens": 32768, "max_output_tokens": 128},
            )
            for output in outputs
        ]


def test_infer_provider_name_prefers_explicit_and_detects_gemini_models():
    assert infer_provider_name("gemini-3-flash-preview") == "gemini"
    with pytest.raises(ValueError, match="Gemini models only"):
        infer_provider_name("Qwen/Qwen3-8B")


def test_create_inference_engine_routes_gemini_models():
    engine = create_inference_engine(model_name="gemini-3-flash-preview")
    assert isinstance(engine, GeminiInferenceEngine)
    assert engine.context_len_tokens == 1_048_576


def test_ensure_inference_results_wraps_legacy_text_only_engine():
    engine = _LegacyTextEngine()
    results = ensure_inference_results(
        engine,
        [[{"role": "user", "content": "hello"}]],
        batch_size=1,
        model_role="debater",
    )
    assert len(results) == 1
    assert results[0].text == "legacy-output"
    assert results[0].provider_name == "legacy"
    assert results[0].model_role == "debater"


def test_prompt_token_counter_uses_exact_tokenizer_counts(monkeypatch):
    class _FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            assert tokenize is True
            assert add_generation_prompt is True
            return [0] * (len(messages) + 7)

    monkeypatch.setattr(PromptTokenCounter, "_get_tokenizer", lambda self: _FakeTokenizer())
    counter = PromptTokenCounter("gemini-3-flash-preview")
    count = counter.count_chat_tokens([{"role": "user", "content": "hello world"}])
    assert count == 8


def test_gemini_engine_generate_batch_results_extracts_usage_from_fake_client():
    class _FakeUsage:
        def model_dump(self, exclude_none=True):
            return {
                "prompt_token_count": 13,
                "candidates_token_count": 7,
                "total_token_count": 20,
            }

    class _FakeHTTPResponse:
        status_code = 200
        url = "https://example.invalid/generate"
        headers = {"x-request-id": "req-123"}

    class _FakeResponse:
        text = "gemini-output"
        usage_metadata = _FakeUsage()
        response_id = "resp-1"
        model_version = "gemini-3-flash-preview-001"
        sdk_http_response = _FakeHTTPResponse()

    class _FakeModels:
        def generate_content(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        def __init__(self):
            self.models = _FakeModels()

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = _FakeClient()
    results = engine.generate_batch_results(
        [[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]],
        sampling_kwargs={"max_tokens": 64, "top_p": 0.2, "top_k": 4},
        model_role="judge",
    )
    assert len(results) == 1
    result = results[0]
    assert result.text == "gemini-output"
    assert result.usage["total_token_count"] == 20
    assert result.provider_meta["request_id"] == "req-123"
    assert result.model_role == "judge"
    assert result.token_budget["requested_max_output_tokens"] == 64


def test_gemini_engine_count_prompt_tokens_uses_count_tokens_api():
    class _FakeModels:
        last_kwargs = None

        def count_tokens(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(total_tokens=19)

    class _FakeClient:
        def __init__(self):
            self.models = _FakeModels()

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = _FakeClient()
    assert engine.count_prompt_tokens(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    ) == 19
    kwargs = engine._client.models.last_kwargs
    assert "config" not in kwargs
    assert kwargs["contents"][0].role == "user"
    assert kwargs["contents"][0].parts[0].text == "sys"
    assert kwargs["contents"][1].role == "user"
    assert kwargs["contents"][1].parts[0].text == "hello"


def test_gemini_engine_generate_batch_results_runs_requests_in_parallel_and_preserves_order():
    class _FakeUsage:
        def model_dump(self, exclude_none=True):
            del exclude_none
            return {
                "prompt_token_count": 13,
                "candidates_token_count": 7,
                "total_token_count": 20,
            }

    class _FakeModels:
        def __init__(self):
            self._lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def generate_content(self, **kwargs):
            contents = kwargs["contents"]
            prompt_text = contents[0].parts[0].text
            with self._lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                time.sleep(0.05)
                return SimpleNamespace(
                    text=f"out:{prompt_text}",
                    usage_metadata=_FakeUsage(),
                    response_id=f"resp:{prompt_text}",
                    model_version="gemini-3-flash-preview-001",
                    sdk_http_response=SimpleNamespace(
                        status_code=200,
                        url="https://example.invalid/generate",
                        headers={"x-request-id": f"req:{prompt_text}"},
                    ),
                )
            finally:
                with self._lock:
                    self.in_flight -= 1

    fake_models = _FakeModels()
    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = SimpleNamespace(models=fake_models)
    results = engine.generate_batch_results(
        [
            [{"role": "user", "content": "alpha"}],
            [{"role": "user", "content": "beta"}],
            [{"role": "user", "content": "gamma"}],
            [{"role": "user", "content": "delta"}],
        ],
        batch_size=4,
        sampling_kwargs={"max_tokens": 64},
        model_role="debater",
    )

    assert [result.text for result in results] == [
        "out:alpha",
        "out:beta",
        "out:gamma",
        "out:delta",
    ]
    assert fake_models.max_in_flight > 1


def test_gemini_engine_supports_structured_image_parts_from_local_file(tmp_path: Path):
    image_path = tmp_path / "question.png"
    _write_png(image_path)

    class _FakeUsage:
        def model_dump(self, exclude_none=True):
            del exclude_none
            return {
                "prompt_token_count": 17,
                "candidates_token_count": 6,
                "total_token_count": 23,
            }

    class _FakeModels:
        def generate_content(self, **kwargs):
            contents = kwargs["contents"]
            assert len(contents) == 1
            assert contents[0].role == "user"
            assert contents[0].parts[0].text == "Solve from the image."
            assert contents[0].parts[1].inline_data.mime_type == "image/png"
            assert contents[0].parts[1].inline_data.data.startswith(b"\x89PNG")
            return SimpleNamespace(
                text="gemini-output",
                usage_metadata=_FakeUsage(),
                response_id="resp-structured-image",
                model_version="gemini-3-flash-preview-001",
                sdk_http_response=SimpleNamespace(
                    status_code=200,
                    url="https://example.invalid/generate",
                    headers={"x-request-id": "req-structured-image"},
                ),
            )

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = SimpleNamespace(models=_FakeModels())
    results = engine.generate_batch_results(
        [[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Solve from the image."},
                    {"type": "image", "image_uri": str(image_path), "source_key": "image"},
                ],
            }
        ]],
        sampling_kwargs={"max_tokens": 64},
        model_role="debater",
    )
    assert len(results) == 1
    assert results[0].text == "gemini-output"


def test_gemini_engine_falls_back_to_text_when_image_is_unreachable(tmp_path: Path):
    missing_image_path = tmp_path / "missing-question.png"

    class _FakeUsage:
        def model_dump(self, exclude_none=True):
            del exclude_none
            return {
                "prompt_token_count": 17,
                "candidates_token_count": 6,
                "total_token_count": 23,
            }

    class _FakeModels:
        def generate_content(self, **kwargs):
            contents = kwargs["contents"]
            assert len(contents) == 1
            assert contents[0].role == "user"
            assert contents[0].parts[0].text == "Solve from the image."
            assert "Image unavailable" in contents[0].parts[1].text
            return SimpleNamespace(
                text="gemini-output",
                usage_metadata=_FakeUsage(),
                response_id="resp-structured-image-fallback",
                model_version="gemini-3-flash-preview-001",
                sdk_http_response=SimpleNamespace(
                    status_code=200,
                    url="https://example.invalid/generate",
                    headers={"x-request-id": "req-structured-image-fallback"},
                ),
            )

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = SimpleNamespace(models=_FakeModels())
    results = engine.generate_batch_results(
        [[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Solve from the image."},
                    {
                        "type": "image",
                        "image_uri": str(missing_image_path),
                        "source_key": "image",
                        "fallback_text": "[Image unavailable: use the text prompt only.]",
                    },
                ],
            }
        ]],
        sampling_kwargs={"max_tokens": 64},
        model_role="debater",
    )
    assert len(results) == 1
    assert results[0].text == "gemini-output"


def test_gemini_engine_raises_when_remote_image_fetch_fails(monkeypatch):
    def _boom(_ref: str):
        raise ConnectionError("network down")

    class _FakeModels:
        def generate_content(self, **kwargs):
            raise AssertionError("generate_content should not be called when image fetch fails")

    monkeypatch.setattr("debate_v_majority.engines.gemini_api._bytes_from_remote_ref", _boom)

    engine = GeminiInferenceEngine(model_name="gemini-3-flash-preview")
    engine._client = SimpleNamespace(models=_FakeModels())

    with pytest.raises(RuntimeError, match="Failed to fetch explicit image 'diagram'"):
        engine.generate_batch_results(
            [[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Solve from the diagram."},
                        {
                            "type": "image",
                            "image_uri": "https://example.invalid/diagram.png",
                            "source_key": "diagram",
                        },
                    ],
                }
            ]],
            sampling_kwargs={"max_tokens": 64},
            model_role="debater",
        )


def test_run_sampled_emits_sample_call_metadata(tmp_path: Path):
    dataset_path = tmp_path / "aime25.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 2+2?", "answer": "4"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    engine = _TypedEngine(outputs_by_call=[["Reasoning... \\boxed{4}", "Another path... \\boxed{4}"]])
    rows = run_sampled(
        dataset="aime25",
        items=items,
        engine=engine,
        n_samples=2,
        batch_size=4,
        mode_label="majority",
        progress_file=io.StringIO(),
    )
    row = rows[0]
    assert row["sample_call_metadata"][0]["provider_name"] == "gemini"
    assert row["sample_call_metadata"][0]["model_role"] == "debater"
    assert row["sample_call_metadata"][0]["usage"]["total_token_count"] == 16
    assert row["token_usage_summary"]["total_tokens"] == 32


def test_run_debate_hle_on_gemini_builds_multimodal_initial_prompt(tmp_path: Path):
    dataset_path = tmp_path / "hle_multimodal.jsonl"
    image_path = tmp_path / "hle-question.png"
    _write_png(image_path)
    _write_jsonl(
        dataset_path,
        [
            {
                "id": "hle-image-1",
                "question": "Which option is correct?\nA) red\nB) blue",
                "answer": "blue",
                "answer_type": "multipleChoice",
                "category": "physics",
                "Verified_Classes": "Gold subset",
                "image": str(image_path),
            }
        ],
    )
    items, _ = _make_dataset_subset(
        dataset="hle",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
        hle_variant="verified",
    )
    debater_engine = _TypedEngine(outputs_by_call=[["Confidence: 0.8\n\\boxed{B}"]], model_name="gemini-3-flash-preview")
    judge_engine = _TypedEngine(outputs_by_call=[["\\boxed{B}"]], model_name="gemini-3-flash-preview")

    results = run_debate(
        dataset="hle",
        items=items,
        engine=debater_engine,
        judge_engine=judge_engine,
        n_agents=1,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=1,
        use_personas=False,
        progress_file=io.StringIO(),
    )

    initial_context = debater_engine.calls[0]["contexts"][0]
    assert initial_context[0]["role"] == "user"
    assert isinstance(initial_context[0]["content"], list)
    assert initial_context[0]["content"][0]["type"] == "text"
    assert "Relevant images are attached with this prompt." in initial_context[0]["content"][0]["text"]
    assert initial_context[0]["content"][1]["type"] == "image"
    assert initial_context[0]["content"][1]["image_uri"] == str(image_path)
    judge_context = judge_engine.calls[0]["contexts"][0]
    assert judge_context[-1]["role"] == "user"
    assert isinstance(judge_context[-1]["content"], list)
    assert judge_context[-1]["content"][0]["type"] == "text"
    assert "Relevant task images are attached with this judge prompt." in judge_context[-1]["content"][0]["text"]
    assert judge_context[-1]["content"][1]["type"] == "image"
    assert judge_context[-1]["content"][1]["image_uri"] == str(image_path)
    assert results[0][0]["final_judge_answer"] == "B"


def test_run_debate_emits_agent_and_judge_call_metadata(tmp_path: Path):
    dataset_path = tmp_path / "aime.jsonl"
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 1+1?", "answer": "2"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    debater_engine = _TypedEngine(outputs_by_call=[["\\boxed{2}", "\\boxed{2}"]], model_name="gemini-3-flash-preview")
    judge_engine = _TypedEngine(outputs_by_call=[["\\boxed{2}"]], model_name="gemini-3-flash-preview")

    results = run_debate(
        dataset="aime25",
        items=items,
        engine=debater_engine,
        n_agents=2,
        n_rounds=0,
        judge_rounds=[0],
        batch_size=4,
        judge_block_size=1,
        judge_engine=judge_engine,
        progress_file=io.StringIO(),
    )
    row = results[0][0]
    assert row["agent_round_outputs"][0][0]["call_metadata"]["provider_name"] == "gemini"
    assert row["agent_round_outputs"][0][0]["call_metadata"]["model_role"] == "debater"
    assert row["judge_trace"]["judge_raw_call_metadata"]["provider_name"] == "gemini"
    assert row["judge_trace"]["judge_raw_call_metadata"]["model_role"] == "judge"
    assert row["judge_trace"]["judge_raw_call_metadata"]["token_counts"]["input_tokens"] == 11
    assert row["token_usage_summary"]["debater"]["n_calls"] == 2
    assert row["token_usage_summary"]["all"]["n_calls"] == 3
    assert row["judge_round_token_usage"]["aggregate"]["total_tokens"] == 16
    assert judge_engine.calls[0]["sampling_kwargs"] == {"max_tokens": 32768}


def test_run_debate_wraps_round2_gemini_requests_with_cache_control(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "aime25.jsonl"
    monkeypatch.setenv("GEMINI_EXPLICIT_CACHE_TTL_SECONDS", "7200")
    _write_jsonl(
        dataset_path,
        [{"problem": "What is 2+2?", "answer": "4"}],
    )
    items, _ = _make_dataset_subset(
        dataset="aime25",
        test_path=dataset_path,
        n=1,
        seed=1,
        ids=[0],
        range_str=None,
    )
    debater_engine = _TypedEngine(
        outputs_by_call=[
            ["Reasoning one \\boxed{4}", "Reasoning two \\boxed{4}"],
            ["Updated one \\boxed{4}", "Updated two \\boxed{4}"],
        ],
        model_name="gemini-3-flash-preview",
    )
    judge_engine = _TypedEngine(outputs_by_call=[["\\boxed{4}"]], model_name="gemini-3-flash-preview")

    results = run_debate(
        dataset="aime25",
        items=items,
        engine=debater_engine,
        n_agents=2,
        n_rounds=1,
        judge_rounds=[1],
        batch_size=4,
        judge_block_size=1,
        judge_engine=judge_engine,
        progress_file=io.StringIO(),
    )
    assert len(debater_engine.calls) == 2
    round2_context = debater_engine.calls[1]["contexts"][0]
    assert round2_context[0]["role"] == "cache_control"
    assert round2_context[0]["cache_scope"] == "debate_round_prefix"
    assert round2_context[0]["cache_ttl_seconds"] == "7200"
    assert int(round2_context[0]["cache_prefix_message_count"]) == len(round2_context) - 2
    row = results[1][0]
    assert row["token_usage_summary"]["debater"]["n_calls"] == 4
    assert row["token_usage_summary"]["all"]["n_calls"] == 5
    assert row["agent_round_outputs"][0][1]["call_metadata"]["token_counts"]["uncached_input_tokens"] == 11


def test_merge_token_counts_preserves_nested_call_counts_and_estimation_flags():
    merged = _merge_token_counts(
        [
            {
                "n_calls": 3,
                "input_tokens": 30,
                "uncached_input_tokens": 24,
                "output_tokens": 12,
                "total_tokens": 42,
                "cached_input_tokens": 6,
                "has_estimated_inputs": True,
                "has_estimated_outputs": False,
            },
            {
                "n_calls": 2,
                "input_tokens": 20,
                "uncached_input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "cached_input_tokens": 0,
                "has_estimated_inputs": False,
                "has_estimated_outputs": True,
            },
        ]
    )
    assert merged["n_calls"] == 5
    assert merged["input_tokens"] == 50
    assert merged["cached_input_tokens"] == 6
    assert merged["has_estimated_inputs"] is True
    assert merged["has_estimated_outputs"] is True


def test_cli_main_honors_output_override_and_writes_phase7_logical_blocks(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "aime25.jsonl"
    _write_jsonl(dataset_path, [{"problem": "What is 2+2?", "answer": "4"}])
    custom_output = tmp_path / "custom" / "single.jsonl"
    manifest_path = tmp_path / "manifest.json"

    class _FakeEngine:
        model_name = "gemini-3-flash-preview"
        provider_name = "gemini"

        def shutdown(self):
            return None

    monkeypatch.setattr(cli_main_impl, "_default_dataset_test_path", lambda *args, **kwargs: dataset_path)
    monkeypatch.setattr(cli_engine_runtime, "create_inference_engine", lambda **kwargs: _FakeEngine())
    monkeypatch.setattr(
        cli_main_impl,
        "run_sampled",
        lambda **kwargs: [
            {
                "mode": "single",
                "dataset": "aime25",
                "subset_size": 1,
                "subset_seed": 123,
                "model_name": "gemini-3-flash-preview",
                "question": "What is 2+2?",
                "raw_task": {"problem": "What is 2+2?"},
                "item_uid": "aime25:0",
                "dataset_revision": "rev-1",
                "item_display_id": "aime25-item-1",
                "subset_id": 0,
                "orig_id": 0,
                "final_answer": "4",
                "final_correct": 1,
                "sample_call_metadata": [{"provider_name": "gemini"}],
                "token_usage_summary": {"n_calls": 1, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
                "--mode",
                "single",
                "--subset_ids",
                "0",
                "--model_name",
                "gemini-3-flash-preview",
                "--output",
            str(custom_output),
            "--final_manifest",
            str(manifest_path),
            "--emit_trace_level",
            "full",
            "--quiet",
        ],
    )

    cli_main()

    rows = [json.loads(line) for line in custom_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_meta"]["output_path"] == str(custom_output)
    assert row["run_meta"]["final_manifest_path"] == str(manifest_path)
    assert row["task"]["item_uid"] == "aime25:0"
    assert row["strategy"]["mode"] == "single"
    assert row["results"]["final_answer"] == "4"
    assert row["display"]["question_short"] == "What is 2+2?"
    assert row["trace"]["engine_calls"][0]["provider_name"] == "gemini"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["locked_config"]["output_schema_version"] == "phase7.logical.v1"


def test_arg_parser_accepts_judge_runtime_and_cost_flags():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "gpqa",
            "--model_name",
            "gemini-3-flash-preview",
            "--judge_runtime_model",
            "gemini-3-flash-preview",
            "--token_ledger_path",
            "out/token_ledger.jsonl",
            "--max_run_cost_usd",
            "1.5",
            "--max_total_cost_usd",
            "10.0",
        ]
    )
    assert args.judge_runtime_model == "gemini-3-flash-preview"
    assert args.token_ledger_path == "out/token_ledger.jsonl"
    assert args.max_run_cost_usd == 1.5
    assert args.max_total_cost_usd == 10.0
    assert not hasattr(args, "gpus")
    assert not hasattr(args, "gpu_memory_utilization")
    assert not hasattr(args, "enable_yarn")
    assert not hasattr(args, "enforce_eager")


def test_main_routes_provider_specific_engines_for_debate_and_persona_roles(tmp_path, monkeypatch):
    engine_calls: list[dict[str, object]] = []
    sampling_models: list[str] = []
    written_paths: list[str] = []
    run_debate_args: dict[str, object] = {}

    class _FakeEngine:
        def __init__(self, *, model_name: str, model_role: str | None):
            self.model_name = model_name
            self.provider_name = "gemini"
            self.model_role = model_role
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    def _fake_create_inference_engine(**kwargs):
        engine_calls.append(dict(kwargs))
        return _FakeEngine(
            model_name=str(kwargs["model_name"]),
            model_role=kwargs.get("model_role"),
        )

    def _fake_make_dataset_subset(**kwargs):
        del kwargs
        return (
            [
                SimpleNamespace(
                    raw_task={"problem": "What is 1+1?", "answer": "2"},
                    item_uid="aime25:item-1",
                    item_display_id="item-1",
                    dataset_revision="aime25.test.rev",
                    subset_id=0,
                    orig_id=0,
                )
            ],
            {"subset_size": 1, "seed": 7},
        )

    def _fake_build_sampling_config(model_name):
        sampling_models.append(str(model_name))
        return SimpleNamespace(max_tokens=128, temperature=0.3, top_p=0.95, top_k=-1)

    def _fake_run_debate(**kwargs):
        run_debate_args.update(kwargs)
        return {1: [{"final_correct": 1}]}

    monkeypatch.setattr("debate_v_majority.cli.engine_runtime.create_inference_engine", _fake_create_inference_engine)
    monkeypatch.setattr("debate_v_majority.cli.main_impl._make_dataset_subset", _fake_make_dataset_subset)
    monkeypatch.setattr("debate_v_majority.cli.main_impl.build_sampling_config", _fake_build_sampling_config)
    monkeypatch.setattr("debate_v_majority.cli.main_impl.set_sampling_config", lambda cfg: None)
    monkeypatch.setattr("debate_v_majority.cli.main_impl.run_debate", _fake_run_debate)
    monkeypatch.setattr("debate_v_majority.cli.main_impl._write_jsonl", lambda path, records: written_paths.append(str(path)))
    monkeypatch.setattr("debate_v_majority.cli.main_impl._timestamp_tag", lambda: "20260309_120000")

    monkeypatch.setattr(
        "sys.argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--judge_runtime_model",
            "gemini-3-flash-preview",
            "--use_personas",
            "--persona_backend",
            "llm",
            "--generator_model",
            "gemini-3-flash-preview",
            "--judge_generator_model",
            "gemini-3-flash-preview",
            "--subset_n",
            "1",
            "--n_rounds",
            "1",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main()

    assert sampling_models == ["gemini-3-flash-preview"]
    assert [(call["model_name"], call["model_role"]) for call in engine_calls] == [
        ("gemini-3-flash-preview", "debater"),
    ]
    assert all("gpus" not in call for call in engine_calls)
    assert all("gpu_memory_utilization" not in call for call in engine_calls)
    assert all("enable_yarn" not in call for call in engine_calls)
    assert all("enforce_eager" not in call for call in engine_calls)
    assert run_debate_args["engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["judge_engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["persona_generator_engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["persona_judge_engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["persona_backend"] == "llm"
    assert run_debate_args["generator_model"] == "gemini-3-flash-preview"
    assert run_debate_args["judge_generator_model"] == "gemini-3-flash-preview"
    assert len(written_paths) == 1


def test_main_auto_persona_backend_creates_llm_persona_engines(tmp_path, monkeypatch):
    engine_calls: list[dict[str, object]] = []
    run_debate_args: dict[str, object] = {}

    class _FakeEngine:
        def __init__(self, *, model_name: str, model_role: str | None):
            self.model_name = model_name
            self.provider_name = "gemini"
            self.model_role = model_role

        def shutdown(self):
            return None

    def _fake_create_inference_engine(**kwargs):
        engine_calls.append(dict(kwargs))
        return _FakeEngine(
            model_name=str(kwargs["model_name"]),
            model_role=kwargs.get("model_role"),
        )

    monkeypatch.setattr(
        "debate_v_majority.cli.main_impl._make_dataset_subset",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    raw_task={"problem": "What is 1+1?", "answer": "2"},
                    item_uid="aime25:item-1",
                    item_display_id="item-1",
                    dataset_revision="aime25.test.rev",
                    subset_id=0,
                    orig_id=0,
                )
            ],
            {"subset_size": 1, "seed": 7},
        ),
    )
    monkeypatch.setattr("debate_v_majority.cli.engine_runtime.create_inference_engine", _fake_create_inference_engine)
    monkeypatch.setattr(
        "debate_v_majority.cli.main_impl.build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=0.3, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr("debate_v_majority.cli.main_impl.set_sampling_config", lambda cfg: None)
    monkeypatch.setattr("debate_v_majority.cli.main_impl.run_debate", lambda **kwargs: (run_debate_args.update(kwargs) or {1: [{"final_correct": 1}]}))
    monkeypatch.setattr("debate_v_majority.cli.main_impl._write_jsonl", lambda path, records: None)
    monkeypatch.setattr("debate_v_majority.cli.main_impl._timestamp_tag", lambda: "20260309_120000")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--use_personas",
            "--generator_model",
            "gemini-3-flash-preview",
            "--judge_generator_model",
            "gemini-3-flash-preview",
            "--subset_n",
            "1",
            "--n_rounds",
            "1",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main()

    assert [(call["model_name"], call["model_role"]) for call in engine_calls] == [
        ("gemini-3-flash-preview", "debater"),
    ]
    assert run_debate_args["persona_backend"] == "llm"
    assert run_debate_args["persona_generator_engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["persona_judge_engine"].model_name == "gemini-3-flash-preview"
    assert run_debate_args["persona_generator_engine"] is run_debate_args["persona_judge_engine"]


def test_main_reuses_generator_engine_for_judge_generation_when_effective_provider_matches(tmp_path, monkeypatch):
    engine_calls: list[dict[str, object]] = []
    run_debate_args: dict[str, object] = {}

    class _FakeEngine:
        def __init__(self, *, model_name: str, model_role: str | None):
            self.model_name = model_name
            self.provider_name = "gemini"
            self.model_role = model_role

        def shutdown(self):
            return None

    def _fake_create_inference_engine(**kwargs):
        engine_calls.append(dict(kwargs))
        return _FakeEngine(
            model_name=str(kwargs["model_name"]),
            model_role=kwargs.get("model_role"),
        )

    monkeypatch.setattr(
        "debate_v_majority.cli.main_impl._make_dataset_subset",
        lambda **kwargs: (
            [
                SimpleNamespace(
                    raw_task={"problem": "What is 1+1?", "answer": "2"},
                    item_uid="aime25:item-1",
                    item_display_id="item-1",
                    dataset_revision="aime25.test.rev",
                    subset_id=0,
                    orig_id=0,
                )
            ],
            {"subset_size": 1, "seed": 7},
        ),
    )
    monkeypatch.setattr("debate_v_majority.cli.engine_runtime.create_inference_engine", _fake_create_inference_engine)
    monkeypatch.setattr(
        "debate_v_majority.cli.main_impl.build_sampling_config",
        lambda model_name: SimpleNamespace(max_tokens=128, temperature=0.3, top_p=0.95, top_k=-1),
    )
    monkeypatch.setattr("debate_v_majority.cli.main_impl.set_sampling_config", lambda cfg: None)
    monkeypatch.setattr("debate_v_majority.cli.main_impl.run_debate", lambda **kwargs: (run_debate_args.update(kwargs) or {1: [{"final_correct": 1}]}))
    monkeypatch.setattr("debate_v_majority.cli.main_impl._write_jsonl", lambda path, records: None)
    monkeypatch.setattr("debate_v_majority.cli.main_impl._timestamp_tag", lambda: "20260309_120000")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debate-v-majority",
            "--dataset",
            "aime25",
            "--mode",
            "debate",
            "--model_name",
            "gemini-3-flash-preview",
            "--use_personas",
            "--persona_backend",
            "llm",
            "--generator_model",
            "gemini-3-flash-preview",
            "--judge_generator_model",
            "gemini-3-flash-preview",
            "--subset_n",
            "1",
            "--n_rounds",
            "1",
            "--out_dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    cli_main()

    assert [(call["model_name"], call["model_role"]) for call in engine_calls] == [
        ("gemini-3-flash-preview", "debater"),
    ]
    assert all("gpus" not in call for call in engine_calls)
    assert all("gpu_memory_utilization" not in call for call in engine_calls)
    assert all("enable_yarn" not in call for call in engine_calls)
    assert all("enforce_eager" not in call for call in engine_calls)
    assert run_debate_args["persona_generator_engine"] is run_debate_args["persona_judge_engine"]
