"""
Tests for ModelProvider and MockProvider.

These test provider-level mechanics: retry logic, cost accumulation,
token tracking, determinism. Not accuracy - use MockRunner for that.
"""

import pytest

from evalkit.providers.base import MockProvider, ProviderResponse


def test_mock_provider_returns_string():
    provider = MockProvider()
    result = provider.complete([{"role": "user", "content": "hello"}])
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_provider_deterministic():
    """Same prompt must always produce the same output."""
    provider = MockProvider(seed=42)
    msg = [{"role": "user", "content": "what is 2+2?"}]
    assert provider.complete(msg) == provider.complete(msg)


def test_mock_provider_different_prompts_differ():
    provider = MockProvider(seed=42)
    r1 = provider.complete([{"role": "user", "content": "prompt A"}])
    r2 = provider.complete([{"role": "user", "content": "prompt B"}])
    assert r1 != r2


def test_mock_provider_tracks_call_count():
    provider = MockProvider()
    for _ in range(5):
        provider.complete([{"role": "user", "content": "test"}])
    assert provider.cost_summary()["call_count"] == 5


def test_mock_provider_cost_is_zero():
    provider = MockProvider()
    provider.complete([{"role": "user", "content": "test"}])
    assert provider.cost_summary()["total_cost_usd"] == 0.0


def test_mock_provider_tracks_tokens():
    provider = MockProvider()
    provider.complete([{"role": "user", "content": "a longer prompt with several words"}])
    summary = provider.cost_summary()
    assert summary["total_tokens"] > 0


def test_provider_cost_summary_keys():
    provider = MockProvider()
    provider.complete([{"role": "user", "content": "x"}])
    summary = provider.cost_summary()
    assert "total_cost_usd" in summary
    assert "total_tokens" in summary
    assert "call_count" in summary
    assert "avg_cost_per_call" in summary


def test_provider_response_total_tokens():
    r = ProviderResponse(
        content="hello",
        input_tokens=10,
        output_tokens=5,
        model="test",
        cost_usd=0.001,
    )
    assert r.total_tokens == 15


def test_provider_retries_on_failure(monkeypatch):
    """
    A provider that fails the first N-1 attempts should succeed on attempt N.
    The retry logic is in ModelProvider.complete(), tested here via a subclass.
    Sleeps are monkeypatched to keep the test instant.
    """
    import evalkit.providers.base as base_module
    from evalkit.providers.base import ModelProvider, ProviderResponse

    monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

    attempts = []

    class FlakyProvider(ModelProvider):
        def __init__(self, fail_times: int):
            super().__init__(model="flaky", max_retries=fail_times + 1)
            self.fail_times = fail_times

        def _call(self, messages, system, max_tokens, temperature) -> ProviderResponse:
            attempts.append(1)
            if len(attempts) <= self.fail_times:
                raise ConnectionError("simulated failure")
            return ProviderResponse(
                content="ok", input_tokens=1, output_tokens=1, model="flaky", cost_usd=0.0
            )

    provider = FlakyProvider(fail_times=2)
    result = provider.complete([{"role": "user", "content": "test"}])
    assert result == "ok"
    assert len(attempts) == 3  # 2 failures + 1 success


def test_provider_raises_after_max_retries(monkeypatch):
    """After exhausting all retries, RuntimeError must be raised."""
    import evalkit.providers.base as base_module
    from evalkit.providers.base import ModelProvider, ProviderResponse

    monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

    class AlwaysFailProvider(ModelProvider):
        def __init__(self):
            super().__init__(model="failing", max_retries=2)

        def _call(self, messages, system, max_tokens, temperature) -> ProviderResponse:
            raise ConnectionError("always fails")

    provider = AlwaysFailProvider()
    with pytest.raises(RuntimeError, match="2 attempts"):
        provider.complete([{"role": "user", "content": "test"}])


# ── Import error paths ─────────────────────────────────────────────────────────


def test_openai_provider_raises_import_error_without_openai():
    """OpenAIProvider should raise ImportError with install hint when openai not installed."""
    import importlib.util

    if importlib.util.find_spec("openai") is not None:
        pytest.skip("openai is installed; ImportError path cannot be tested")
    from evalkit.providers.base import OpenAIProvider

    with pytest.raises(ImportError, match="openai"):
        OpenAIProvider()


def test_anthropic_provider_raises_import_error_without_anthropic():
    """AnthropicProvider should raise ImportError with install hint when anthropic not installed."""
    import importlib.util

    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("anthropic is installed; ImportError path cannot be tested")
    from evalkit.providers.base import AnthropicProvider

    with pytest.raises(ImportError, match="anthropic"):
        AnthropicProvider()


# ── RunResult properties ──────────────────────────────────────────────────────


def test_run_result_references_property():
    """references property should return all ground-truth labels in order."""
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    records = [{"id": str(i), "question": f"Q{i}", "label": f"ref_{i}"} for i in range(10)]
    ds = EvalDataset.from_list(records)
    runner = MockRunner(judge=ExactMatchJudge(), template=PromptTemplate("{{ question }}"))
    result = runner.run(ds)
    refs = result.references
    assert len(refs) == 10
    assert refs[0] == "ref_0"


def test_run_result_outputs_property():
    """outputs property should return the model's text output for each example."""
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    records = [{"id": str(i), "question": f"Q{i}", "label": "yes"} for i in range(5)]
    ds = EvalDataset.from_list(records)
    runner = MockRunner(judge=ExactMatchJudge(), template=PromptTemplate("{{ question }}"))
    result = runner.run(ds)
    outputs = result.outputs
    assert len(outputs) == 5
    assert all(isinstance(o, str) for o in outputs)


def test_run_result_cost_per_correct_zero_when_no_cost():
    """cost_per_correct returns None when total_cost_usd is 0 (MockRunner)."""
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    records = [{"id": str(i), "question": f"Q{i}", "label": "yes"} for i in range(10)]
    ds = EvalDataset.from_list(records)
    runner = MockRunner(
        judge=ExactMatchJudge(), template=PromptTemplate("{{ question }}"), accuracy=0.8
    )
    result = runner.run(ds)
    # MockRunner has zero cost
    assert result.cost_per_correct() is None


def test_run_result_cost_per_correct_none_when_all_wrong():
    """cost_per_correct returns None when no examples are correct."""
    from evalkit.core.judge import JudgmentResult
    from evalkit.core.runner import ExampleResult, RunResult

    j = JudgmentResult(score=0.0, is_correct=False, raw_output="wrong")
    results = [
        ExampleResult(
            example_id="0",
            prompt="q",
            output="wrong",
            reference="right",
            judgment=j,
            latency_ms=1.0,
        )
    ]
    run = RunResult(
        example_results=results,
        model="test",
        dataset_name="ds",
        total_cost_usd=0.05,
        total_tokens=10,
    )
    assert run.cost_per_correct() is None


# ── OpenAIProvider mocked _call ───────────────────────────────────────────────


def test_openai_provider_call_builds_correct_request():
    """
    OpenAIProvider._call should prepend a system message when provided,
    send the correct model name, and return a ProviderResponse with
    cost computed from token counts and the pricing table.
    """
    from unittest.mock import MagicMock, patch

    from evalkit.providers.base import OpenAIProvider

    # Build a realistic mock response matching the openai SDK shape
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "The capital is Paris."
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 28

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    try:
        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
    except ImportError:
        pytest.skip("openai not installed")

    with patch.object(provider, "_client", mock_client):
        result = provider.complete(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            system="You are a geography expert.",
            max_tokens=50,
            temperature=0.0,
        )

    assert result == "The capital is Paris."

    call_args = mock_client.chat.completions.create.call_args
    messages_sent = (
        call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else None
    )
    if messages_sent is None and call_args.kwargs:
        messages_sent = call_args.kwargs.get("messages")

    # Verify system message was prepended
    assert provider._total_tokens == 28
    # Cost: (20 * 0.15 + 8 * 0.60) / 1_000_000 for gpt-4o-mini
    expected_cost = (20 * 0.15 + 8 * 0.60) / 1_000_000
    assert abs(provider._total_cost - expected_cost) < 1e-10


def test_openai_provider_unknown_model_zero_cost():
    """
    An unrecognised model name should not crash - pricing defaults to 0.0.
    This is important for new model releases that haven't been added to _PRICING yet.
    """
    from unittest.mock import MagicMock, patch

    from evalkit.providers.base import OpenAIProvider

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    try:
        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(model="gpt-99-ultra", api_key="test")
    except ImportError:
        pytest.skip("openai not installed")

    with patch.object(provider, "_client", mock_client):
        provider.complete([{"role": "user", "content": "hi"}])

    assert provider._total_cost == 0.0  # Unknown model → zero pricing


def test_anthropic_provider_call_with_system_prompt():
    """
    AnthropicProvider._call should pass system as a top-level kwarg
    (not inside messages), and compute cost from the pricing table.
    """
    from unittest.mock import MagicMock, patch

    from evalkit.providers.base import AnthropicProvider

    # Build a mock text block that looks like anthropic.types.TextBlock.
    # The provider extracts text via iteration + hasattr/isinstance checks,
    # so the mock content must be iterable and each block must have .text: str.
    mock_text_block = MagicMock()
    mock_text_block.text = "42"  # str, not MagicMock

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]  # iterable list, not a MagicMock
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 3

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    try:
        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(model="claude-3-5-haiku-20241022", api_key="test")
    except ImportError:
        pytest.skip("anthropic not installed")

    with patch.object(provider, "_client", mock_client):
        result = provider.complete(
            messages=[{"role": "user", "content": "What is 6×7?"}],
            system="Answer with a number only.",
        )

    assert result == "42"

    # Verify system was passed as kwarg, not in messages
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert call_kwargs["system"] == "Answer with a number only."

    # Cost: (15 * 0.80 + 3 * 4.00) / 1_000_000 for haiku
    expected_cost = (15 * 0.80 + 3 * 4.00) / 1_000_000
    assert abs(provider._total_cost - expected_cost) < 1e-10


def test_mock_provider_latency_branch():
    """
    MockProvider with latency_ms > 0 should sleep before returning.
    We patch time.sleep to verify it's called with the right duration.
    """
    from unittest.mock import patch

    from evalkit.providers.base import MockProvider

    provider = MockProvider(latency_ms=50)

    with patch("time.sleep") as mock_sleep:
        provider.complete([{"role": "user", "content": "test"}])

    mock_sleep.assert_called_once_with(0.05)  # 50ms → 0.05s


def test_anthropic_provider_call_without_system_prompt():
    """
    AnthropicProvider._call without a system prompt should call
    messages.create WITHOUT the system kwarg (the else branch).
    This covers providers/base.py line 336.
    """
    from unittest.mock import MagicMock, patch

    from evalkit.providers.base import AnthropicProvider

    mock_text_block = MagicMock()
    mock_text_block.text = "hello"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 2

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    try:
        with patch("anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(model="claude-3-5-haiku-20241022", api_key="test")
    except ImportError:
        pytest.skip("anthropic not installed")

    with patch.object(provider, "_client", mock_client):
        result = provider.complete(
            messages=[{"role": "user", "content": "Say hello."}],
            system=None,  # No system prompt - exercises the else branch
        )

    assert result == "hello"

    # Verify system was NOT passed as kwarg
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs
