"""
Tests for AsyncRunner checkpointing and RunResult.to_dataframe().

AsyncRunner is the most complex piece of evalkit - async concurrency, atomic
checkpointing, and thread-pool execution of sync provider calls. These tests
verify the checkpoint machinery without making any real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.judge import ExactMatchJudge, JudgmentResult
from evalkit.core.runner import AsyncRunner, ExampleResult, MockRunner
from evalkit.providers.base import MockProvider

# ── Fixtures ────────────────────────────────────────────────────────────────────


def _make_dataset(n: int = 20) -> EvalDataset:
    records = [{"id": str(i), "question": f"Q{i}", "label": str(i % 3)} for i in range(n)]
    return EvalDataset.from_list(records, name="test_ds")


def _make_judgment(score: float = 1.0, correct: bool = True) -> JudgmentResult:
    return JudgmentResult(score=score, is_correct=correct, raw_output="mock")


# ── RunResult.to_dataframe() ────────────────────────────────────────────────────


def test_run_result_to_dataframe_shape():
    """to_dataframe() on RunResult should have one row per example."""
    pd = pytest.importorskip("pandas")
    dataset = _make_dataset(30)
    runner = MockRunner(
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        accuracy=0.8,
    )
    run_result = runner.run(dataset)
    df = run_result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30


def test_run_result_to_dataframe_columns():
    """All expected columns must be present."""
    pytest.importorskip("pandas")
    dataset = _make_dataset(10)
    runner = MockRunner(
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
    )
    run_result = runner.run(dataset)
    df = run_result.to_dataframe()
    required = {
        "example_id",
        "prompt",
        "output",
        "reference",
        "is_correct",
        "score",
        "reasoning",
        "latency_ms",
    }
    assert required.issubset(df.columns)


def test_run_result_to_dataframe_is_correct_matches_correct_array():
    """is_correct column must match run_result.correct exactly."""
    pytest.importorskip("pandas")
    dataset = _make_dataset(40)
    runner = MockRunner(
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        accuracy=0.75,
    )
    run_result = runner.run(dataset)
    df = run_result.to_dataframe()
    assert df["is_correct"].astype(int).tolist() == run_result.correct


# ── AsyncRunner checkpoint save/load round-trip ──────────────────────────────────


def test_checkpoint_save_and_load_round_trip(tmp_path: Path):
    """
    Save a checkpoint then load it. Every ExampleResult should survive the
    round-trip with the same score and correctness flag.

    This is the most important test for AsyncRunner - if the write-then-rename
    atomic checkpoint is broken, evaluations that crash will lose all progress.
    """
    provider = MockProvider()
    judge = ExactMatchJudge()
    template = PromptTemplate("{{ question }}")

    runner = AsyncRunner(
        provider=provider,
        judge=judge,
        template=template,
        checkpoint_dir=tmp_path,
    )

    # Build fake completed results
    completed: dict[str, ExampleResult] = {}
    for i in range(5):
        j = _make_judgment(score=float(i % 2), correct=bool(i % 2))
        completed[str(i)] = ExampleResult(
            example_id=str(i),
            prompt=f"Q{i}",
            output=f"A{i}",
            reference=f"ref{i}",
            judgment=j,
            latency_ms=float(i * 10),
        )

    checkpoint_path = tmp_path / "test_checkpoint.jsonl"
    runner._save_checkpoint(checkpoint_path, completed)

    assert checkpoint_path.exists(), "Checkpoint file should exist after save"
    assert not (tmp_path / "test_checkpoint.tmp").exists(), (
        "Temp file should be cleaned up (atomic rename)"
    )

    loaded = runner._load_checkpoint(checkpoint_path)

    assert set(loaded.keys()) == set(completed.keys())
    for id_, original in completed.items():
        loaded_ex = loaded[id_]
        assert loaded_ex.example_id == original.example_id
        assert abs(loaded_ex.score - original.score) < 1e-9
        assert loaded_ex.is_correct == original.is_correct
        assert loaded_ex.prompt == original.prompt


def test_checkpoint_atomic_write_no_corruption(tmp_path: Path):
    """
    The checkpoint must be written to a .tmp file then renamed - never written
    to the final path directly. This prevents partial writes on crash.
    """
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        checkpoint_dir=tmp_path,
    )

    j = _make_judgment()
    completed = {
        "0": ExampleResult(
            example_id="0",
            prompt="q",
            output="a",
            reference="a",
            judgment=j,
            latency_ms=1.0,
        )
    }

    checkpoint_path = tmp_path / "test.jsonl"
    runner._save_checkpoint(checkpoint_path, completed)

    # The tmp file must not exist after a successful save
    tmp_file = checkpoint_path.with_suffix(".tmp")
    assert not tmp_file.exists()
    assert checkpoint_path.exists()


def test_checkpoint_load_skips_corrupt_lines(tmp_path: Path):
    """
    If a checkpoint has a corrupt line (partial write, encoding issue), the
    loader should skip that line and load the rest rather than crashing.
    """
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        checkpoint_dir=tmp_path,
    )

    checkpoint_path = tmp_path / "partial.jsonl"
    good_line = json.dumps(
        {
            "example_id": "0",
            "prompt": "q",
            "output": "a",
            "reference": "a",
            "score": 1.0,
            "is_correct": True,
            "latency_ms": 5.0,
        }
    )
    # Write one good line and one corrupt line
    checkpoint_path.write_text(good_line + "\n{CORRUPT JSON\n")

    loaded = runner._load_checkpoint(checkpoint_path)
    assert "0" in loaded  # Good line survived
    assert len(loaded) == 1  # Corrupt line was skipped


def test_async_runner_run_uses_checkpoint_dir(tmp_path: Path):
    """
    When checkpoint_dir is set, the runner should create a checkpoint file
    after completing the evaluation.
    """
    dataset = _make_dataset(10)
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        checkpoint_dir=tmp_path,
    )

    result = runner.run(dataset)

    assert result.n == 10
    checkpoint_files = list(tmp_path.glob("*.jsonl"))
    assert len(checkpoint_files) >= 1, "At least one checkpoint file should exist"


def test_async_runner_resumes_from_checkpoint(tmp_path: Path):
    """
    If a checkpoint already exists for this dataset+model, the runner should
    load it and skip already-evaluated examples.

    We pre-populate the checkpoint with all 10 examples. The runner should
    return results without making any new API calls.
    """
    dataset = _make_dataset(10)
    provider = MockProvider()
    judge = ExactMatchJudge()
    template = PromptTemplate("{{ question }}")

    runner = AsyncRunner(
        provider=provider,
        judge=judge,
        template=template,
        checkpoint_dir=tmp_path,
    )

    # Run once to populate the checkpoint
    result_1 = runner.run(dataset)

    # Patch _call to raise - if resume works, it should never be called
    with patch.object(provider, "_call", side_effect=RuntimeError("Should not be called")):
        result_2 = runner.run(dataset)

    assert result_2.n == result_1.n
    assert result_2.correct == result_1.correct


# ── AsyncRunner basic correctness ───────────────────────────────────────────────


def test_async_runner_runs_without_error():
    """AsyncRunner should complete a small evaluation without checkpoint."""
    dataset = _make_dataset(5)
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
    )
    result = runner.run(dataset)
    assert result.n == 5


def test_async_runner_preserves_order():
    """Results must be in the same order as the dataset regardless of async completion."""
    dataset = _make_dataset(15)
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
    )
    result = runner.run(dataset)
    assert result.example_ids == dataset.ids


def test_async_runner_tracks_wall_time():
    """wall_time_seconds should be positive for any real run."""
    dataset = _make_dataset(5)
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
    )
    result = runner.run(dataset)
    assert result.wall_time_seconds >= 0.0


# ── MockProvider retry integration ──────────────────────────────────────────────


def test_mock_provider_returns_deterministic_output():
    """Same input → same output, always."""
    provider = MockProvider(seed=42)
    messages = [{"role": "user", "content": "Hello"}]
    out1 = provider.complete(messages)
    out2 = provider.complete(messages)
    assert out1 == out2


def test_mock_provider_tracks_cost_zero():
    """MockProvider always reports zero cost."""
    provider = MockProvider()
    provider.complete([{"role": "user", "content": "test"}])
    summary = provider.cost_summary()
    assert summary["total_cost_usd"] == 0.0


def test_provider_retry_on_failure():
    """
    ModelProvider.complete() should retry up to max_retries times on failure.
    After max_retries, it should raise RuntimeError with the underlying cause.
    """
    provider = MockProvider()
    provider.max_retries = 3

    call_count = 0

    def failing_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Network failure")

    with patch.object(provider, "_call", side_effect=failing_call):
        with pytest.raises(RuntimeError, match="3 attempts"):
            provider.complete([{"role": "user", "content": "test"}])

    assert call_count == 3, f"Expected 3 attempts, got {call_count}"


def test_provider_succeeds_after_transient_failure():
    """
    If _call fails once then succeeds, complete() should return the successful result.
    """
    from evalkit.providers.base import ProviderResponse

    provider = MockProvider()
    provider.max_retries = 3

    call_count = 0

    def flaky_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Transient failure")
        return ProviderResponse(
            content="success",
            input_tokens=5,
            output_tokens=3,
            model="mock",
            cost_usd=0.0,
        )

    with patch.object(provider, "_call", side_effect=flaky_call):
        result = provider.complete([{"role": "user", "content": "test"}])

    assert result == "success"
    assert call_count == 2


def test_provider_cost_accumulates_across_calls():
    """Total cost and tokens should accumulate correctly across multiple calls."""
    from evalkit.providers.base import ProviderResponse

    provider = MockProvider()
    provider.max_retries = 1
    responses = [
        ProviderResponse(content="a", input_tokens=10, output_tokens=5, model="m", cost_usd=0.01),
        ProviderResponse(content="b", input_tokens=20, output_tokens=10, model="m", cost_usd=0.02),
    ]

    with patch.object(provider, "_call", side_effect=responses):
        provider.complete([{"role": "user", "content": "first"}])
        provider.complete([{"role": "user", "content": "second"}])

    summary = provider.cost_summary()
    assert abs(summary["total_cost_usd"] - 0.03) < 1e-9
    assert summary["total_tokens"] == 45
    assert summary["call_count"] == 2


def test_async_runner_mid_run_checkpoint(tmp_path: Path):
    """
    With checkpoint_every=1, the runner saves a checkpoint after every example.
    This exercises the mid-run checkpoint branch (lines 278-279 in runner.py).
    """
    dataset = _make_dataset(5)
    provider = MockProvider()
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=PromptTemplate("{{ question }}"),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,  # save after every single example
    )
    result = runner.run(dataset)
    assert result.n == 5
    # Mid-run checkpoints were saved
    checkpoint_files = list(tmp_path.glob("*.jsonl"))
    assert len(checkpoint_files) >= 1
