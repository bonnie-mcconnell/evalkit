"""
Tests for AsyncRunner, MockRunner, and RunResult.

MockRunner is tested at the runner level (has reference access) and uses
ExactMatchJudge, which is the correct integration: mock output = reference
string on correct examples, so the judge correctly scores them as 1.0.
"""

import pytest

from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.judge import ExactMatchJudge
from evalkit.core.runner import MockRunner, RunResult


@pytest.fixture
def dataset():
    records = [
        {"id": str(i), "question": f"What is {i}+{i}?", "label": str(i * 2)} for i in range(30)
    ]
    return EvalDataset.from_list(records, name="arithmetic")


@pytest.fixture
def template():
    return PromptTemplate("{{ question }}")


# ── MockRunner correctness ─────────────────────────────────────────────────────


def test_mock_runner_returns_run_result(dataset, template):
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)
    result = runner.run(dataset)
    assert isinstance(result, RunResult)
    assert result.n == 30


def test_mock_runner_accuracy_approximately_correct(dataset, template):
    """
    With accuracy=0.80, roughly 80% of examples should be correct.
    We use a tolerance of ±15% to keep this test robust across seeds.
    """
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80, seed=42)
    result = runner.run(dataset)
    observed = sum(result.correct) / result.n
    assert 0.60 <= observed <= 1.00, (
        f"Expected ~0.80 accuracy, got {observed:.2f}. "
        "Tolerance is generous - if this fails, the mock is broken."
    )


def test_mock_runner_deterministic(dataset, template):
    """Same seed + same dataset must always produce identical results."""
    r1 = MockRunner(judge=ExactMatchJudge(), template=template, seed=7).run(dataset)
    r2 = MockRunner(judge=ExactMatchJudge(), template=template, seed=7).run(dataset)
    assert r1.correct == r2.correct


def test_mock_runner_different_seeds_differ(dataset, template):
    """Different seeds should produce different (but both valid) results."""
    r1 = MockRunner(judge=ExactMatchJudge(), template=template, seed=1).run(dataset)
    r2 = MockRunner(judge=ExactMatchJudge(), template=template, seed=999).run(dataset)
    # They might occasionally agree, but on n=30 this is astronomically unlikely
    assert r1.correct != r2.correct


def test_mock_runner_preserves_dataset_order(dataset, template):
    """Results must be in the same order as the dataset."""
    runner = MockRunner(judge=ExactMatchJudge(), template=template)
    result = runner.run(dataset)
    assert result.example_ids == dataset.ids


def test_mock_runner_zero_accuracy(dataset, template):
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.0)
    result = runner.run(dataset)
    assert sum(result.correct) == 0


def test_mock_runner_full_accuracy(dataset, template):
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=1.0)
    result = runner.run(dataset)
    assert sum(result.correct) == result.n


def test_mock_runner_invalid_accuracy(template):
    with pytest.raises(ValueError, match="accuracy"):
        MockRunner(judge=ExactMatchJudge(), template=template, accuracy=1.5)


# ── RunResult properties ───────────────────────────────────────────────────────


def test_run_result_correct_is_binary(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template).run(dataset)
    assert all(c in [0, 1] for c in result.correct)


def test_run_result_scores_in_unit_interval(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template).run(dataset)
    assert all(0.0 <= s <= 1.0 for s in result.scores)


def test_run_result_n_matches_dataset(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template).run(dataset)
    assert result.n == len(dataset)


def test_run_result_summary_has_required_keys(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template).run(dataset)
    summary = result.summary()
    for key in ("model", "dataset", "n", "raw_accuracy", "total_cost_usd"):
        assert key in summary


def test_run_result_cost_zero_for_mock(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template).run(dataset)
    assert result.total_cost_usd == 0.0
    assert result.cost_per_correct() is None  # cost is 0, so cost_per_correct is undefined


def test_run_result_raw_accuracy_consistent(dataset, template):
    result = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=1.0).run(dataset)
    assert result.summary()["raw_accuracy"] == pytest.approx(1.0)


def test_async_runner_checkpoint_resilient_to_blank_lines(tmp_path, dataset, template):
    """
    A checkpoint file with blank lines or a trailing newline should not crash
    the resume logic. This guards against partially-written checkpoint files.
    """
    from evalkit.core.runner import AsyncRunner
    from evalkit.providers.base import MockProvider

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    provider = MockProvider(seed=42)
    runner = AsyncRunner(
        provider=provider,
        judge=ExactMatchJudge(),
        template=template,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=5,
    )

    # Write a checkpoint file with blank lines (simulating partial write)
    cp_file = checkpoint_dir / "arithmetic_mock-provider-v1_checkpoint.jsonl"
    cp_file.write_text(
        '{"example_id": "0", "prompt": "q", "output": "0", "reference": "0", '
        '"score": 1.0, "is_correct": true, "latency_ms": 0.0}\n'
        "\n"  # blank line - must not crash
        '{"example_id": "1", "prompt": "q", "output": "2", "reference": "2", '
        '"score": 1.0, "is_correct": true, "latency_ms": 0.0}\n'
    )

    # Loading should not raise and should recover the 2 valid records
    completed = runner._load_checkpoint(cp_file)
    assert len(completed) == 2
    assert "0" in completed
    assert "1" in completed


def test_run_result_to_dataframe_raises_without_pandas(monkeypatch, dataset, template):
    """
    RunResult.to_dataframe() raises ImportError when pandas is absent
    (lines 159-160 in runner.py).
    """
    import sys

    monkeypatch.setitem(sys.modules, "pandas", None)  # type: ignore[arg-type]

    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)
    run_result = runner.run(dataset)

    with pytest.raises((ImportError, TypeError)):
        run_result.to_dataframe()
