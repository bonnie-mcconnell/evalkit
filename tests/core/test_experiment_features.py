"""
Tests for new ExperimentResult features:
  - compare() - paired significance test between two results
  - worst_examples() - error analysis on misclassified examples
  - to_dataframe() - per-example DataFrame export

And new EvalDataset features:
  - split() - stratified train/test split
  - sample() - random sampling

And PromptTemplate.validate() - pre-run field checking.
"""

from __future__ import annotations

import pytest

from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.experiment import ComparisonResult, Experiment, ExperimentResult
from evalkit.core.judge import ExactMatchJudge
from evalkit.core.runner import MockRunner

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_dataset(n: int = 100, n_classes: int = 2, name: str = "test") -> EvalDataset:
    """Create a balanced dataset with n examples across n_classes."""
    records = [{"id": str(i), "question": f"Q{i}", "label": str(i % n_classes)} for i in range(n)]
    return EvalDataset.from_list(records, name=name)


def _run_experiment(
    dataset: EvalDataset,
    accuracy: float = 0.80,
    seed: int = 42,
    name: str = "exp",
) -> ExperimentResult:
    judge = ExactMatchJudge()
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=judge, template=template, accuracy=accuracy, seed=seed)
    return Experiment(name, dataset, runner, n_resamples=500).run()


# ── EvalDataset.split() ────────────────────────────────────────────────────────


def test_split_correct_sizes():
    """split(0.2) on 100 examples should give ~80 train, ~20 test."""
    dataset = _make_dataset(100)
    train, test = dataset.split(test_size=0.2)
    assert len(train) + len(test) == 100
    assert 15 <= len(test) <= 25  # generous bounds for stratified rounding


def test_split_no_overlap():
    """No example should appear in both train and test."""
    dataset = _make_dataset(100)
    train, test = dataset.split(test_size=0.3)
    train_ids = {e.id for e in train}
    test_ids = {e.id for e in test}
    assert train_ids.isdisjoint(test_ids)


def test_split_covers_all_examples():
    """Every example should appear in exactly one split."""
    dataset = _make_dataset(100)
    train, test = dataset.split(test_size=0.2)
    all_ids = {e.id for e in train} | {e.id for e in test}
    assert all_ids == {e.id for e in dataset}


def test_split_stratified_preserves_distribution():
    """
    Stratified split should preserve class distribution in both subsets.
    With 50% class-0 and 50% class-1, each split should be close to 50/50.
    """
    dataset = _make_dataset(200, n_classes=2)
    train, test = dataset.split(test_size=0.2, stratify=True)

    def class_frac(ds: EvalDataset, label: str) -> float:
        return sum(1 for e in ds if e.reference == label) / len(ds)

    # Both splits should be within 10% of 50/50
    for split_ds in (train, test):
        assert abs(class_frac(split_ds, "0") - 0.5) < 0.10


def test_split_deterministic_with_same_seed():
    """Same seed always produces the same split."""
    dataset = _make_dataset(100)
    _, test_a = dataset.split(test_size=0.2, seed=7)
    _, test_b = dataset.split(test_size=0.2, seed=7)
    assert [e.id for e in test_a] == [e.id for e in test_b]


def test_split_different_seeds_differ():
    """Different seeds should (almost certainly) produce different splits."""
    dataset = _make_dataset(100)
    _, test_a = dataset.split(test_size=0.2, seed=1)
    _, test_b = dataset.split(test_size=0.2, seed=99)
    assert [e.id for e in test_a] != [e.id for e in test_b]


def test_split_invalid_test_size_raises():
    dataset = _make_dataset(50)
    with pytest.raises(ValueError, match="test_size"):
        dataset.split(test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        dataset.split(test_size=1.0)


def test_split_preserves_dataset_names():
    dataset = _make_dataset(50, name="my_data")
    train, test = dataset.split(test_size=0.2)
    assert "train" in train.name
    assert "test" in test.name


# ── EvalDataset.sample() ───────────────────────────────────────────────────────


def test_sample_correct_size():
    dataset = _make_dataset(100)
    sampled = dataset.sample(30)
    assert len(sampled) == 30


def test_sample_no_duplicates():
    """Sample without replacement - no example should appear twice."""
    dataset = _make_dataset(100)
    sampled = dataset.sample(50)
    ids = [e.id for e in sampled]
    assert len(ids) == len(set(ids))


def test_sample_larger_than_dataset_returns_original():
    """Requesting more than available should return the full dataset."""
    dataset = _make_dataset(20)
    sampled = dataset.sample(100)
    assert len(sampled) == 20


def test_sample_deterministic():
    dataset = _make_dataset(100)
    a = dataset.sample(20, seed=5)
    b = dataset.sample(20, seed=5)
    assert [e.id for e in a] == [e.id for e in b]


# ── PromptTemplate.validate() ─────────────────────────────────────────────────


def test_validate_compatible_template_returns_empty():
    """No errors when all template variables exist in the dataset."""
    dataset = _make_dataset(20)
    template = PromptTemplate("Q: {{ question }}")
    errors = template.validate(dataset)
    assert errors == []


def test_validate_missing_field_returns_errors():
    """Missing template field should produce at least one error."""
    dataset = _make_dataset(20)
    template = PromptTemplate("{{ question }} - context: {{ context }}")
    errors = template.validate(dataset)
    assert len(errors) > 0
    assert any("context" in e for e in errors)


def test_validate_caps_errors_at_five():
    """
    validate() should stop reporting after 5 errors to avoid flooding output.
    A dataset of 100 examples all missing a field should not produce 100 errors.
    """
    dataset = _make_dataset(100)
    template = PromptTemplate("{{ nonexistent }}")
    errors = template.validate(dataset)
    # At most 6 entries: 5 errors + 1 "and potentially N more" message
    assert len(errors) <= 6


def test_validate_wrong_field_name_is_informative():
    """Error message should mention both the wrong variable and available fields."""
    dataset = _make_dataset(5)
    template = PromptTemplate("{{ questoin }}")  # typo: questoin vs question
    errors = template.validate(dataset)
    assert len(errors) > 0
    # The error should tell the user what field failed
    assert "questoin" in errors[0] or "questoin" in str(errors)


# ── ExperimentResult.compare() ────────────────────────────────────────────────


def test_compare_returns_comparison_result():
    dataset = _make_dataset(80)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="model_a")
    result_b = _run_experiment(dataset, accuracy=0.60, seed=2, name="model_b")
    comp = result_a.compare(result_b)
    assert isinstance(comp, ComparisonResult)


def test_compare_clearly_better_model_rejects_null():
    """
    85% vs 60% accuracy on 80 examples - McNemar should reject H₀.
    """
    dataset = _make_dataset(80)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="a")
    result_b = _run_experiment(dataset, accuracy=0.60, seed=2, name="b")
    comp = result_a.compare(result_b)
    assert comp.reject_null, f"Expected rejection. p={comp.p_value:.4f}"


def test_compare_identical_models_fails_to_reject():
    """Same runner (same seed, same accuracy) - should not reject H₀."""
    dataset = _make_dataset(100)
    result_a = _run_experiment(dataset, accuracy=0.75, seed=42, name="a")
    result_b = _run_experiment(dataset, accuracy=0.75, seed=42, name="b")
    comp = result_a.compare(result_b)
    assert not comp.reject_null


def test_compare_raises_on_misaligned_datasets():
    """
    Comparing experiments run on different datasets must raise ValueError.
    Misaligned paired tests are the most common error in model evaluation.
    We use datasets with explicitly different IDs to guarantee misalignment.
    """
    records_a = [{"id": f"a_{i}", "question": f"Q{i}", "label": "0"} for i in range(40)]
    records_b = [{"id": f"b_{i}", "question": f"Q{i}", "label": "0"} for i in range(40)]
    dataset_a = EvalDataset.from_list(records_a, name="data_a")
    dataset_b = EvalDataset.from_list(records_b, name="data_b")
    result_a = _run_experiment(dataset_a, name="a")
    result_b = _run_experiment(dataset_b, name="b")
    with pytest.raises(ValueError, match="different example sets"):
        result_a.compare(result_b)


def test_compare_str_output_is_informative():
    """str(ComparisonResult) should include both model names and the decision."""
    dataset = _make_dataset(100)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="gpt4o")
    result_b = _run_experiment(dataset, accuracy=0.65, seed=2, name="gpt4o_mini")
    comp = result_a.compare(result_b)
    text = str(comp)
    assert "gpt4o" in text
    assert "gpt4o_mini" in text
    assert "p=" in text


def test_compare_requires_n_if_not_significant():
    """
    When H₀ is not rejected, str() should include a suggested minimum N.
    This is the actionable guidance users need.
    """
    dataset = _make_dataset(30)  # Small N → likely not significant for small delta
    result_a = _run_experiment(dataset, accuracy=0.72, seed=1, name="a")
    result_b = _run_experiment(dataset, accuracy=0.68, seed=2, name="b")
    comp = result_a.compare(result_b)
    if not comp.reject_null:
        text = str(comp)
        # Should mention increasing N - format is "Increase N to ≥XXXX"
        assert "Increase N" in text or "increase n" in text.lower() or "≥" in text


def test_compare_symmetry_of_accuracies():
    """ComparisonResult should store both accuracies correctly."""
    dataset = _make_dataset(80)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="a")
    result_b = _run_experiment(dataset, accuracy=0.65, seed=2, name="b")
    comp = result_a.compare(result_b)
    assert abs(comp.accuracy_a - result_a.metrics["Accuracy"].value) < 1e-9
    assert abs(comp.accuracy_b - result_b.metrics["Accuracy"].value) < 1e-9


# ── ExperimentResult.worst_examples() ─────────────────────────────────────────


def test_worst_examples_returns_only_wrong():
    """worst_examples() should only contain incorrectly answered examples."""
    dataset = _make_dataset(100)
    result = _run_experiment(dataset, accuracy=0.80)
    worst = result.worst_examples(20)
    for ex in worst:
        # All returned examples must have been judged incorrect
        assert not any(
            r.example_id == ex["example_id"] and r.is_correct
            for r in result.run_result.example_results
        )


def test_worst_examples_respects_n_limit():
    """worst_examples(10) should return at most 10 examples."""
    dataset = _make_dataset(100)
    result = _run_experiment(dataset, accuracy=0.50)  # ~50% wrong
    worst = result.worst_examples(10)
    assert len(worst) <= 10


def test_worst_examples_fewer_than_n_when_few_errors():
    """If fewer than n examples are wrong, all wrong examples are returned."""
    dataset = _make_dataset(100)
    result = _run_experiment(dataset, accuracy=0.98, seed=999)  # very few wrong
    n_wrong = sum(1 for r in result.run_result.example_results if not r.is_correct)
    worst = result.worst_examples(50)
    assert len(worst) == n_wrong


def test_worst_examples_contains_expected_keys():
    """Each worst example dict must have the keys callers need for debugging."""
    dataset = _make_dataset(100)
    result = _run_experiment(dataset, accuracy=0.70)
    worst = result.worst_examples(5)
    required_keys = {"example_id", "prompt", "output", "reference", "score", "reasoning"}
    for ex in worst:
        assert required_keys.issubset(ex.keys()), f"Missing keys: {required_keys - ex.keys()}"


def test_worst_examples_empty_when_perfect():
    """
    A near-perfect model should have very few or no worst examples.
    We use accuracy=0.99 rather than 1.0 because accuracy=1.0 causes
    the power analysis formula to degenerate (variance = p*(1-p) = 0).
    """
    dataset = _make_dataset(50)
    result = _run_experiment(dataset, accuracy=0.99, seed=0)
    # With 99% accuracy on 50 examples, expect 0 or 1 wrong
    worst = result.worst_examples()
    n_wrong = sum(1 for r in result.run_result.example_results if not r.is_correct)
    assert len(worst) == n_wrong


# ── ExperimentResult.to_dataframe() ───────────────────────────────────────────


def test_to_dataframe_returns_dataframe():
    """to_dataframe() should return a pandas DataFrame when pandas is installed."""
    pd = pytest.importorskip("pandas")
    dataset = _make_dataset(50)
    result = _run_experiment(dataset)
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_to_dataframe_correct_shape():
    """DataFrame should have one row per example and the expected columns."""
    pytest.importorskip("pandas")
    dataset = _make_dataset(50)
    result = _run_experiment(dataset)
    df = result.to_dataframe()
    assert len(df) == 50
    expected_cols = {
        "example_id",
        "prompt",
        "output",
        "reference",
        "is_correct",
        "score",
        "reasoning",
        "latency_ms",
    }
    assert expected_cols.issubset(set(df.columns))


def test_to_dataframe_is_correct_column_matches_run_result():
    """is_correct column must match the run_result.correct binary array."""
    pytest.importorskip("pandas")
    dataset = _make_dataset(50)
    result = _run_experiment(dataset)
    df = result.to_dataframe()
    df_correct = df["is_correct"].astype(int).tolist()
    run_correct = result.run_result.correct
    assert df_correct == run_correct


def test_to_dataframe_filterable():
    """Common use: filter to wrong answers for error analysis."""
    pytest.importorskip("pandas")
    dataset = _make_dataset(100)
    result = _run_experiment(dataset, accuracy=0.70)
    df = result.to_dataframe()
    wrong = df[~df["is_correct"]]
    assert len(wrong) > 0
    assert (wrong["is_correct"] == False).all()  # noqa: E712


# ── PowerAnalysis.sample_size_table() ─────────────────────────────────────────


def test_sample_size_table_returns_string():
    """sample_size_table() should return a non-empty string."""
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=0.05)
    result = pa.sample_size_table()
    assert isinstance(result, str)
    assert len(result) > 100


def test_sample_size_table_contains_effect_sizes():
    """Table should contain each effect size label."""
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=0.05)
    result = pa.sample_size_table(effect_sizes=[0.05, 0.10])
    assert "5%" in result
    assert "10%" in result


def test_sample_size_table_numbers_decrease_with_larger_effect():
    """
    Required N should decrease as the effect size increases - larger effects
    are easier to detect.
    """
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=0.05, power=0.80)
    small = pa.for_proportion_difference(0.05).minimum_n
    large = pa.for_proportion_difference(0.15).minimum_n
    assert small > large, "Smaller effects need more examples"


def test_sample_size_table_invalid_test_raises():
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=0.05)
    with pytest.raises(ValueError, match="Unknown test"):
        pa.sample_size_table(test="bogus")


# ── ComparisonResult.winner property ──────────────────────────────────────────


def test_comparison_result_winner_mcnemar_a_better():
    """McNemar: odds ratio > 1 means experiment_a wins."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="McNemar",
        statistic=5.0,
        p_value=0.025,
        effect_size=2.5,  # OR > 1 → a wins
        reject_null=True,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.80,
        accuracy_b=0.65,
    )
    assert comp.winner == "a"


def test_comparison_result_winner_mcnemar_b_better():
    """McNemar: odds ratio < 1 means experiment_b wins."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="McNemar",
        statistic=5.0,
        p_value=0.025,
        effect_size=0.4,  # OR < 1 → b wins
        reject_null=True,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.65,
        accuracy_b=0.80,
    )
    assert comp.winner == "b"


def test_comparison_result_winner_wilcoxon_a_better():
    """Wilcoxon: positive rank-biserial correlation means experiment_a wins."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="Wilcoxon",
        statistic=1200.0,
        p_value=0.01,
        effect_size=0.35,  # r > 0 → a wins
        reject_null=True,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.80,
        accuracy_b=0.70,
    )
    assert comp.winner == "a"


def test_comparison_result_winner_wilcoxon_b_better():
    """Wilcoxon: negative rank-biserial correlation means experiment_b wins."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="Wilcoxon",
        statistic=800.0,
        p_value=0.03,
        effect_size=-0.28,  # r < 0 → b wins
        reject_null=True,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.70,
        accuracy_b=0.80,
    )
    assert comp.winner == "b"


def test_comparison_result_is_frozen():
    """ComparisonResult must be immutable - result objects should not be mutated."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="McNemar",
        statistic=5.0,
        p_value=0.025,
        effect_size=2.5,
        reject_null=True,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.80,
        accuracy_b=0.65,
    )
    with pytest.raises((AttributeError, TypeError)):
        comp.p_value = 0.99  # type: ignore[misc]


def test_approx_required_n_when_identical_accuracy():
    """When both models have identical accuracy, required N is capped at 99,999."""
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="McNemar",
        statistic=0.0,
        p_value=1.0,
        effect_size=1.0,
        reject_null=False,
        alpha=0.05,
        n_pairs=100,
        note="",
        accuracy_a=0.75,
        accuracy_b=0.75,
    )
    assert comp._approx_required_n() == 99_999


# ── compare() Wilcoxon path ────────────────────────────────────────────────────


def test_compare_explicit_wilcoxon_test():
    """compare(test='wilcoxon') should use Wilcoxon signed-rank, not McNemar."""
    dataset = _make_dataset(80)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="a")
    result_b = _run_experiment(dataset, accuracy=0.60, seed=2, name="b")
    comp = result_a.compare(result_b, test="wilcoxon")
    assert comp.test_name == "Wilcoxon"
    assert isinstance(comp.statistic, float)
    assert 0.0 <= comp.p_value <= 1.0


def test_compare_auto_uses_mcnemar_for_binary_scores():
    """
    compare(test='auto') should choose McNemar when all scores are 0.0 or 1.0
    (i.e. ExactMatchJudge, which produces binary scores).
    """
    dataset = _make_dataset(80)
    result_a = _run_experiment(dataset, accuracy=0.85, seed=1, name="a")
    result_b = _run_experiment(dataset, accuracy=0.65, seed=2, name="b")
    comp = result_a.compare(result_b, test="auto")
    assert comp.test_name == "McNemar"


# ── additional_metric failure path ────────────────────────────────────────────


def test_experiment_additional_metric_failure_is_logged(caplog):
    """
    When an additional metric raises, the error should be logged as a warning
    and the experiment should still complete successfully with Accuracy.
    """
    import logging

    import numpy as np

    from evalkit.metrics.base import Metric

    class BrokenMetric(Metric):
        @property
        def name(self) -> str:
            return "BrokenMetric"

        def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
            raise RuntimeError("This metric always fails")

    dataset = _make_dataset(60)
    result = _run_experiment(
        dataset,
        accuracy=0.75,
        name="test_broken_metric",
    )

    from evalkit.core.dataset import PromptTemplate
    from evalkit.core.experiment import Experiment
    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    judge = ExactMatchJudge()
    tmpl = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=judge, template=tmpl, accuracy=0.75, seed=42)

    with caplog.at_level(logging.WARNING, logger="evalkit.core.experiment"):
        result = Experiment(
            "broken_metric_test",
            dataset,
            runner,
            additional_metrics=[BrokenMetric()],
            n_resamples=300,
        ).run()

    # Experiment completes - Accuracy is present
    assert "Accuracy" in result.metrics
    # BrokenMetric is absent
    assert "BrokenMetric" not in result.metrics
    # Warning was logged
    assert any("BrokenMetric" in r.message for r in caplog.records)


def test_approx_required_n_with_non_significant_result():
    """
    _approx_required_n computes a sensible N when two models have different
    accuracy. This exercises lines 137-144 in experiment.py.
    The function is called when reject_null=False in ComparisonResult.__str__.
    """
    from evalkit.core.experiment import ComparisonResult

    comp = ComparisonResult(
        experiment_a="a",
        experiment_b="b",
        test_name="McNemar",
        statistic=0.5,
        p_value=0.48,
        effect_size=1.1,
        reject_null=False,
        alpha=0.05,
        n_pairs=50,
        note="",
        accuracy_a=0.72,
        accuracy_b=0.68,
    )
    # Should not return 99_999 (delta=0.04 > 0.001) - should compute a real N
    n = comp._approx_required_n()
    assert 100 < n < 50_000  # plausible range for 4pp difference at 80% power
    # And str() should include it
    text = str(comp)
    assert "Increase N" in text


def test_experiment_result_to_dataframe_raises_without_pandas(monkeypatch):
    """
    ExperimentResult.to_dataframe() raises ImportError with install hint
    when pandas is not available (lines 316-317 in experiment.py).
    """
    import sys

    monkeypatch.setitem(sys.modules, "pandas", None)  # type: ignore[arg-type]

    dataset = _make_dataset(30)
    result = _run_experiment(dataset, accuracy=0.80, name="no_pandas")

    with pytest.raises((ImportError, TypeError)):
        result.to_dataframe()
