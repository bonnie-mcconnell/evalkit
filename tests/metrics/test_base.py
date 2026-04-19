"""
Tests for MetricResult and the bootstrap CI machinery.

These tests check statistical properties, not exact values. The key invariants:
1. CIs contain the true parameter at approximately the nominal coverage rate.
2. Stratified bootstrap produces narrower CIs than non-stratified for imbalanced data.
3. Larger N → narrower CIs (consistency of the estimator).
4. The point estimate lies within the CI (enforced by MetricResult.__post_init__).
"""

import numpy as np
import pytest

from evalkit.metrics.base import Metric, MetricResult


class MockMetric(Metric):
    """Test double: a metric that just computes the mean of predictions."""

    @property
    def name(self) -> str:
        return "MockMean"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        return float(np.mean(predictions))


def test_metric_result_validates_ci_contains_point():
    """Point estimate must lie within the CI."""
    with pytest.raises(ValueError, match="lies outside CI"):
        MetricResult(
            name="Test",
            value=0.90,
            ci_lower=0.50,
            ci_upper=0.80,
            n=100,
        )


def test_metric_result_ci_width():
    r = MetricResult(name="Test", value=0.70, ci_lower=0.65, ci_upper=0.75, n=100)
    assert abs(r.ci_width - 0.10) < 1e-9
    assert abs(r.margin_of_error - 0.05) < 1e-9


def test_metric_result_str_contains_key_info():
    r = MetricResult(name="Accuracy", value=0.73, ci_lower=0.68, ci_upper=0.78, n=200)
    s = str(r)
    assert "0.73" in s
    assert "0.68" in s
    assert "0.78" in s
    assert "n=200" in s


@pytest.mark.slow
def test_bootstrap_ci_contains_true_value_approximately():
    """
    Coverage test: 95% CI should contain the true value ~95% of the time.

    We run 100 simulated experiments. The coverage should be between 88% and 99%
    (generous bounds due to the small number of experiments - this is a
    statistical test of a statistical procedure).

    stratify=False is used here because MockMetric._point_estimate computes a
    mean over predictions. Stratified bootstrap resamples within classes defined
    by the *references* array; when references == predictions (binary data), it
    locks the class balance and collapses CI width to near-zero. That is correct
    behaviour for stratification on a classification metric, but wrong for testing
    the bootstrap machinery itself.
    """
    rng = np.random.default_rng(42)
    true_mean = 0.70
    n_experiments = 100
    n_per_experiment = 200
    covered = 0

    for i in range(n_experiments):
        data = rng.binomial(1, true_mean, size=n_per_experiment)
        metric = MockMetric(n_resamples=1000, seed=i)
        result = metric.compute(data.tolist(), data.tolist(), stratify=False)
        if result.ci_lower <= true_mean <= result.ci_upper:
            covered += 1

    coverage = covered / n_experiments
    assert 0.88 <= coverage <= 0.99, (
        f"CI coverage was {coverage:.2f}, expected ~0.95. The bootstrap CI may be miscalibrated."
    )


def test_ci_narrows_with_larger_n():
    """
    Larger samples should produce narrower CIs.

    Uses stratify=False - see test_bootstrap_ci_contains_true_value_approximately
    for the explanation. Stratified bootstrap on binary data where predictions==references
    collapses CI width to near-zero regardless of n, so it can't be used to test
    the n→width relationship.
    """
    rng = np.random.default_rng(0)
    metric_small = MockMetric(n_resamples=2000, seed=0)
    metric_large = MockMetric(n_resamples=2000, seed=0)

    data_small = rng.binomial(1, 0.7, size=50).tolist()
    data_large = rng.binomial(1, 0.7, size=500).tolist()

    result_small = metric_small.compute(data_small, data_small, stratify=False)
    result_large = metric_large.compute(data_large, data_large, stratify=False)

    assert result_large.ci_width < result_small.ci_width, (
        f"Expected CI to narrow with more data. "
        f"Small n=50: width={result_small.ci_width:.4f}, "
        f"Large n=500: width={result_large.ci_width:.4f}"
    )


def test_empty_input_raises():
    metric = MockMetric()
    with pytest.raises(ValueError, match="empty"):
        metric.compute([], [])


def test_mismatched_length_raises():
    metric = MockMetric()
    with pytest.raises(ValueError, match="same length"):
        metric.compute([1, 2, 3], [1, 2])


def test_zero_resamples_raises():
    """n_resamples=0 must raise immediately with a clear error, not an IndexError later."""
    with pytest.raises(ValueError, match="n_resamples"):
        MockMetric(n_resamples=0)


def test_stratified_vs_unstratified():
    """
    Stratified bootstrap should not crash, and should return valid CIs.
    (Coverage comparison requires a large Monte Carlo study; we just test
    that the mechanism runs without error.)
    """
    np.random.default_rng(5)
    # Imbalanced: 90% class 0, 10% class 1
    data = np.concatenate([np.zeros(90), np.ones(10)]).tolist()

    metric = MockMetric(n_resamples=1000, seed=5)
    result = metric.compute(data, data, stratify=True)

    assert 0.0 <= result.ci_lower <= result.value <= result.ci_upper <= 1.0
    assert result.n == 100


def test_metric_low_n_resamples_logs_warning(caplog):
    """
    n_resamples < 1000 should log a stability warning at construction time.
    Line 104 in base.py.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.base"):
        MockMetric(n_resamples=500)
    assert any("500" in r.message or "low" in r.message.lower() for r in caplog.records)


def test_metric_invalid_ci_level_raises():
    """ci_level outside (0,1) must raise immediately - line 104 in base.py."""
    with pytest.raises(ValueError, match="ci_level"):
        MockMetric(ci_level=1.5)


def test_metric_ci_level_zero_raises():
    with pytest.raises(ValueError, match="ci_level"):
        MockMetric(ci_level=0.0)
