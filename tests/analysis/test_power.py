"""
Tests for PowerAnalysis.

Key invariants:
- Larger effect size → smaller required N
- Lower desired power → smaller required N
- Observed N >= minimum N → achieved_power >= desired_power
"""

import pytest

from evalkit.analysis.power import PowerAnalysis


@pytest.fixture
def pa():
    return PowerAnalysis(alpha=0.05, power=0.80)


def test_proportion_larger_effect_needs_smaller_n(pa):
    n_small_effect = pa.for_proportion_difference(0.02).minimum_n
    n_large_effect = pa.for_proportion_difference(0.10).minimum_n
    assert n_large_effect < n_small_effect


def test_proportion_higher_power_needs_more_n():
    pa_low = PowerAnalysis(alpha=0.05, power=0.70)
    pa_high = PowerAnalysis(alpha=0.05, power=0.90)
    assert (
        pa_high.for_proportion_difference(0.05).minimum_n
        > pa_low.for_proportion_difference(0.05).minimum_n
    )


def test_adequate_n_achieves_desired_power(pa):
    result = pa.for_proportion_difference(0.05, p1=0.70)
    min_n = result.minimum_n
    result_at_min = pa.for_proportion_difference(0.05, p1=0.70, observed_n=min_n)
    assert result_at_min.achieved_power >= 0.75  # Some tolerance for approximation


def test_tiny_n_underpowered(pa):
    result = pa.for_proportion_difference(0.05, p1=0.70, observed_n=20)
    assert result.achieved_power is not None
    assert result.achieved_power < 0.80
    assert not result.is_adequate


def test_ci_precision_scales_correctly(pa):
    n_tight = pa.for_ci_precision(0.02).minimum_n
    n_loose = pa.for_ci_precision(0.10).minimum_n
    assert n_tight > n_loose


def test_ci_precision_typical_values(pa):
    """N for ±5% CI at 95% confidence, accuracy~0.7, should be ~300-320."""
    result = pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=0.70)
    assert 250 <= result.minimum_n <= 400, f"Expected ~320, got {result.minimum_n}"


def test_wilcoxon_medium_effect_requires_reasonable_n(pa):
    """Cohen's d=0.5 (medium) should require ~33-40 pairs at 80% power."""
    result = pa.for_wilcoxon(cohens_d=0.5)
    assert 25 <= result.minimum_n <= 60, f"Got {result.minimum_n}"


def test_mcnemar_result_has_correct_fields(pa):
    result = pa.for_mcnemar(effect_size=2.0)
    assert result.minimum_n > 0
    assert result.test_type == "McNemar"
    assert result.alpha == 0.05
    assert result.desired_power == 0.80


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        PowerAnalysis(alpha=1.5, power=0.80)


def test_invalid_power_raises():
    with pytest.raises(ValueError, match="power"):
        PowerAnalysis(alpha=0.05, power=0.0)


def test_proportion_invalid_p1_raises(pa):
    with pytest.raises(ValueError, match="p1"):
        pa.for_proportion_difference(0.05, p1=1.5)


def test_proportion_invalid_p1_negative_raises(pa):
    with pytest.raises(ValueError, match="p1"):
        pa.for_proportion_difference(0.05, p1=-0.1)


def test_ci_precision_invalid_half_width_raises(pa):
    with pytest.raises(ValueError, match="desired_half_width"):
        pa.for_ci_precision(desired_half_width=1.5)


def test_ci_precision_invalid_accuracy_raises(pa):
    with pytest.raises(ValueError, match="expected_accuracy"):
        pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=1.5)


def test_mcnemar_invalid_effect_size_raises(pa):
    with pytest.raises(ValueError, match="odds ratio"):
        pa.for_mcnemar(effect_size=0)


def test_mcnemar_negative_effect_size_raises(pa):
    with pytest.raises(ValueError, match="odds ratio"):
        pa.for_mcnemar(effect_size=-1.0)


def test_wilcoxon_invalid_cohens_d_raises(pa):
    with pytest.raises(ValueError, match="cohens_d"):
        pa.for_wilcoxon(cohens_d=0)


def test_wilcoxon_negative_cohens_d_raises(pa):
    with pytest.raises(ValueError, match="cohens_d"):
        pa.for_wilcoxon(cohens_d=-0.5)


def test_ci_precision_adequate_n_is_adequate(pa):
    """Providing exactly minimum_n should mark as adequate."""
    result = pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=0.70)
    min_n = result.minimum_n
    result_at_min = pa.for_ci_precision(
        desired_half_width=0.05, expected_accuracy=0.70, observed_n=min_n
    )
    assert result_at_min.achieved_power == 1.0
    assert result_at_min.is_adequate


def test_ci_precision_small_n_is_not_adequate(pa):
    """n=20 cannot achieve ±5% CI - should be flagged as not adequate."""
    result = pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=0.70, observed_n=20)
    assert result.achieved_power == 0.0
    assert not result.is_adequate


def test_ci_precision_no_observed_n_is_adequate_by_default(pa):
    """A planning-only call (no observed_n) should always report is_adequate=True."""
    result = pa.for_ci_precision(desired_half_width=0.05)
    assert result.achieved_power is None
    assert result.is_adequate


def test_power_result_str_contains_key_info(pa):
    result = pa.for_proportion_difference(0.05)
    s = str(result)
    assert "Minimum N" in s
    assert "TwoProportionZ" in s


def test_observed_n_zero_raises(pa):
    """observed_n=0 is meaningless - must raise with clear message."""
    with pytest.raises(ValueError, match="observed_n"):
        pa.for_proportion_difference(0.05, observed_n=0)


def test_observed_n_negative_raises(pa):
    with pytest.raises(ValueError, match="observed_n"):
        pa.for_ci_precision(0.05, observed_n=-10)


def test_observed_n_zero_mcnemar_raises(pa):
    with pytest.raises(ValueError, match="observed_n"):
        pa.for_mcnemar(effect_size=2.0, observed_n=0)


def test_observed_n_zero_wilcoxon_raises(pa):
    with pytest.raises(ValueError, match="observed_n"):
        pa.for_wilcoxon(cohens_d=0.5, observed_n=0)


def test_mcnemar_with_observed_n_returns_achieved_power(pa):
    """for_mcnemar with observed_n should populate achieved_power."""
    result = pa.for_mcnemar(effect_size=2.0, observed_n=100)
    assert result.achieved_power is not None
    assert 0.0 <= result.achieved_power <= 1.0


def test_wilcoxon_with_observed_n_returns_achieved_power(pa):
    """for_wilcoxon with observed_n should populate achieved_power."""
    result = pa.for_wilcoxon(cohens_d=0.5, observed_n=60)
    assert result.achieved_power is not None
    assert 0.0 <= result.achieved_power <= 1.0


def test_sample_size_table_mcnemar():
    """sample_size_table with mcnemar test type should run without error."""
    pa = PowerAnalysis(alpha=0.05)
    output = pa.sample_size_table(test="mcnemar", effect_sizes=[1.5, 2.0])
    assert "McNemar" in output


def test_sample_size_table_wilcoxon():
    """sample_size_table with wilcoxon test type uses Cohen's d labels."""
    pa = PowerAnalysis(alpha=0.05)
    output = pa.sample_size_table(test="wilcoxon", effect_sizes=[0.3, 0.5])
    assert "d =" in output


def test_proportion_invalid_effect_size_raises(pa):
    """effect_size outside (0,1) should raise."""
    with pytest.raises(ValueError, match="effect_size"):
        pa.for_proportion_difference(effect_size=1.5)


def test_mcnemar_invalid_discordant_proportion_raises(pa):
    """discordant_proportion outside (0,1) should raise."""
    with pytest.raises(ValueError, match="discordant_proportion"):
        pa.for_mcnemar(effect_size=2.0, discordant_proportion=0.0)
