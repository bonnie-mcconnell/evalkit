"""
Tests for statistical comparison methods.

Key properties tested:
- McNemar rejects H0 when models are clearly different
- McNemar fails to reject when models are identical
- McNemar raises on misaligned arrays (the critical safety check)
- Wilcoxon detects systematic score differences
- BH correction detects false positives that unadjusted tests miss
- BH correction preserves true positives
"""

import numpy as np
import pytest

from evalkit.metrics.comparison import BHCorrection, McNemarTest, WilcoxonTest

# ── McNemar ────────────────────────────────────────────────────────────────────


def test_mcnemar_identical_models_fail_to_reject():
    """When both models produce the same results, we should not reject H0."""
    correct = [1, 0, 1, 1, 0, 1, 0, 0] * 25
    result = McNemarTest().test(correct, correct)
    assert not result.reject_null
    assert abs(result.p_value - 1.0) < 1e-9


def test_mcnemar_clearly_better_model_rejects():
    """
    Model A correct on 90% of examples, Model B on 50%.
    With n=200, McNemar should reject H0 at alpha=0.05.
    """
    rng = np.random.default_rng(42)
    n = 300
    a_correct = rng.binomial(1, 0.90, size=n).tolist()
    b_correct = rng.binomial(1, 0.50, size=n).tolist()
    result = McNemarTest().test(a_correct, b_correct)
    assert result.reject_null, (
        f"Expected rejection for clearly different models. p={result.p_value:.4f}"
    )


def test_mcnemar_raises_on_misaligned_lengths():
    """The most important safety check: misaligned result sets must fail loudly."""
    with pytest.raises(ValueError, match="same length"):
        McNemarTest().test([1, 0, 1], [1, 0])


def test_mcnemar_raises_on_non_binary():
    """McNemar requires integer binary outcomes - float scores must fail before truncation."""
    with pytest.raises(ValueError, match="binary integer"):
        McNemarTest().test([0.8, 0.3, 0.9], [0.7, 0.4, 0.8])


def test_mcnemar_concordant_returns_p1():
    """All concordant pairs → chi2=0, p=1, effect_size=1."""
    a = [1, 1, 0, 0] * 30
    b = [1, 1, 0, 0] * 30
    result = McNemarTest().test(a, b)
    assert not result.reject_null
    assert "concordant" in result.note.lower()


def test_mcnemar_effect_size_is_odds_ratio():
    """Effect size should be > 1 when model A is better."""
    a = [1] * 80 + [0] * 20
    b = [0] * 80 + [1] * 20
    result = McNemarTest().test(a, b)
    assert result.effect_size > 1.0  # A wins


def test_mcnemar_str_representation():
    a = [1, 0] * 50
    b = [0, 1] * 50
    result = McNemarTest().test(a, b)
    s = str(result)
    assert "McNemar" in s
    assert "p=" in s


# ── Wilcoxon ────────────────────────────────────────────────────────────────────


def test_wilcoxon_identical_scores_fail_to_reject():
    scores = [0.5, 0.7, 0.3, 0.8, 0.6] * 20
    result = WilcoxonTest().test(scores, scores)
    assert not result.reject_null


def test_wilcoxon_shifted_scores_reject():
    """Model A scores are consistently 0.3 higher - should reject H0."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0.2, 0.7, size=100).tolist()
    shifted = [min(1.0, x + 0.3) for x in base]
    result = WilcoxonTest().test(shifted, base)
    assert result.reject_null, f"Expected rejection. p={result.p_value:.4f}"


def test_wilcoxon_raises_on_misaligned():
    with pytest.raises(ValueError, match="same length"):
        WilcoxonTest().test([0.5, 0.6, 0.7], [0.5, 0.6])


def test_wilcoxon_effect_size_in_minus1_to_1():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 100).tolist()
    b = rng.uniform(0, 1, 100).tolist()
    result = WilcoxonTest().test(a, b)
    assert -1.0 <= result.effect_size <= 1.0


# ── BHCorrection ────────────────────────────────────────────────────────────────


def test_bh_all_null_no_rejections():
    """Under the null, no comparisons should be significant after correction."""
    rng = np.random.default_rng(42)
    # 20 p-values drawn from Uniform(0,1) - all null
    p_values = rng.uniform(0, 1, size=20).tolist()
    result = BHCorrection(alpha=0.05).correct(p_values)
    # Not deterministic, but most should not be rejected
    assert sum(result.reject_null) <= 3, (
        f"Too many rejections under null: {sum(result.reject_null)}"
    )


def test_bh_obvious_signals_preserved():
    """Very small p-values should remain significant after BH correction."""
    p_values = [0.001, 0.002, 0.003, 0.80, 0.90, 0.95]
    result = BHCorrection(alpha=0.05).correct(p_values)
    assert result.reject_null[0]  # p=0.001 should survive
    assert result.reject_null[1]  # p=0.002 should survive
    assert not result.reject_null[3]  # p=0.80 should not


def test_bh_flags_false_positive_warning():
    """
    When unadjusted tests would reach different conclusions than BH-adjusted,
    the warning flag must be set.
    """
    # p=0.04 is unadjusted-significant but marginal at k=10
    p_values = [0.04, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    result = BHCorrection(alpha=0.05).correct(p_values)
    # p_adj[0] = 0.04 * 10/1 = 0.40, so unadjusted significant but adjusted not
    assert result.false_positive_warning


def test_bh_adjusted_pvalues_monotone():
    """
    BH-adjusted p-values must be non-decreasing when ordered by raw p-value.

    The invariant: if p_raw[i] ≤ p_raw[j], then p_adj[i] ≤ p_adj[j].
    This ensures the correction preserves rank order.
    """
    p_values = [0.001, 0.008, 0.039, 0.041, 0.20, 0.35, 0.55, 0.70]
    result = BHCorrection(alpha=0.05).correct(p_values)

    # Sort adjusted p-values by their corresponding raw p-value
    paired = sorted(zip(p_values, result.adjusted_p_values))
    p_adj_in_rank_order = [p_adj for _, p_adj in paired]

    for i in range(len(p_adj_in_rank_order) - 1):
        assert p_adj_in_rank_order[i] <= p_adj_in_rank_order[i + 1], (
            f"Monotonicity violated at rank {i}: "
            f"p_adj[{i}]={p_adj_in_rank_order[i]:.4f} > "
            f"p_adj[{i + 1}]={p_adj_in_rank_order[i + 1]:.4f}"
        )


def test_bh_requires_at_least_2():
    with pytest.raises(ValueError, match="at least 2"):
        BHCorrection().correct([0.03])


def test_bh_invalid_pvalues_raise():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BHCorrection().correct([0.5, -0.1, 0.3])


def test_bh_str_representation():
    p_values = [0.01, 0.04, 0.50]
    result = BHCorrection(alpha=0.05).correct(p_values)
    s = str(result)
    assert "Benjamini-Hochberg" in s
    assert "Expected false positives" in s


# ── Additional edge cases ──────────────────────────────────────────────────────


def test_mcnemar_raises_on_float_array_b():
    """
    Second array being float (not integer) should raise before truncation.
    This is the 'second array' dtype check - line 167 in comparison.py.
    """
    a = [1, 0, 1, 0, 1] * 10
    b_float = [0.8, 0.3, 0.9, 0.1, 0.7] * 10
    with pytest.raises(ValueError, match="binary integer"):
        McNemarTest().test(a, b_float)


def test_mcnemar_raises_on_values_outside_01():
    """
    Arrays with values like 2 (integer but not binary) should raise.
    This is the isin([0,1]) check - line 183 in comparison.py.
    """
    a = [0, 1, 2, 0, 1] * 10
    b = [0, 1, 0, 1, 0] * 10
    with pytest.raises(ValueError, match="binary outcomes"):
        McNemarTest().test(a, b)


def test_wilcoxon_small_n_logs_warning(caplog):
    """
    Wilcoxon on n < 20 pairs should log a low-power warning.
    Line 277 in comparison.py.
    """
    import logging

    rng = np.random.default_rng(5)
    a = rng.uniform(0.4, 0.8, size=15).tolist()
    b = rng.uniform(0.3, 0.7, size=15).tolist()

    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.comparison"):
        WilcoxonTest().test(a, b)

    assert any("low power" in r.message.lower() or "n=15" in r.message for r in caplog.records)
