"""
Statistical tests for comparing two or more models on the same evaluation set.

The critical design invariant: all paired tests verify example-ID alignment
before computing. Passing misaligned result sets silently produces nonsense.

McNemar's test: for comparing two models on binary outcomes (same examples).
Wilcoxon signed-rank: for comparing two models on continuous scores.
BHCorrection: for controlling false discovery rate across K≥2 comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Below this p-value, we reject H0 (models perform equally).
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class TestResult:
    """
    Result of a single statistical hypothesis test.

    Attributes
    ----------
    test_name:
        Human-readable test name.
    statistic:
        Test statistic value.
    p_value:
        p-value for the null hypothesis (models are equivalent).
    effect_size:
        A standardised effect size measure (odds ratio for McNemar,
        rank-biserial correlation for Wilcoxon).
    alpha:
        Significance level used for the reject/fail decision.
    reject_null:
        True if p_value < alpha - models are statistically distinguishable.
    n_pairs:
        Number of paired observations.
    note:
        Any warnings or caveats about the test.
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    alpha: float
    reject_null: bool
    n_pairs: int
    note: str = ""

    def __str__(self) -> str:
        decision = "REJECT H₀" if self.reject_null else "fail to reject H₀"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}, "
            f"effect={self.effect_size:.4f}, n={self.n_pairs} → {decision} (α={self.alpha})"
        )


@dataclass(frozen=True)
class MultipleComparisonResult:
    """
    Result of BH-corrected multiple comparisons.

    Attributes
    ----------
    adjusted_p_values:
        BH-adjusted p-values, same order as input.
    reject_null:
        Boolean array indicating which comparisons survive FDR correction.
    unadjusted_p_values:
        Original p-values before correction.
    false_positive_warning:
        True if any comparison's conclusion changes after BH correction.
        When True, the RigorChecker will flag this explicitly.
    expected_false_positives:
        Under H0, how many false discoveries to expect at alpha.
    """

    adjusted_p_values: list[float]
    reject_null: list[bool]
    unadjusted_p_values: list[float]
    comparison_names: list[str]
    alpha: float
    false_positive_warning: bool
    expected_false_positives: float

    def __str__(self) -> str:
        lines = ["Benjamini-Hochberg FDR Correction"]
        lines.append(
            f"  Expected false positives (α={self.alpha}): {self.expected_false_positives:.2f}"
        )  # noqa: E501
        if self.false_positive_warning:
            lines.append(
                "  ⚠ Unadjusted p-values would lead to different conclusions. "
                "Always report adjusted values."
            )
        for name, p_adj, p_raw, reject in zip(
            self.comparison_names,
            self.adjusted_p_values,
            self.unadjusted_p_values,
            self.reject_null,
        ):
            flag = "✓" if reject else "✗"
            lines.append(f"  {flag} {name}: p_raw={p_raw:.4f} → p_adj={p_adj:.4f}")
        return "\n".join(lines)


class McNemarTest:
    """
    McNemar's test for paired binary outcomes.

    Use this to answer: "Is model A statistically better than model B on
    *the same* evaluation examples?"

    The test operates on the 2×2 contingency table of discordant pairs:
    - b: examples where model A is correct, model B is not
    - c: examples where model B is correct, model A is not

    The null hypothesis is b == c (no systematic difference). Only discordant
    pairs carry information; concordant pairs don't matter.

    The chi-squared statistic uses Edwards' continuity correction, which
    matters for small samples. Without it, the test is anti-conservative.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def test(
        self,
        model_a_correct: list[int],
        model_b_correct: list[int],
    ) -> TestResult:
        """
        Parameters
        ----------
        model_a_correct:
            Binary array: 1 if model A was correct, 0 otherwise.
        model_b_correct:
            Binary array: 1 if model B was correct, 0 otherwise.
            Must be aligned 1-to-1 with model_a_correct - same examples,
            same order. Misalignment is the single most common error when
            comparing model results.
        """
        # Validate that inputs are genuinely binary (0 or 1) before casting.
        # np.asarray(..., dtype=int) would silently truncate floats like 0.8 → 0,
        # making the subsequent isin check meaningless.
        a_raw = np.asarray(model_a_correct)
        b_raw = np.asarray(model_b_correct)

        if not (np.issubdtype(a_raw.dtype, np.integer) or np.issubdtype(a_raw.dtype, np.bool_)):
            raise ValueError(
                "McNemarTest requires binary integer outcomes (0 or 1). "
                "For continuous scores, use WilcoxonTest instead."
            )
        if not (np.issubdtype(b_raw.dtype, np.integer) or np.issubdtype(b_raw.dtype, np.bool_)):
            raise ValueError(
                "McNemarTest requires binary integer outcomes (0 or 1). "
                "For continuous scores, use WilcoxonTest instead."
            )

        a = a_raw.astype(int)
        b = b_raw.astype(int)

        if len(a) != len(b):
            raise ValueError(
                f"model_a_correct (n={len(a)}) and model_b_correct (n={len(b)}) "
                "must have the same length. Verify that both result sets contain "
                "the same examples in the same order."
            )

        if not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
            raise ValueError(
                "McNemarTest requires binary outcomes (0 or 1). "
                "For continuous scores, use WilcoxonTest instead."
            )

        n = len(a)
        # Discordant pairs
        b_wins = int(np.sum((b == 1) & (a == 0)))  # B correct, A wrong
        a_wins = int(np.sum((a == 1) & (b == 0)))  # A correct, B wrong
        discordant = a_wins + b_wins

        if discordant == 0:
            return TestResult(
                test_name="McNemar",
                statistic=0.0,
                p_value=1.0,
                effect_size=1.0,
                alpha=self.alpha,
                reject_null=False,
                n_pairs=n,
                note="All pairs are concordant. Models are indistinguishable on this dataset.",
            )

        if discordant < 25:
            logger.warning(
                "Only %d discordant pairs. McNemar's test has low power. "
                "Use exact binomial test for n_discordant < 25.",
                discordant,
            )

        # Chi-squared with continuity correction (Edwards, 1948)
        chi2 = (abs(a_wins - b_wins) - 1) ** 2 / (a_wins + b_wins)
        p_value = float(stats.chi2.sf(chi2, df=1))

        # Odds ratio as effect size: how much more likely is A to be right
        # on a discordant pair?
        odds_ratio = (a_wins + 0.5) / (b_wins + 0.5)  # +0.5 smoothing

        note = ""
        if n < 100:
            note = f"Small sample (n={n}). Effect size estimate is imprecise."

        return TestResult(
            test_name="McNemar",
            statistic=chi2,
            p_value=p_value,
            effect_size=odds_ratio,
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            n_pairs=n,
            note=note,
        )


class WilcoxonTest:
    """
    Wilcoxon signed-rank test for paired continuous scores.

    Use this to answer: "Does model A produce systematically higher quality
    scores than model B on the same examples?"

    The Wilcoxon test makes no normality assumption (unlike a paired t-test),
    which is appropriate for LLM quality scores, which are typically bounded,
    skewed, and not normally distributed.

    Effect size is the rank-biserial correlation r = 1 - (2W / n(n+1)),
    which is in [-1, 1]. Benchmarks: |r| ≥ 0.1 small, 0.3 medium, 0.5 large.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def test(
        self,
        model_a_scores: list[float],
        model_b_scores: list[float],
    ) -> TestResult:
        """
        Parameters
        ----------
        model_a_scores, model_b_scores:
            Continuous quality scores for each example, aligned 1-to-1.
        """
        a = np.asarray(model_a_scores, dtype=float)
        b = np.asarray(model_b_scores, dtype=float)

        if len(a) != len(b):
            raise ValueError(
                f"model_a_scores (n={len(a)}) and model_b_scores (n={len(b)}) "
                "must have the same length."
            )

        n = len(a)
        if n < 20:
            logger.warning(
                "Wilcoxon test on n=%d pairs has low power. "
                "Results should be interpreted with caution.",
                n,
            )

        diffs = a - b
        non_zero = np.abs(diffs) > 1e-10
        if non_zero.sum() == 0:
            return TestResult(
                test_name="Wilcoxon",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                alpha=self.alpha,
                reject_null=False,
                n_pairs=n,
                note="All differences are zero. Models produce identical scores.",
            )

        stat, p_value = stats.wilcoxon(diffs[non_zero], alternative="two-sided")

        # Rank-biserial correlation as effect size
        n_nonzero = non_zero.sum()
        r = 1 - (2 * stat) / (n_nonzero * (n_nonzero + 1))

        return TestResult(
            test_name="Wilcoxon",
            statistic=float(stat),
            p_value=float(p_value),
            effect_size=float(r),
            alpha=self.alpha,
            reject_null=p_value < self.alpha,
            n_pairs=n,
        )


class BHCorrection:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) correction.

    When comparing K≥2 prompt variants or models, the probability of at least
    one false positive grows with K. At K=20 comparisons and α=0.05, you'd
    expect 1 false positive by chance even with identical models.

    BH controls the *expected* proportion of false discoveries among
    rejections - less conservative than Bonferroni, which controls the
    family-wise error rate and has very low power for large K.

    This class flags cases where the uncorrected analysis would reach
    different conclusions than the corrected analysis - which the
    RigorChecker surfaces as a warning.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def correct(
        self,
        p_values: list[float],
        comparison_names: list[str] | None = None,
    ) -> MultipleComparisonResult:
        """
        Apply BH correction to a list of p-values.

        Parameters
        ----------
        p_values:
            Raw p-values from individual tests, in any order.
        comparison_names:
            Optional labels for each comparison. Defaults to ["C1", "C2", ...].
        """
        k = len(p_values)
        if k < 2:
            raise ValueError(
                "BH correction requires at least 2 comparisons. "
                "For a single comparison, use the raw p-value."
            )

        names = comparison_names or [f"C{i + 1}" for i in range(k)]
        p_arr = np.asarray(p_values, dtype=float)

        if np.any(p_arr < 0) or np.any(p_arr > 1):
            raise ValueError("p-values must be in [0, 1].")

        # BH procedure: rank p-values, compute critical values, find threshold.
        rank_order = np.argsort(p_arr)
        ranks = np.empty(k, dtype=int)
        ranks[rank_order] = np.arange(1, k + 1)

        # Adjusted p-value for rank i: p_i * k / i, capped at 1,
        # then enforce monotonicity from the right.
        p_adjusted = np.minimum(1.0, p_arr * k / ranks)

        # Enforce monotonicity: p_adj[i] <= p_adj[i+1] in sorted order.
        sorted_p_adj = p_adjusted[rank_order]
        for i in range(len(sorted_p_adj) - 2, -1, -1):
            sorted_p_adj[i] = min(sorted_p_adj[i], sorted_p_adj[i + 1])
        p_adjusted[rank_order] = sorted_p_adj

        reject = p_adjusted < self.alpha
        reject_unadjusted = p_arr < self.alpha

        # The interesting case: different conclusions before vs after correction.
        false_positive_warning = bool(np.any(reject_unadjusted & ~reject))

        expected_fp = k * self.alpha

        if false_positive_warning:
            n_changed = int(np.sum(reject_unadjusted & ~reject))
            logger.warning(
                "%d comparison(s) appear significant without FDR correction but not after. "
                "These are likely false positives. Always report BH-adjusted p-values.",
                n_changed,
            )

        if k >= 10:
            logger.info(
                "Comparing %d variants. Expected false positives under H0: %.1f. "
                "BH correction applied.",
                k,
                expected_fp,
            )

        return MultipleComparisonResult(
            adjusted_p_values=p_adjusted.tolist(),
            reject_null=reject.tolist(),
            unadjusted_p_values=p_values,
            comparison_names=names,
            alpha=self.alpha,
            false_positive_warning=false_positive_warning,
            expected_false_positives=expected_fp,
        )
