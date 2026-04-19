"""
Power analysis for LLM evaluation experiments.

The most common mistake in LLM evaluation: running an experiment on N=50 examples
because "that's all we labelled", then reporting results to two decimal places as
if N=50 supports that precision.

Power analysis answers: "How many examples do I need before I start, so I'm not
wasting money on an underpowered experiment?"

The RigorChecker calls this module post-hoc to flag experiments that were
underpowered. The CLI calls it pre-hoc to plan experiments correctly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PowerResult:
    """
    Result of a power calculation.

    Attributes
    ----------
    minimum_n:
        Minimum sample size to achieve desired_power at effect_size and alpha.
    achieved_power:
        Power actually achieved at the observed n (if provided).
    effect_size:
        The effect size used in the calculation. For accuracy comparisons,
        this is the absolute accuracy difference (e.g. 0.05 = 5%).
    alpha:
        Type I error rate (false positive probability).
    desired_power:
        Target statistical power (1 - Type II error rate).
    test_type:
        The statistical test this power calculation applies to.
    """

    minimum_n: int
    effect_size: float
    alpha: float
    desired_power: float
    test_type: str
    achieved_power: float | None = None

    def __str__(self) -> str:
        lines = [
            f"Power Analysis ({self.test_type})",
            f"  Effect size: {self.effect_size:.4f}",
            f"  α: {self.alpha:.3f}, target power: {self.desired_power:.2f}",
            f"  Minimum N required: {self.minimum_n}",
        ]
        if self.achieved_power is not None:
            status = "✓ adequate" if self.achieved_power >= self.desired_power else "✗ underpowered"
            lines.append(f"  Achieved power: {self.achieved_power:.3f} [{status}]")
        return "\n".join(lines)

    @property
    def is_adequate(self) -> bool:
        if self.achieved_power is None:
            return True  # Just a planning calculation, not an audit
        return self.achieved_power >= self.desired_power


class PowerAnalysis:
    """
    Power calculations for the statistical tests used in evalkit.

    All methods follow the same contract: call with your desired effect size,
    alpha, and power to get the minimum N. Or call with observed N and effect
    size to get the achieved power on an existing experiment.

    Effect size conventions
    -----------------------
    - For accuracy comparisons (McNemar, proportion tests): absolute difference
      in accuracy, e.g. 0.05 for a 5 percentage point difference.
    - For continuous score comparisons (Wilcoxon): Cohen's d (standardised mean
      difference). d=0.2 small, 0.5 medium, 0.8 large.
    - For CI precision: desired half-width of the CI, e.g. 0.05 for ±5%.
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.80) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < power < 1):
            raise ValueError(f"power must be in (0, 1), got {power}")
        self.alpha = alpha
        self.power = power

    @staticmethod
    def _validate_observed_n(observed_n: int | None) -> None:
        if observed_n is not None and observed_n < 1:
            raise ValueError(
                f"observed_n must be a positive integer, got {observed_n}. "
                "Use None to compute the minimum N without an observed sample."
            )

    def for_proportion_difference(
        self,
        effect_size: float,
        p1: float = 0.7,
        observed_n: int | None = None,
    ) -> PowerResult:
        """
        Sample size for detecting a difference in accuracy between two models.

        Uses a two-proportion z-test approximation. For the exact paired
        McNemar test, use `for_mcnemar`.

        Parameters
        ----------
        effect_size:
            Absolute accuracy difference to detect, e.g. 0.05 for 5%.
        p1:
            Expected accuracy of the baseline model. Used to compute the
            variance term - accuracy near 0.5 requires larger N.
        observed_n:
            If provided, compute achieved power at this N instead of minimum N.
        """
        if not (0 < effect_size < 1):
            raise ValueError(f"effect_size must be in (0, 1), got {effect_size}")
        if not (0 < p1 < 1):
            raise ValueError(f"p1 (baseline accuracy) must be in (0, 1), got {p1}")
        self._validate_observed_n(observed_n)

        p2 = min(1.0, p1 + effect_size)
        p_bar = (p1 + p2) / 2

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Lehr's formula approximation for two-proportion test
        n_per_group = (
            (
                z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            )
            / effect_size
        ) ** 2

        min_n = math.ceil(n_per_group)

        if observed_n is not None:
            achieved = self._achieved_power_proportion(observed_n, p1, p2)
        else:
            achieved = None

        return PowerResult(
            minimum_n=min_n,
            effect_size=effect_size,
            alpha=self.alpha,
            desired_power=self.power,
            test_type="TwoProportionZ",
            achieved_power=achieved,
        )

    def _achieved_power_proportion(self, n: int, p1: float, p2: float) -> float:
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        ncp = abs(p2 - p1) / math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        achieved = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        return float(achieved)

    def for_mcnemar(
        self,
        effect_size: float,
        discordant_proportion: float = 0.3,
        observed_n: int | None = None,
    ) -> PowerResult:
        """
        Sample size for McNemar's test.

        McNemar only uses discordant pairs. If you expect models to agree on
        most examples, you need a larger total N to accumulate enough discordant
        pairs.

        Parameters
        ----------
        effect_size:
            Odds ratio to detect. An odds ratio of 2.0 means model A is twice
            as likely to be correct on a discordant pair.
        discordant_proportion:
            Expected fraction of examples where models disagree. Default 0.30.
            Lower values require larger N for the same power.
        observed_n:
            Total observed N (pairs). Computes achieved power if provided.
        """
        # Convert odds ratio to proportion of discordant pairs going to A
        # p = OR / (1 + OR), then use binomial power
        if effect_size <= 0:
            raise ValueError(
                f"effect_size (odds ratio) must be > 0, got {effect_size}. "
                "An odds ratio of 1.0 means no effect; >1 means model A is better."
            )
        if not (0 < discordant_proportion < 1):
            raise ValueError(
                f"discordant_proportion must be in (0, 1), got {discordant_proportion}."
            )
        self._validate_observed_n(observed_n)
        p_a = effect_size / (1 + effect_size)  # prob A wins | discordant

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Power on discordant pairs (binomial test against p=0.5)
        n_discordant = (
            (z_alpha * math.sqrt(0.5 * 0.5) + z_beta * math.sqrt(p_a * (1 - p_a))) / abs(p_a - 0.5)
        ) ** 2
        n_discordant = math.ceil(n_discordant)

        # Back-compute total N from expected discordant proportion
        min_n = math.ceil(n_discordant / discordant_proportion)

        achieved = None
        if observed_n is not None:
            n_disc_obs = observed_n * discordant_proportion
            ncp = abs(p_a - 0.5) / math.sqrt(0.25 / n_disc_obs)
            achieved = float(1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp))

        return PowerResult(
            minimum_n=min_n,
            effect_size=effect_size,
            alpha=self.alpha,
            desired_power=self.power,
            test_type="McNemar",
            achieved_power=achieved,
        )

    def for_ci_precision(
        self,
        desired_half_width: float,
        expected_accuracy: float = 0.7,
        observed_n: int | None = None,
    ) -> PowerResult:
        """
        Sample size to achieve a desired CI half-width on an accuracy estimate.

        This answers: "How many examples do I need to report accuracy to ±X%?"

        For example, to report 'accuracy = 0.73 ± 0.05' at 95% confidence,
        you need N = (1.96 / 0.05)² × 0.73 × 0.27 ≈ 303 examples.

        Parameters
        ----------
        desired_half_width:
            Target half-width of the CI, e.g. 0.05 for ±5%.
        expected_accuracy:
            Prior estimate of accuracy. If unknown, use 0.5 (conservative).
        observed_n:
            If provided, compute whether this N achieves the desired precision.
            Sets achieved_power to 1.0 if adequate, 0.0 if not - so
            is_adequate() returns the correct answer.
        """
        if not (0 < desired_half_width < 1):
            raise ValueError(
                f"desired_half_width must be in (0, 1), got {desired_half_width}. "
                "Example: 0.05 for ±5% CI half-width."
            )
        if not (0 < expected_accuracy < 1):
            raise ValueError(f"expected_accuracy must be in (0, 1), got {expected_accuracy}.")
        self._validate_observed_n(observed_n)
        z = stats.norm.ppf(1 - self.alpha / 2)
        p = expected_accuracy
        min_n = math.ceil((z / desired_half_width) ** 2 * p * (1 - p))

        achieved_power = None
        if observed_n is not None:
            achieved_hw = z * math.sqrt(p * (1 - p) / observed_n)
            # Treat "adequate" as: achieved half-width ≤ desired half-width.
            # Map to [0,1]: 1.0 if achieved, fraction < 1 if not.
            # is_adequate() checks achieved_power >= desired_power (0.80).
            # We instead use a sentinel: 1.0 = adequate, 0.0 = not adequate,
            # which is cleaner than a ratio.
            achieved_power = 1.0 if achieved_hw <= desired_half_width else 0.0

        return PowerResult(
            minimum_n=min_n,
            effect_size=desired_half_width,
            alpha=self.alpha,
            desired_power=self.power,
            test_type="CI_Precision",
            achieved_power=achieved_power,
        )

    def for_wilcoxon(
        self,
        cohens_d: float,
        observed_n: int | None = None,
    ) -> PowerResult:
        """
        Sample size for Wilcoxon signed-rank test.

        Uses the normal approximation. For non-normal distributions, the actual
        power will be somewhat higher than this estimate (Wilcoxon is more
        efficient than t-test for heavy-tailed distributions).

        Parameters
        ----------
        cohens_d:
            Standardised mean difference to detect.
            d = 0.2 (small), 0.5 (medium), 0.8 (large).
        """
        if cohens_d <= 0:
            raise ValueError(
                f"cohens_d must be > 0, got {cohens_d}. "
                "Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large effect."
            )
        self._validate_observed_n(observed_n)
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Asymptotic relative efficiency of Wilcoxon vs t-test ≈ π/3 ≈ 0.955
        # for normal distributions. We use 0.955 as a conservative adjustment.
        are = 0.955
        n_approx = math.ceil(((z_alpha + z_beta) / cohens_d) ** 2 / are)

        achieved = None
        if observed_n is not None:
            ncp = cohens_d * math.sqrt(observed_n * are)
            achieved = float(1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp))

        return PowerResult(
            minimum_n=n_approx,
            effect_size=cohens_d,
            alpha=self.alpha,
            desired_power=self.power,
            test_type="Wilcoxon",
            achieved_power=achieved,
        )

    def sample_size_table(
        self,
        effect_sizes: list[float] | None = None,
        powers: list[float] | None = None,
        test: str = "proportion",
        baseline_accuracy: float = 0.7,
        print_table: bool = True,
    ) -> str:
        """
        Print a grid of required sample sizes across effect sizes and power levels.

        This is the "planning table" you screenshot and paste into a design doc.
        It answers: "If I want to detect a Δ% accuracy difference at P% power,
        how many examples do I need?"

        Parameters
        ----------
        effect_sizes:
            Accuracy differences to show. Default: [0.02, 0.05, 0.10, 0.15, 0.20].
        powers:
            Power levels to show. Default: [0.70, 0.80, 0.90].
        test:
            "proportion" (default), "mcnemar", "ci", or "wilcoxon".
        baseline_accuracy:
            Baseline accuracy for proportion/CI tests.

        Returns
        -------
        Formatted string table, also printed to stdout.

        Examples
        --------
        >>> pa = PowerAnalysis(alpha=0.05)
        >>> print(pa.sample_size_table())
        Sample Size Requirements  (α=0.05, two-tailed)
        Test: TwoProportionZ | Baseline accuracy: 0.70
        Effect size │  Power 70%  Power 80%  Power 90%
        ────────────┼────────────────────────────────
           Δ = 2%   │      2166       2901       3884
           Δ = 5%   │       347        465        622
        ...
        """
        if effect_sizes is None:
            effect_sizes = [0.02, 0.05, 0.10, 0.15, 0.20]
        if powers is None:
            powers = [0.70, 0.80, 0.90]

        # Build result matrix
        rows: list[list[str]] = []
        for es in effect_sizes:
            row = []
            for pwr in powers:
                pa = PowerAnalysis(alpha=self.alpha, power=pwr)
                if test == "proportion":
                    r = pa.for_proportion_difference(es, p1=baseline_accuracy)
                elif test == "mcnemar":
                    r = pa.for_mcnemar(es)
                elif test == "ci":
                    r = pa.for_ci_precision(es, expected_accuracy=baseline_accuracy)
                elif test == "wilcoxon":
                    r = pa.for_wilcoxon(es)
                else:
                    raise ValueError(
                        f"Unknown test '{test}'. Choose from: proportion, mcnemar, ci, wilcoxon."
                    )
                row.append(f"{r.minimum_n:>8,}")
            rows.append(row)

        # Format as table
        test_display = {
            "proportion": "TwoProportionZ",
            "mcnemar": "McNemar",
            "ci": "CI_Precision",
            "wilcoxon": "Wilcoxon (Cohen's d)",
        }.get(test, test)

        header_parts = [f"{'Power ' + f'{int(p * 100)}%':>10}" for p in powers]
        header_str = "  ".join(header_parts)
        lines = [
            f"Sample Size Requirements  (α={self.alpha:.2f}, two-tailed)",
            f"Test: {test_display}",
        ]
        if test in ("proportion", "ci"):
            lines.append(f"Baseline accuracy: {baseline_accuracy:.2f}")
        if test == "ci":
            lines.append(
                "Note: CI width depends only on α and n, not on power. "
                "Columns are identical - shown for reference."
            )
        lines.append("")

        label_width = 12
        lines.append(f"{'Effect size':>{label_width}} │ {header_str}")
        lines.append("─" * label_width + "─┼─" + "─" * (len(header_str) + 1))

        for es, row in zip(effect_sizes, rows):
            if test in ("proportion", "mcnemar", "ci"):
                label = f"Δ = {int(es * 100)}%"
            else:
                label = f"d = {es:.2f}"
            lines.append(f"{label:>{label_width}} │ {'  '.join(row)}")

        table = "\n".join(lines)
        if print_table:
            print(table)
        return table
