"""
Inter-rater agreement metrics.

Used when LLM-as-judge or multiple human raters produce judgments that need
to be validated before being trusted as evaluation signals. Low agreement
means the metric is unreliable - the RigorChecker enforces this.

Cohen's kappa: for two raters on categorical (nominal) judgments.
Krippendorff's alpha: for 2+ raters on ordinal, interval, or continuous scores.

Neither class inherits from Metric. Both have fundamentally different interfaces
from the predictions/references pattern, and forcing them into that mold via
type: ignore comments or interface violations is worse than being honest about
what they are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score

from evalkit.metrics.base import MetricResult

logger = logging.getLogger(__name__)

# Landis & Koch (1977) thresholds, widely used in NLP evaluation literature.
KAPPA_THRESHOLDS = {
    "poor": (float("-inf"), 0.0),
    "slight": (0.0, 0.20),
    "fair": (0.20, 0.40),
    "moderate": (0.40, 0.60),
    "substantial": (0.60, 0.80),
    "almost_perfect": (0.80, float("inf")),
}

# Minimum κ/α for LLM-as-judge results to be considered publication-worthy.
MIN_ACCEPTABLE_KAPPA = 0.60


def _interpret(value: float) -> str:
    """Map a kappa/alpha value to a Landis & Koch verbal label.

    The threshold table covers (-inf, inf) with no gaps, so every finite
    value matches exactly one bucket. Non-finite inputs (NaN, inf) are
    treated as "poor" as a safe conservative default.
    """
    if not (value == value):  # NaN check (NaN != NaN)
        return "poor"
    for label, (lo, hi) in KAPPA_THRESHOLDS.items():
        if lo <= value < hi:
            return label
    return "almost_perfect"  # pragma: no cover - unreachable for finite values


@dataclass(frozen=True)
class AgreementResult:
    """
    Agreement metric with CI and human-readable interpretation.

    Attributes
    ----------
    metric:
        The underlying MetricResult (value = kappa or alpha).
    interpretation:
        Landis & Koch verbal label for the agreement level.
    is_acceptable:
        True if agreement meets the minimum threshold (κ/α ≥ 0.60).
    """

    metric: MetricResult
    interpretation: str
    is_acceptable: bool

    def __str__(self) -> str:
        flag = "✓" if self.is_acceptable else "✗"
        suffix = (
            "" if self.is_acceptable else f" - below minimum threshold ({MIN_ACCEPTABLE_KAPPA})"
        )  # noqa: E501
        return f"{flag} {self.metric} [{self.interpretation}]{suffix}"


class CohenKappa:
    """
    Cohen's kappa for inter-rater agreement on categorical judgments.

    Kappa corrects for chance agreement, unlike raw percent agreement.
    Use this when two raters assign categorical labels (correct/incorrect,
    A/B/C/D for multiple-choice, etc.).

    For ordinal or continuous scores, use KrippendorffAlpha instead -
    kappa treats all disagreements equally and ignores the magnitude of error.

    Parameters
    ----------
    weights:
        None for unweighted (nominal) kappa, "linear" or "quadratic" for
        weighted variants. Only meaningful for ordered categories.
    ci_level:
        Confidence level for bootstrap CI.
    n_resamples:
        Bootstrap resamples. 10,000 for publication, 1,000 for quick checks.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        weights: str | None = None,
        ci_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> None:
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
        self.weights = weights
        self.ci_level = ci_level
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def _kappa(self, rater1: np.ndarray, rater2: np.ndarray) -> float:
        try:
            return float(cohen_kappa_score(rater1, rater2, weights=self.weights))
        except ValueError:
            # sklearn raises if only one class is present in a bootstrap resample.
            # Return 0 (no agreement beyond chance) as a conservative fallback.
            return 0.0

    def compute(self, rater1: list[object], rater2: list[object]) -> AgreementResult:
        """
        Compute Cohen's kappa with a bootstrap CI.

        Parameters
        ----------
        rater1, rater2:
            Lists of categorical judgments from two raters, aligned 1-to-1.
            Must be the same length and contain at least two distinct classes.
        """
        if len(rater1) != len(rater2):
            raise ValueError(
                f"rater1 (n={len(rater1)}) and rater2 (n={len(rater2)}) must be the same length."
            )
        if len(rater1) == 0:
            raise ValueError("Cannot compute kappa on empty arrays.")

        r1 = np.asarray(rater1)
        r2 = np.asarray(rater2)
        n = len(r1)

        point = self._kappa(r1, r2)

        boot_stats = np.empty(self.n_resamples)
        for i in range(self.n_resamples):
            idx = self.rng.integers(0, n, size=n)
            boot_stats[i] = self._kappa(r1[idx], r2[idx])

        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        ci_lower = min(ci_lower, point)
        ci_upper = max(ci_upper, point)

        metric_result = MetricResult(
            name="CohenKappa" + (f"(weighted={self.weights})" if self.weights else ""),
            value=point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            n=n,
            n_resamples=self.n_resamples,
        )

        interp = _interpret(point)
        is_ok = point >= MIN_ACCEPTABLE_KAPPA

        if not is_ok:
            logger.warning(
                "Cohen's kappa=%.3f (%s) is below the acceptable threshold (%.2f). "
                "Judgments from this rater pair are unreliable.",
                point,
                interp,
                MIN_ACCEPTABLE_KAPPA,
            )

        return AgreementResult(metric=metric_result, interpretation=interp, is_acceptable=is_ok)


class KrippendorffAlpha:
    """
    Krippendorff's alpha for inter-rater reliability.

    Prefer this over Cohen's kappa when:
    - you have more than two judges
    - your scores are ordinal or continuous (e.g. 1-10 quality ratings)
    - some items have missing judgments from some raters

    The bootstrap resamples over items (columns of the reliability matrix),
    not individual ratings. This is correct because items are the independent
    observations, not rater-item pairs.

    Parameters
    ----------
    level_of_measurement:
        "nominal", "ordinal", "interval", or "ratio".
        Use "interval" for Likert scales and LLM quality scores (0-1 or 1-5).
    ci_level:
        Confidence level for the bootstrap CI.
    n_resamples:
        Bootstrap resamples.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        level_of_measurement: str = "interval",
        ci_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> None:
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
        self.level = level_of_measurement
        self.ci_level = ci_level
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def compute(self, ratings: list[list[float | None]]) -> AgreementResult:
        """
        Compute Krippendorff's alpha with a bootstrap CI.

        Parameters
        ----------
        ratings:
            Reliability matrix as a list of rater lists.
            ``ratings[i][j]`` is rater i's score for item j.
            Use None for missing values (items a rater did not score).
            Shape: (n_raters, n_items), n_raters ≥ 2.
        """
        try:
            import krippendorff
        except ImportError:
            raise ImportError(
                "krippendorff is required for KrippendorffAlpha. pip install krippendorff"
            )

        if len(ratings) < 2:
            raise ValueError("KrippendorffAlpha requires at least 2 raters.")

        matrix = np.array([[np.nan if v is None else float(v) for v in rater] for rater in ratings])
        n_items = matrix.shape[1]

        point = float(krippendorff.alpha(matrix, level_of_measurement=self.level))

        # Bootstrap over items (columns) - items are independent observations.
        boot_stats = np.empty(self.n_resamples)
        for i in range(self.n_resamples):
            col_idx = self.rng.integers(0, n_items, size=n_items)
            try:
                boot_stats[i] = float(
                    krippendorff.alpha(matrix[:, col_idx], level_of_measurement=self.level)
                )
            except Exception:
                boot_stats[i] = np.nan

        valid = boot_stats[~np.isnan(boot_stats)]
        degenerate_frac = 1 - len(valid) / self.n_resamples
        if degenerate_frac > 0.10:
            logger.warning(
                "%.0f%% of bootstrap resamples were degenerate. CI is unreliable.",
                degenerate_frac * 100,
            )

        if len(valid) == 0:
            ci_lower = ci_upper = point
        else:
            alpha_val = 1 - self.ci_level
            ci_lower = float(np.percentile(valid, 100 * alpha_val / 2))
            ci_upper = float(np.percentile(valid, 100 * (1 - alpha_val / 2)))
            ci_lower = min(ci_lower, point)
            ci_upper = max(ci_upper, point)

        metric_result = MetricResult(
            name=f"KrippendorffAlpha({self.level})",
            value=point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            n=n_items,
            n_resamples=self.n_resamples,
        )

        interp = _interpret(point)
        is_ok = point >= MIN_ACCEPTABLE_KAPPA

        if not is_ok:
            logger.warning(
                "Krippendorff's alpha=%.3f (%s) is below the acceptable threshold (%.2f). "
                "Multi-rater judgments are unreliable at this agreement level.",
                point,
                interp,
                MIN_ACCEPTABLE_KAPPA,
            )

        return AgreementResult(metric=metric_result, interpretation=interp, is_acceptable=is_ok)
