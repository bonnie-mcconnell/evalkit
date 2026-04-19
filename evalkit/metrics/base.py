"""
Base types for evalkit metrics.

Every metric in evalkit returns a MetricResult - never a bare float. This is the
architectural enforcement of the central thesis: point estimates without uncertainty
quantification are not results, they're guesses.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricResult:
    """
    A metric value with its uncertainty quantification.

    The CI bounds are from bootstrap resampling (percentile method) unless
    the metric has an analytic CI. Both are reported to 4 decimal places
    internally; formatting for display is the report layer's responsibility.

    Attributes
    ----------
    name:
        Human-readable metric name, e.g. "Accuracy".
    value:
        Point estimate on [0, 1] for most metrics, or on the metric's
        natural scale (e.g. BLEU is [0, 1], log-likelihood is (-inf, 0)).
    ci_lower:
        Lower bound of the (1 - alpha) confidence interval.
    ci_upper:
        Upper bound.
    ci_level:
        Confidence level, e.g. 0.95 for a 95% CI.
    n:
        Number of examples used to compute this metric. Required for
        downstream power analysis and RigorChecker auditing.
    n_resamples:
        Number of bootstrap resamples used. None if CI is analytic.
    extra:
        Additional metric-specific fields (e.g. per-class F1, confusion matrix).
    """

    name: str
    value: float
    ci_lower: float
    ci_upper: float
    n: int
    ci_level: float = 0.95
    n_resamples: int | None = 10_000
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow a tiny floating-point tolerance (1e-9) for values exactly
        # at a CI boundary after arithmetic, but catch genuine violations.
        if (self.value < self.ci_lower - 1e-9) or (self.value > self.ci_upper + 1e-9):
            raise ValueError(
                f"Point estimate {self.value:.4f} lies outside CI "
                f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]. "
                "This indicates a bug in bootstrap resampling."
            )

    def __str__(self) -> str:
        pct = int(self.ci_level * 100)
        return (
            f"{self.name}: {self.value:.4f} "
            f"({pct}% CI: {self.ci_lower:.4f}–{self.ci_upper:.4f}, n={self.n})"
        )

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval - a direct measure of precision."""
        return self.ci_upper - self.ci_lower

    @property
    def margin_of_error(self) -> float:
        """Half-width of the CI, analogous to polling margin of error."""
        return self.ci_width / 2


class Metric(ABC):
    """
    Abstract base class for all evalkit metrics.

    Subclasses implement `compute`, which takes arrays of predictions and
    ground-truth labels and returns a MetricResult with bootstrap CIs.

    The bootstrap is run here in the base class via `bootstrap_ci` to ensure
    consistent resampling behaviour across all metrics. Subclasses provide
    `_point_estimate`, which computes the metric on a single (possibly
    resampled) array pair.
    """

    def __init__(self, ci_level: float = 0.95, n_resamples: int = 10_000, seed: int = 42) -> None:
        if not (0 < ci_level < 1):
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
        if n_resamples < 1000:
            logger.warning(
                "n_resamples=%d is low. Use ≥1000 for stable CIs, ≥10000 for publication.",
                n_resamples,
            )
        self.ci_level = ci_level
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name returned in MetricResult."""
        ...

    @abstractmethod
    def _point_estimate(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
    ) -> float:
        """
        Compute the metric on a single sample.

        This is called both on the full dataset (for the point estimate) and
        on each bootstrap resample (for the CI). Keep it stateless and fast.
        """
        ...

    def bootstrap_ci(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        stratify: bool = True,
    ) -> tuple[float, float]:
        """
        Percentile bootstrap CI via stratified resampling.

        Stratification ensures that rare-class examples appear in resamples
        in approximately the correct proportion, preventing CI collapse on
        imbalanced datasets. When `stratify=True`, resampling is done
        separately within each unique class in `references` and results
        are concatenated.

        Parameters
        ----------
        predictions, references:
            Aligned arrays of model outputs and ground-truth labels.
        stratify:
            Whether to resample within strata defined by `references`.
            Disable only for regression targets with continuous labels.

        Returns
        -------
        (ci_lower, ci_upper) at self.ci_level.
        """
        n = len(predictions)
        boot_stats = np.empty(self.n_resamples)
        class_indices = self._stratify_indices(references) if stratify else None

        for i in range(self.n_resamples):
            if class_indices is not None:
                idx = self._stratified_sample(class_indices)
            else:
                idx = self.rng.integers(0, n, size=n)
            boot_stats[i] = self._point_estimate(predictions[idx], references[idx])

        alpha = 1 - self.ci_level
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return lower, upper

    def _stratify_indices(self, references: np.ndarray) -> dict[Any, np.ndarray]:
        """Return a mapping from class label to the indices of that class in references."""
        return {cls: np.where(references == cls)[0] for cls in np.unique(references)}

    def _stratified_sample(self, class_indices: dict[Any, np.ndarray]) -> np.ndarray:
        """
        Sample with replacement within each stratum, then concatenate.

        Each stratum is resampled to its original size, preserving marginal
        class frequencies in expectation across resamples.
        """
        parts = []
        for idx in class_indices.values():
            parts.append(self.rng.choice(idx, size=len(idx), replace=True))
        return np.concatenate(parts)

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
        stratify: bool = True,
    ) -> MetricResult:
        """
        Compute the metric with bootstrap CI.

        Parameters
        ----------
        predictions:
            Model outputs. Type depends on the metric subclass.
        references:
            Ground-truth labels, aligned 1-to-1 with predictions.
        stratify:
            Passed through to `bootstrap_ci`.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )
        if len(predictions) == 0:
            raise ValueError("Cannot compute metric on empty arrays.")

        preds = np.asarray(predictions)
        refs = np.asarray(references)

        point = self._point_estimate(preds, refs)
        ci_lower, ci_upper = self.bootstrap_ci(preds, refs, stratify=stratify)

        # Clamp the point estimate to the CI after floating-point arithmetic.
        ci_lower = min(ci_lower, point)
        ci_upper = max(ci_upper, point)

        return MetricResult(
            name=self.name,
            value=point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            n=len(predictions),
            n_resamples=self.n_resamples,
        )
