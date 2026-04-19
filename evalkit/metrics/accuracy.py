"""
Classification and generation accuracy metrics.

All metrics inherit from Metric and return MetricResult objects with bootstrap CIs.
The pattern is: implement `_point_estimate`, let the base class handle the bootstrap.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from evalkit.metrics.base import Metric, MetricResult

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    """
    Fraction of predictions that exactly match the reference label.

    For binary classification, this is the standard accuracy. For multiclass,
    it is micro-averaged (i.e. total correct / total examples). Stratified
    bootstrap CI is always used because accuracy on imbalanced datasets is
    misleading - the CI width reveals how misleading.
    """

    @property
    def name(self) -> str:
        return "Accuracy"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        return float(np.mean(predictions == references))

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
        stratify: bool = True,
    ) -> MetricResult:
        result = super().compute(predictions, references, stratify=stratify)

        # Warn about class imbalance in the reference labels.
        # Note: when called from Experiment._compute_metrics, references=[1]*n so
        # this warning won't fire there - imbalance detection is delegated to
        # RigorChecker.audit(label_distribution=...). This warning fires when
        # users call Accuracy.compute directly with real label arrays.
        refs = np.asarray(references)
        classes, counts = np.unique(refs, return_counts=True)
        if len(classes) > 1:
            majority_frac = counts.max() / counts.sum()
            if majority_frac >= 0.9:
                logger.warning(
                    "Class imbalance detected: %.0f%% of examples are class '%s'. "
                    "Accuracy may be misleading. Consider F1Score or BalancedAccuracy.",
                    majority_frac * 100,
                    classes[np.argmax(counts)],
                )

        return result


class BalancedAccuracy(Metric):
    """
    Mean per-class recall, robust to class imbalance.

    Equivalent to macro-averaged recall. Use this instead of Accuracy
    when classes are imbalanced (>3:1 ratio).

    Note: always use the default `stratify=True` when calling compute().
    With unstratified bootstrap, rare classes may be absent from some
    resamples, making the bootstrap estimate of balanced accuracy biased
    upward (the missing class contributes nothing rather than 0.0).
    """

    @property
    def name(self) -> str:
        return "BalancedAccuracy"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        classes = np.unique(references)
        per_class_recall = []
        for cls in classes:
            mask = references == cls
            if mask.sum() == 0:  # pragma: no cover - stratified bootstrap guarantees presence
                continue
            per_class_recall.append(np.mean(predictions[mask] == cls))
        return float(np.mean(per_class_recall))


class F1Score(Metric):
    """
    F1 score with bootstrap CI.

    Supports binary, macro, and micro averaging. For binary tasks, reports
    per-class breakdown in MetricResult.extra.

    Parameters
    ----------
    average:
        "binary", "macro", "micro", or "weighted". Passed to sklearn.
    pos_label:
        The positive class label for binary F1. Ignored for multiclass.
    """

    def __init__(
        self,
        average: str = "macro",
        pos_label: Any = 1,
        ci_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__(ci_level=ci_level, n_resamples=n_resamples, seed=seed)
        self.average = average
        self.pos_label = pos_label

    @property
    def name(self) -> str:
        return f"F1Score({self.average})"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        return float(
            f1_score(
                references,
                predictions,
                average=self.average,
                pos_label=self.pos_label,
                zero_division=0,
            )
        )

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
        stratify: bool = True,
    ) -> MetricResult:
        result = super().compute(predictions, references, stratify=stratify)

        # Compute per-class F1 for the extra field.
        preds = np.asarray(predictions)
        refs = np.asarray(references)
        classes = np.unique(refs)

        if len(classes) <= 10:  # Skip per-class breakdown for high-cardinality
            per_class = {}
            for cls in classes:
                per_class[str(cls)] = float(
                    f1_score(refs, preds, labels=[cls], average="macro", zero_division=0)
                )
            return MetricResult(
                name=result.name,
                value=result.value,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                ci_level=result.ci_level,
                n=result.n,
                n_resamples=result.n_resamples,
                extra={"per_class_f1": per_class},
            )

        return result


class BLEUScore(Metric):
    """
    BLEU-4 score for text generation tasks.

    Uses corpus-level BLEU (not sentence-level) to avoid sentence brevity
    penalty artifacts. Predictions and references must be strings.

    Note: BLEU is a weak metric. The CI here is honest about how weak -
    on small datasets, the CI width will be very large.
    """

    @property
    def name(self) -> str:
        return "BLEU-4"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        try:
            from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        except ImportError:
            raise ImportError("nltk is required for BLEUScore. pip install nltk")

        smooth = SmoothingFunction().method1
        hypotheses = [pred.split() for pred in predictions.tolist()]
        list_of_references = [[ref.split()] for ref in references.tolist()]

        return float(corpus_bleu(list_of_references, hypotheses, smoothing_function=smooth))

    def compute(self, predictions: list[str], references: list[str]) -> MetricResult:  # type: ignore[override]  # noqa: E501
        """
        Parameters
        ----------
        predictions, references:
            Lists of hypothesis and reference strings. Stratification does not
            apply to text generation metrics - we resample over examples uniformly.
        """
        if len(predictions) < 50:
            logger.warning(
                "BLEU scores on N=%d examples are unreliable. "
                "Corpus-level BLEU requires at least N=50; N≥200 is preferred.",
                len(predictions),
            )
        return super().compute(predictions, references, stratify=False)


class ROUGEScore(Metric):
    """
    ROUGE-L score for summarisation and generation tasks.

    ROUGE-L uses longest common subsequence, making it more robust to
    paraphrasing than ROUGE-1/2. All three are computed internally;
    the MetricResult.value is ROUGE-L, with ROUGE-1/2 in extra.

    Parameters
    ----------
    rouge_type:
        Which ROUGE variant to use as the primary metric.
        One of "rouge1", "rouge2", "rougeL". Default "rougeL".
    """

    def __init__(
        self,
        rouge_type: str = "rougeL",
        ci_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__(ci_level=ci_level, n_resamples=n_resamples, seed=seed)
        self.rouge_type = rouge_type

    @property
    def name(self) -> str:
        return f"ROUGE({self.rouge_type})"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("rouge-score is required. pip install rouge-score")

        scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)[self.rouge_type].fmeasure
            for pred, ref in zip(predictions.tolist(), references.tolist())
        ]
        return float(np.mean(scores))

    def compute(self, predictions: list[str], references: list[str]) -> MetricResult:  # type: ignore[override]  # noqa: E501
        """Stratification does not apply to ROUGE - resampling is over examples uniformly."""
        return super().compute(predictions, references, stratify=False)


class ExpectedCalibrationError:
    """
    Expected Calibration Error (ECE) - measures how well confidence scores
    match observed accuracy.

    A model that says "70% confident" should be correct ~70% of the time.
    ECE is the weighted average absolute difference between confidence and
    accuracy across calibration bins.

    This class does not inherit from Metric because its interface is
    fundamentally different: it takes (correct, confidences) not
    (predictions, references). The bootstrap is implemented directly here.

    Parameters
    ----------
    n_bins:
        Number of confidence bins. 10 is standard; use fewer for small N.
    ci_level:
        Confidence level for the bootstrap CI.
    n_resamples:
        Bootstrap resamples.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_bins: int = 10,
        ci_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> None:
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
        self.n_bins = n_bins
        self.ci_level = ci_level
        self.n_resamples = n_resamples
        self.rng = np.random.default_rng(seed)

    def _ece(self, correct: np.ndarray, confidence: np.ndarray) -> float:
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(correct)
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Use <= on the right edge of the last bin so confidence=1.0 is included.
            if i < self.n_bins - 1:
                mask = (confidence >= low) & (confidence < high)
            else:
                mask = (confidence >= low) & (confidence <= high)
            if mask.sum() == 0:  # pragma: no cover - stratified bootstrap guarantees presence
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidence[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    def compute(self, correct: list[int], confidences: list[float]) -> MetricResult:
        """
        Parameters
        ----------
        correct:
            Binary array: 1 if model was correct, 0 otherwise.
        confidences:
            Model's reported confidence for each prediction, in [0, 1].
        """
        if len(correct) != len(confidences):
            raise ValueError(
                f"correct (n={len(correct)}) and confidences (n={len(confidences)}) "
                "must have the same length."
            )
        if len(correct) == 0:
            raise ValueError("Cannot compute ECE on empty arrays.")

        c = np.asarray(correct, dtype=float)
        conf = np.asarray(confidences, dtype=float)

        if np.any((conf < 0) | (conf > 1)):
            raise ValueError("All confidence scores must be in [0, 1].")

        point = self._ece(c, conf)
        n = len(correct)

        boot_stats = np.empty(self.n_resamples)
        for i in range(self.n_resamples):
            idx = self.rng.integers(0, n, size=n)
            boot_stats[i] = self._ece(c[idx], conf[idx])

        alpha = 1 - self.ci_level
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        ci_lower = min(ci_lower, point)
        ci_upper = max(ci_upper, point)

        if point > 0.15:
            logger.warning(
                "ECE=%.3f indicates poor calibration. "
                "Confidence scores should not be used for decision-making at this level.",
                point,
            )

        return MetricResult(
            name="ECE",
            value=point,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            n=n,
            n_resamples=self.n_resamples,
        )
