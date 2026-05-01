"""
Experiment: the top-level object tying dataset, runner, judge, and analysis together.

An Experiment is the unit of work in evalkit. It:
1. Runs pre-flight RigorChecker to warn about design problems before spending money.
2. Executes the runner over the dataset.
3. Computes all configured metrics with bootstrap CIs.
4. Runs the post-hoc RigorChecker audit.
5. Returns an ExperimentResult with everything needed for the report.

Design note: Experiment is intentionally not a god object. It orchestrates
but doesn't implement - metrics, judges, and runners are injected.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

from scipy import stats as _scipy_stats

from evalkit.analysis.rigour import AuditReport, RigorChecker
from evalkit.core.dataset import EvalDataset
from evalkit.core.judge import Judge
from evalkit.core.runner import RunResult
from evalkit.metrics.accuracy import Accuracy
from evalkit.metrics.base import Metric, MetricResult
from evalkit.metrics.comparison import McNemarTest, WilcoxonTest


@runtime_checkable
class _SupportsRun(Protocol):
    """Any runner object evalkit can use - AsyncRunner or MockRunner."""

    judge: Judge

    def run(self, dataset: EvalDataset) -> RunResult: ...


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    """
    Result of comparing two ExperimentResults with a statistical test.

    This is returned by ExperimentResult.compare(other) and contains
    everything you need to report the comparison rigorously. Frozen to
    prevent accidental mutation of result objects.

    Attributes
    ----------
    experiment_a, experiment_b:
        Names of the two experiments being compared.
    test_name:
        Statistical test used ("McNemar" or "Wilcoxon").
    statistic:
        Test statistic value.
    p_value:
        p-value for the null hypothesis (experiments are equivalent).
    effect_size:
        Odds ratio > 1 means A better (McNemar); positive rank-biserial
        correlation means A better (Wilcoxon). Both are on their own scales
        so always read alongside test_name.
    reject_null:
        True if p_value < alpha - experiments are statistically distinguishable.
    alpha:
        Significance level used.
    n_pairs:
        Number of paired observations used.
    note:
        Any caveats about the test (e.g. small discordant-pair count).
    accuracy_a, accuracy_b:
        Point-estimate accuracies for quick reference.
    """

    experiment_a: str
    experiment_b: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    reject_null: bool
    alpha: float
    n_pairs: int
    note: str
    accuracy_a: float
    accuracy_b: float

    @property
    def winner(self) -> str:
        """
        Which experiment performed better, based on the effect size sign.

        For McNemar: odds ratio > 1 means experiment_a wins.
        For Wilcoxon: rank-biserial r > 0 means experiment_a wins.
        Both conventions agree: positive effect → A wins.
        """
        if self.test_name == "McNemar":
            return self.experiment_a if self.effect_size > 1.0 else self.experiment_b
        else:  # Wilcoxon: r > 0 means A better
            return self.experiment_a if self.effect_size > 0.0 else self.experiment_b

    def __str__(self) -> str:
        decision = "REJECT H₀" if self.reject_null else "fail to reject H₀"
        lines = [
            f"Comparison: {self.experiment_a} vs {self.experiment_b}",
            f"  {self.experiment_a}: accuracy = {self.accuracy_a:.4f}",
            f"  {self.experiment_b}: accuracy = {self.accuracy_b:.4f}",
            f"  {self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}, "
            f"effect={self.effect_size:.4f} → {decision} (α={self.alpha})",
        ]
        if self.reject_null:
            lines.append(f"  ✓ {self.winner} is statistically better (p={self.p_value:.4f})")
        else:
            lines.append(
                f"  The difference is NOT statistically significant. "
                f"Increase N to ≥{self._approx_required_n():,} to detect this effect."
            )
        if self.note:
            lines.append(f"  Note: {self.note}")
        return "\n".join(lines)

    def _approx_required_n(self) -> int:
        """Rough N estimate to detect the observed accuracy difference at 80% power."""
        delta = abs(self.accuracy_a - self.accuracy_b)
        if delta < 0.001:
            return 99_999
        p_bar = (self.accuracy_a + self.accuracy_b) / 2
        z_a = _scipy_stats.norm.ppf(0.975)
        z_b = _scipy_stats.norm.ppf(0.80)
        n = (
            (
                z_a * math.sqrt(2 * p_bar * (1 - p_bar))
                + z_b
                * math.sqrt(
                    self.accuracy_a * (1 - self.accuracy_a)
                    + self.accuracy_b * (1 - self.accuracy_b)
                )
            )
            / delta
        ) ** 2
        return int(math.ceil(n))


@dataclass(frozen=True)
class ExperimentResult:
    """
    The complete output of an Experiment run.

    Attributes
    ----------
    run_result:
        Raw per-example results from the runner.
    metrics:
        Dict of metric name → MetricResult with bootstrap CIs.
    preflight_audit:
        RigorChecker results from before the run.
    posthoc_audit:
        RigorChecker results from after the run.
    experiment_name:
        Identifier for this experiment.
    """

    run_result: RunResult
    metrics: dict[str, MetricResult]
    preflight_audit: AuditReport
    posthoc_audit: AuditReport
    experiment_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print a concise human-readable summary. Useful in scripts and notebooks."""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Dataset: {self.run_result.dataset_name} (n={self.run_result.n})")
        print(f"Model:   {self.run_result.model}")
        print(
            f"Cost:    ${self.run_result.total_cost_usd:.4f} | "
            f"Tokens: {self.run_result.total_tokens:,} | "
            f"Time: {self.run_result.wall_time_seconds:.1f}s"
        )
        print()
        print("Metrics (with 95% bootstrap CIs):")
        for result in self.metrics.values():
            print(f"  {result}")
        print()
        print(str(self.posthoc_audit))

    def compare(
        self,
        other: ExperimentResult,
        alpha: float = 0.05,
        test: str = "auto",
    ) -> ComparisonResult:
        """
        Compare this experiment against another with a paired significance test.

        Both experiments must have been run on the same dataset (same example IDs
        in the same order). This is verified before the test runs.

        Parameters
        ----------
        other:
            The experiment to compare against.
        alpha:
            Significance level.
        test:
            "mcnemar" for binary outcomes, "wilcoxon" for continuous scores,
            or "auto" to choose based on the judge type (default).

        Returns
        -------
        ComparisonResult with the test statistic, p-value, effect size, and
        a plain-English interpretation including required N if not significant.

        Examples
        --------
        >>> result_a = Experiment("gpt4o", dataset, runner_a).run()
        >>> result_b = Experiment("gpt4o-mini", dataset, runner_b).run()
        >>> comparison = result_a.compare(result_b)
        >>> print(comparison)
        """
        ids_a = self.run_result.example_ids
        ids_b = other.run_result.example_ids

        if ids_a != ids_b:
            mismatches = sum(a != b for a, b in zip(ids_a, ids_b))
            raise ValueError(
                f"Cannot compare experiments with different example sets. "
                f"{mismatches} example IDs differ between "
                f"'{self.experiment_name}' and '{other.experiment_name}'. "
                "Both experiments must be run on the same dataset in the same order."
            )

        # Determine test: auto → use scores if stochastic judge, binary otherwise
        use_wilcoxon = test == "wilcoxon" or (
            test == "auto" and any(s not in (0.0, 1.0) for s in self.run_result.scores)
        )

        if use_wilcoxon:
            result = WilcoxonTest(alpha=alpha).test(self.run_result.scores, other.run_result.scores)
        else:
            result = McNemarTest(alpha=alpha).test(
                self.run_result.correct, other.run_result.correct
            )

        acc_a = self.metrics["Accuracy"].value
        acc_b = other.metrics["Accuracy"].value

        return ComparisonResult(
            experiment_a=self.experiment_name,
            experiment_b=other.experiment_name,
            test_name=result.test_name,
            statistic=result.statistic,
            p_value=result.p_value,
            effect_size=result.effect_size,
            reject_null=result.reject_null,
            alpha=result.alpha,
            n_pairs=result.n_pairs,
            note=result.note,
            accuracy_a=acc_a,
            accuracy_b=acc_b,
        )

    def worst_examples(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Return the n examples the model most confidently got wrong.

        Useful for error analysis - understanding *why* the model fails
        is often more informative than the aggregate accuracy number.

        Parameters
        ----------
        n:
            Number of examples to return. Default 10.

        Returns
        -------
        List of dicts, each with keys: example_id, prompt, output,
        reference, score, sorted by score descending (most confident wrongs first).
        """
        wrong = [
            {
                "example_id": r.example_id,
                "prompt": r.prompt,
                "output": r.output,
                "reference": r.reference,
                "score": r.score,
                "reasoning": r.judgment.reasoning,
            }
            for r in self.run_result.example_results
            if not r.is_correct
        ]
        # Sort by score descending - high score on a wrong answer = confident mistake
        wrong.sort(key=lambda x: x["score"], reverse=True)
        return wrong[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return per-example results as a pandas DataFrame.

        Columns: example_id, prompt, output, reference, is_correct, score,
                 reasoning, latency_ms.

        Requires pandas. Raises ImportError with install instructions if
        pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas") from None

        rows = [
            {
                "example_id": r.example_id,
                "prompt": r.prompt,
                "output": r.output,
                "reference": r.reference,
                "is_correct": r.is_correct,
                "score": r.score,
                "reasoning": r.judgment.reasoning,
                "latency_ms": r.latency_ms,
            }
            for r in self.run_result.example_results
        ]
        return pd.DataFrame(rows)


class Experiment:
    """
    Top-level orchestrator for an LLM evaluation.

    Parameters
    ----------
    name:
        Identifier for this experiment. Appears in reports and checkpoints.
    dataset:
        The EvalDataset to evaluate on.
    runner:
        AsyncRunner or MockRunner that will call the model.
    additional_metrics:
        Extra Metric instances to compute alongside Accuracy. Each metric
        receives ``run_result.outputs`` as predictions and
        ``run_result.references`` as references, so class-level metrics
        (``BalancedAccuracy``, ``F1Score``) and generation metrics (BLEU,
        ROUGE) all work correctly here::

            result = Experiment(
                "my_eval", dataset, runner,
                additional_metrics=[BalancedAccuracy(), F1Score()],
            ).run()
    rigour_checker:
        RigorChecker instance. Customise thresholds here if needed.
    n_variants:
        Number of prompt variants being compared in this experiment batch.
        Used by RigorChecker for multiple testing warnings.
    judge_kappa:
        Pre-measured inter-rater agreement (κ) for the judge. Required for
        LLM-as-judge experiments - the RigorChecker will flag its absence.
    n_resamples:
        Number of bootstrap resamples for CI computation. Default 10,000
        is publication-quality. Use 1,000 for quick iteration.
    """

    def __init__(
        self,
        name: str,
        dataset: EvalDataset,
        runner: _SupportsRun,
        additional_metrics: list[Metric] | None = None,
        rigour_checker: RigorChecker | None = None,
        n_variants: int = 1,
        judge_kappa: float | None = None,
        n_resamples: int = 10_000,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.runner = runner
        self.additional_metrics = additional_metrics or []
        self.rigour_checker = rigour_checker or RigorChecker()
        self.n_variants = n_variants
        self.judge_kappa = judge_kappa
        self.n_resamples = n_resamples

    def run(self) -> ExperimentResult:
        """
        Execute the experiment and return a complete ExperimentResult.

        Runs pre-flight audit, evaluates the model, computes metrics with
        bootstrap CIs, then runs the post-hoc audit. Pre-flight errors are
        logged but do not halt execution - the post-hoc audit is the
        authoritative record of whether results are trustworthy.
        """
        judge_type = "llm" if self.runner.judge.is_stochastic else "deterministic"
        preflight = self.rigour_checker.pre_flight(
            n_examples=len(self.dataset),
            n_variants=self.n_variants,
            judge_type=judge_type,
            experiment_name=self.name,
        )

        for err in preflight.errors:
            logger.error("Pre-flight: %s", err.message)
        for warn in preflight.warnings:
            logger.warning("Pre-flight: %s", warn.message)

        run_result = self.runner.run(self.dataset)
        metrics = self._compute_metrics(run_result)

        posthoc = self.rigour_checker.audit(
            n_examples=run_result.n,
            accuracy=metrics["Accuracy"].value,
            label_distribution=self.dataset.label_distribution(),
            n_variants=self.n_variants,
            judge_kappa=self.judge_kappa,
            experiment_name=self.name,
        )

        return ExperimentResult(
            run_result=run_result,
            metrics=metrics,
            preflight_audit=preflight,
            posthoc_audit=posthoc,
            experiment_name=self.name,
            metadata={"runner_model": run_result.model},
        )

    def _compute_metrics(self, run_result: RunResult) -> dict[str, MetricResult]:
        """
        Compute accuracy and any additional metrics on the run results.

        The default metric is Accuracy - the fraction of examples the judge
        scored as correct, with a stratified bootstrap CI.

        Additional metrics receive ``run_result.outputs`` as predictions and
        ``run_result.references`` as references, so class-level metrics like
        ``BalancedAccuracy`` and ``F1Score`` see the actual model outputs and
        ground-truth labels rather than the binary correct/incorrect array.
        Generation metrics (BLEU, ROUGE) also work correctly via this path.
        """
        metrics: dict[str, MetricResult] = {}
        correct = run_result.correct  # list[int], 1 or 0

        metrics["Accuracy"] = Accuracy(n_resamples=self.n_resamples).compute(
            correct, [1] * len(correct)
        )

        for metric in self.additional_metrics:
            try:
                result = metric.compute(run_result.outputs, run_result.references)
                metrics[result.name] = result
            except Exception as e:
                logger.warning("Additional metric %s failed: %s", type(metric).__name__, e)

        return metrics
