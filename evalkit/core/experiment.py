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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

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


class PreFlightError(RuntimeError):
    """
    Raised when a strict-mode Experiment encounters pre-flight ERROR findings.

    Attributes
    ----------
    audit:
        The full AuditReport containing all pre-flight findings. Inspect
        ``error.audit.errors`` for the specific issues that blocked execution.

    Example
    -------
    >>> try:
    ...     result = experiment.run()
    ... except PreFlightError as e:
    ...     print(e.audit)          # pretty-print the full report
    ...     for f in e.audit.errors:
    ...         print(f.code, f.action)
    """

    def __init__(self, message: str, audit: AuditReport) -> None:
        super().__init__(message)
        self.audit = audit


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

        Raises
        ------
        ValueError
            If ``test_name`` is not a recognised test. This prevents a silent
            wrong answer if a new test is added in future without updating this
            property. An unknown test's effect size convention may differ from
            both McNemar and Wilcoxon, so defaulting silently is incorrect.
        """
        if self.test_name == "McNemar":
            # Odds ratio: > 1 means A had more correct-B-wrong cases than B had
            # correct-A-wrong cases.  OR > 1 → A wins.
            return self.experiment_a if self.effect_size > 1.0 else self.experiment_b
        elif self.test_name == "Wilcoxon":
            # Rank-biserial correlation r: positive means A scores tend higher.
            # r > 0 → A wins.
            return self.experiment_a if self.effect_size > 0.0 else self.experiment_b
        else:
            raise ValueError(
                f"Cannot determine winner: unknown test_name {self.test_name!r}. "
                "Expected 'McNemar' or 'Wilcoxon'. If a new test was added, update "
                "ComparisonResult.winner with its effect size convention."
            )

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
                f"Increase N to ≥{self._approx_required_n():,} to detect this effect "
                f"(conservative estimate via two-proportion z-test; actual required N "
                f"for a paired {self.test_name} test will be equal or lower)."
            )
        if self.note:
            lines.append(f"  Note: {self.note}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """
        Save the comparison result to a JSON file.

        Useful for attaching to GitHub PRs or CI artefacts::

            comparison = result_a.compare(result_b)
            comparison.save("results/comparison.json")

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created automatically.

        Returns
        -------
        The resolved path of the saved file.
        """
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_a": self.experiment_a,
            "experiment_b": self.experiment_b,
            "accuracy_a": self.accuracy_a,
            "accuracy_b": self.accuracy_b,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "reject_null": self.reject_null,
            "alpha": self.alpha,
            "winner": self.winner,
            "note": self.note,
            "approx_required_n": self._approx_required_n() if not self.reject_null else None,
        }
        dest.write_text(json.dumps(payload, indent=2))
        return dest

    def _approx_required_n(self) -> int:
        """Approximate N to detect the observed accuracy difference at 80% power.

        **Approximation caveat**: this uses the two-proportion z-test power formula,
        not the exact McNemar or Wilcoxon power formula. This is intentional: the
        exact paired-test N requires knowing the discordant-pair fraction (McNemar)
        or the score correlation structure (Wilcoxon), which are not available from
        a single comparison result without access to the raw data.

        The two-proportion z-test is *conservative* relative to the paired tests:
        it ignores the pairing, which always reduces variance and therefore reduces
        the true required N. The displayed N is thus an upper bound - the actual
        required N for a paired test will be equal or lower. This errs in the safe
        direction: it never underestimates how much data you need.

        For exact paired-test power analysis, use ``PowerAnalysis`` directly with
        the full run results.
        """
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
    # --- Mutable accumulator inside a frozen dataclass ---
    #
    # frozen=True prevents *attribute reassignment* (self._comparison_p_values = [])
    # but does NOT prevent *mutation of the object the attribute points to*
    # (self._comparison_p_values.append(x) is fine). This is standard Python
    # semantics: frozen prevents rebinding, not mutation of the referent.
    #
    # default_factory=list guarantees each ExperimentResult instance gets its
    # own independent list - there is no shared mutable state between instances.
    #
    # This pattern is intentional: audit_comparisons() needs p-values from
    # every compare() call made after construction, but those calls happen at
    # user-controlled times we cannot know at __init__. The alternative -
    # a separate ComparisonTracker object - would require users to thread it
    # through every compare() call, which is worse ergonomics for no gain.
    # The semantic oddness is real but the tradeoff is justified.
    #
    # See: https://docs.python.org/3/library/dataclasses.html#frozen-instances
    _comparison_p_values: list[float] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        acc = self.metrics.get("Accuracy")
        acc_str = f"{acc.value:.4f} (CI {acc.ci_lower:.4f}–{acc.ci_upper:.4f})" if acc else "n/a"
        status = "PASS" if self.posthoc_audit.passed else "FAIL"
        return (
            f"ExperimentResult("
            f"name={self.experiment_name!r}, "
            f"n={self.run_result.n}, "
            f"accuracy={acc_str}, "
            f"audit={status})"
        )

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

        if test not in ("auto", "mcnemar", "wilcoxon"):
            raise ValueError(
                f"test={test!r} is not valid. Use 'auto' (default), 'mcnemar', or 'wilcoxon'."
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

        comparison = ComparisonResult(
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

        # Track p-values on both results so the RigorChecker can verify
        # BH correction was applied when n_variants > 1.
        self._comparison_p_values.append(result.p_value)
        other._comparison_p_values.append(result.p_value)

        return comparison

    def audit_comparisons(self) -> AuditReport:
        """
        Run the RigorChecker multiple-testing audit on all comparisons made
        via ``.compare()`` since this experiment was run.

        Call this after all ``.compare()`` calls to verify BH-FDR correction
        was applied appropriately. This closes the multiple-testing audit loop
        that cannot be done during ``run()`` because comparisons happen after.

        Returns
        -------
        AuditReport with findings specific to multiple-testing correctness.

        Example
        -------
        >>> results = [exp.run() for exp in experiments]
        >>> comparisons = [results[0].compare(r) for r in results[1:]]
        >>> bh = BHCorrection().correct([c.p_value for c in comparisons])
        >>> audit = results[0].audit_comparisons()
        >>> print(audit)
        """
        from evalkit.analysis.rigour import AuditFinding, AuditReport, RigorChecker, Severity

        p_vals = self._comparison_p_values
        if not p_vals:
            return AuditReport(findings=[], experiment_name=self.experiment_name)

        # BH correction requires ≥2 comparisons. For a single comparison, return
        # an INFO finding that explicitly communicates "not applicable" rather than
        # a clean pass - a clean pass would be ambiguous: does it mean "checked and
        # passed" or "nothing to check"? The INFO finding removes that ambiguity.
        if len(p_vals) < 2:
            return AuditReport(
                findings=[
                    AuditFinding(
                        code="MULTIPLE_TESTING_NOT_APPLICABLE",
                        severity=Severity.INFO,
                        message=(
                            "Only one comparison was made. Benjamini-Hochberg FDR correction "
                            "requires ≥2 comparisons and was not applied (not needed here)."
                        ),
                        action=(
                            "No action required. If you make additional comparisons in future, "
                            "call audit_comparisons() again to verify BH correction was applied."
                        ),
                    )
                ],
                experiment_name=self.experiment_name,
            )

        rc = RigorChecker()
        return rc.audit(
            n_examples=self.run_result.n,
            accuracy=self.metrics["Accuracy"].value,
            label_distribution=None,
            n_variants=len(p_vals) + 1,
            p_values=p_vals if len(p_vals) >= 2 else None,
            experiment_name=self.experiment_name,
        )

    def worst_examples(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Return the n examples the model got wrong, sorted by score descending.

        For stochastic judges (LLMJudge, SemanticSimilarityJudge), score varies
        continuously and "high score on a wrong answer" means a confidently wrong
        answer - the most revealing failure mode. For deterministic judges
        (ExactMatchJudge, RegexMatchJudge), all wrong examples have score=0.0 so
        the sort order is effectively insertion order.

        Useful for error analysis: understanding *why* the model fails is often
        more informative than the aggregate accuracy number.

        Parameters
        ----------
        n:
            Number of examples to return. Default 10.

        Returns
        -------
        List of dicts, each with keys: example_id, prompt, output,
        reference, score, reasoning. Sorted by score descending.
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
        # Primary sort: score descending (confident mistakes first).
        # Secondary sort: example_id ascending for deterministic ordering
        # when scores are tied (e.g. ExactMatchJudge where all wrong answers
        # have score=0.0, making insertion order meaningless without this).
        wrong.sort(key=lambda x: (-x["score"], str(x["example_id"])))
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
            raise ImportError(
                "pandas is required for to_dataframe(). "
                'Install with: pip install "evalkit-research[dataframe]"'
            ) from None

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

    def save(self, path: str | Path) -> Path:
        """
        Save results to a JSON file for later analysis or CLI comparison.

        The saved format is the same as ``evalkit run --save-results file.json``.
        Saved files can be passed directly to ``evalkit compare``::

            result_a.save("results/model_a.json")
            result_b.save("results/model_b.json")
            # Then on the command line:
            # evalkit compare results/model_a.json results/model_b.json

        Parent directories are created automatically.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created if they
            do not exist. ``.json`` extension recommended.

        Returns
        -------
        The resolved path of the saved file.

        Examples
        --------
        >>> result = Experiment("my_eval", dataset, runner).run()
        >>> result.save("results/my_eval.json")
        PosixPath('results/my_eval.json')
        """
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "status": "complete",
            "experiment_name": self.experiment_name,
            "model": self.run_result.model,
            "dataset_name": self.run_result.dataset_name,
            "n": self.run_result.n,
            "accuracy": self.metrics["Accuracy"].value,
            "metrics": {
                name: {
                    "value": m.value,
                    "ci_lower": m.ci_lower,
                    "ci_upper": m.ci_upper,
                    "ci_level": m.ci_level,
                    "n": m.n,
                }
                for name, m in self.metrics.items()
            },
            "example_ids": self.run_result.example_ids,
            "correct": self.run_result.correct,
            "scores": self.run_result.scores,
            "total_cost_usd": self.run_result.total_cost_usd,
            "total_tokens": self.run_result.total_tokens,
            "audit_passed": self.posthoc_audit.passed,
            "audit_findings": [
                {
                    "code": f.code,
                    "severity": f.severity.value,
                    "message": f.message,
                    "action": f.action,
                }
                for f in self.posthoc_audit.findings
            ],
        }

        dest.write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", dest)
        return dest

    def generate_report(self, path: str | Path | None = None) -> str:
        """
        Generate a self-contained HTML tearsheet.

        Parameters
        ----------
        path:
            If provided, write the HTML to this file path and return the path
            as a string. If None, return the HTML string without writing.

        Returns
        -------
        HTML string if path is None, otherwise the resolved file path.

        Examples
        --------
        >>> result.generate_report("results/my_eval.html")
        'results/my_eval.html'
        >>> html = result.generate_report()  # in-memory
        """
        from evalkit.analysis.report import ReportGenerator

        html = ReportGenerator().generate(self)
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(html)
            logger.info("Report written to %s", dest)
            return str(dest)
        return html


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
    strict:
        If True (default), pre-flight ERROR findings raise PreFlightError
        before the runner executes - no API budget is spent.
        Set False to run anyway and rely on the post-hoc audit instead.
        This is the setting that makes "catch problems before spending money"
        a real guarantee rather than just a log message.
    expected_accuracy:
        Prior estimate of model accuracy for the pre-flight CI precision check.
        Default 0.70. The CI width formula ``p*(1-p)/n`` is maximised at p=0.5
        and decreases as p → 0 or p → 1, so p=0.70 is a slightly conservative
        prior for most models. If your model is expected to perform at e.g.
        p=0.95, the pre-flight CI check will slightly over-estimate the required
        N (by a factor of ``0.70*0.30 / (0.95*0.05)`` ≈ 4×). For high-accuracy
        models pass the actual expected accuracy to get accurate pre-flight
        N recommendations.
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
        strict: bool = True,
        expected_accuracy: float = 0.70,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.runner = runner
        self.additional_metrics = additional_metrics or []
        self.rigour_checker = rigour_checker or RigorChecker()
        self.n_variants = n_variants
        self.judge_kappa = judge_kappa
        self.n_resamples = n_resamples
        self.strict = strict
        self.expected_accuracy = expected_accuracy

    def run(self) -> ExperimentResult:
        """
        Execute the experiment and return a complete ExperimentResult.

        Runs pre-flight audit, then evaluates the model, computes metrics with
        bootstrap CIs, and runs the post-hoc audit.

        When ``strict=True`` (the default), any pre-flight ERROR halts execution
        immediately - no API budget is spent. This makes the "catch problems
        before spending money" promise a real guarantee. Set ``strict=False`` on
        the Experiment to log errors and continue anyway.
        """
        judge_type = "llm" if self.runner.judge.is_stochastic else "deterministic"
        preflight = self.rigour_checker.pre_flight(
            n_examples=len(self.dataset),
            n_variants=self.n_variants,
            expected_accuracy=self.expected_accuracy,
            judge_type=judge_type,
            experiment_name=self.name,
        )

        for warn in preflight.warnings:
            logger.warning("Pre-flight: %s", warn.message)

        if preflight.errors:
            for err in preflight.errors:
                logger.error("Pre-flight: %s", err.message)
            if self.strict:
                codes = ", ".join(f.code for f in preflight.errors)
                raise PreFlightError(
                    f"Pre-flight audit failed ({codes}). "
                    "The experiment was not run to avoid wasting API budget. "
                    "Fix the issues above, or pass strict=False to Experiment() "
                    "to run anyway and rely on the post-hoc audit.",
                    audit=preflight,
                )

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

        # Compute Accuracy on the binary correct array using the public API.
        # warn_on_imbalance=False because the correct[] array always has refs=[1]*n -
        # any imbalance detection on the real label distribution is RigorChecker's job.
        acc_metric = Accuracy(n_resamples=self.n_resamples)
        preds_arr = np.asarray(correct)
        refs_arr = np.ones(len(correct), dtype=preds_arr.dtype)
        result = acc_metric.compute(
            preds_arr.tolist(), refs_arr.tolist(), stratify=False, warn_on_imbalance=False
        )
        metrics["Accuracy"] = result

        for metric in self.additional_metrics:
            try:
                result = metric.compute(run_result.outputs, run_result.references)
                metrics[result.name] = result
            except ImportError as e:
                # Optional dependency not installed - skip gracefully and tell the user.
                logger.warning(
                    "Additional metric %s skipped: missing optional dependency (%s). "
                    'Install it with: pip install "evalkit-research[generation]" or the '
                    "relevant extra.",
                    type(metric).__name__,
                    e,
                )
            except Exception as e:
                # Any other failure is a bug - raise so it isn't silently hidden.
                raise RuntimeError(
                    f"Additional metric {type(metric).__name__!r} raised an unexpected error. "
                    f"Check that your metric's _point_estimate handles the output types "
                    f"produced by your runner. Original error: {e}"
                ) from e

        return metrics
