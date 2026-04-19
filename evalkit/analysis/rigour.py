"""
RigorChecker: Automated statistical audit for LLM evaluation experiments.

This is the central thesis made executable. Every evalkit experiment runs
through this audit automatically. The output is the kind of thing that should
make users uncomfortable - in a productive way.

Design principle: the checker surfaces *actionable* problems, not just flags.
Every warning explains what the problem is, why it matters, and what to do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from evalkit.analysis.power import PowerAnalysis
from evalkit.metrics.agreement import MIN_ACCEPTABLE_KAPPA
from evalkit.metrics.comparison import BHCorrection

logger = logging.getLogger(__name__)


class Severity(Enum):
    """
    Audit finding severity levels.

    ERROR: Results are unreliable and should not be reported.
    WARNING: Results require caveats; proceed with caution.
    INFO: Minor issues or best-practice suggestions.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AuditFinding:
    """
    A single finding from the RigorChecker.

    Every finding has a code (for programmatic filtering), a human-readable
    message, and an optional suggested action.
    """

    code: str
    severity: Severity
    message: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        icons = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}
        icon = icons[self.severity]
        lines = [f"{icon} [{self.code}] {self.message}", f"   → {self.action}"]
        return "\n".join(lines)


_SEVERITY_SORT_KEY = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def _severity_sort_key(finding: AuditFinding) -> int:
    """Sort key for AuditFindings: errors first, then warnings, then info.

    This is the single definition of finding sort order. report.py and cli.py
    import this function rather than the dict, so there is no leaking of the
    raw dict across module boundaries.
    """
    return _SEVERITY_SORT_KEY[finding.severity]


@dataclass
class AuditReport:
    """
    The complete output of a RigorChecker pass.

    Attributes
    ----------
    findings:
        All findings, sorted by severity (errors first).
    passed:
        True if there are no ERROR-level findings.
    experiment_name:
        Name of the experiment being audited.
    """

    findings: list[AuditFinding]
    experiment_name: str = "unnamed"

    @property
    def passed(self) -> bool:
        return not any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    def __str__(self) -> str:
        if not self.findings:
            return "✅ RigorChecker: No issues found. Experiment appears statistically sound."

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║           evalkit  RigorChecker  Report              ║",
            "╚══════════════════════════════════════════════════════╝",
            f"Experiment: {self.experiment_name}",
            f"Status: {'PASS' if self.passed else 'FAIL'}  "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)",
            "",
        ]

        for finding in sorted(self.findings, key=_severity_sort_key):
            lines.append(str(finding))
            lines.append("")

        if not self.passed:
            lines.append(
                "⚠  Results from this experiment should not be reported without "
                "addressing the ERROR-level findings above."
            )

        return "\n".join(lines)


class RigorChecker:
    """
    Automated statistical audit for evalkit experiments.

    Run pre-flight (before expensive API calls) to catch design problems.
    Run post-hoc (after results arrive) to audit the completed experiment.

    The two-pass design is intentional: pre-flight prevents wasted spend;
    post-hoc produces the shareable audit trail.

    Parameters
    ----------
    power_alpha:
        Significance level for power calculations (default 0.05).
    desired_power:
        Minimum acceptable statistical power (default 0.80).
    min_n:
        Absolute minimum sample size, regardless of power (default 30).
    min_judge_agreement:
        Minimum Cohen's kappa / Krippendorff's alpha for LLM-as-judge
        results to be considered reliable (default 0.60).
    """

    def __init__(
        self,
        power_alpha: float = 0.05,
        desired_power: float = 0.80,
        min_n: int = 30,
        min_judge_agreement: float = MIN_ACCEPTABLE_KAPPA,
    ) -> None:
        self.power = PowerAnalysis(alpha=power_alpha, power=desired_power)
        self.desired_power = desired_power
        self.min_n = min_n
        self.min_judge_agreement = min_judge_agreement

    # ------------------------------------------------------------------ #
    # Pre-flight checks (call BEFORE running the experiment)              #
    # ------------------------------------------------------------------ #

    def pre_flight(
        self,
        n_examples: int,
        n_variants: int = 1,
        expected_accuracy: float = 0.7,
        desired_half_width: float = 0.05,
        judge_type: str = "deterministic",
        experiment_name: str = "unnamed",
    ) -> AuditReport:
        """
        Pre-flight checks to catch problems before spending money on API calls.

        Parameters
        ----------
        n_examples:
            How many examples you plan to evaluate.
        n_variants:
            Number of prompt variants or models being compared.
        expected_accuracy:
            Prior estimate of model accuracy (used for power calculation).
        desired_half_width:
            Target CI half-width, e.g. 0.05 for accuracy ± 5%.
        judge_type:
            "deterministic" (exact match), "llm" (LLM-as-judge), or "human".
            LLM judges require agreement validation.
        """
        if n_examples < 0:
            raise ValueError(f"n_examples must be non-negative, got {n_examples}")

        findings: list[AuditFinding] = []

        # Check 1: Absolute minimum sample size
        findings.extend(self._check_absolute_minimum(n_examples))

        # Check 2: CI precision
        findings.extend(self._check_ci_precision(n_examples, expected_accuracy, desired_half_width))

        # Check 3: Multiple testing warning (pre-flight version)
        if n_variants > 1:
            findings.extend(self._check_multiple_testing_design(n_variants))

        # Check 4: LLM judge agreement reminder
        if judge_type == "llm":
            findings.append(
                AuditFinding(
                    code="JUDGE_AGREEMENT_REQUIRED",
                    severity=Severity.INFO,
                    message=(
                        "You are using an LLM judge. Inter-rater agreement has not been measured."
                    ),
                    action=(
                        "Run a subset (N≥50) through both your LLM judge and human raters, "
                        "then compute Cohen's kappa. κ < 0.60 means the judge is unreliable."
                    ),
                )
            )

        return AuditReport(findings=findings, experiment_name=experiment_name)

    # ------------------------------------------------------------------ #
    # Post-hoc audit (call AFTER results arrive)                          #
    # ------------------------------------------------------------------ #

    def audit(
        self,
        n_examples: int,
        accuracy: float | None = None,
        label_distribution: dict[str, int] | None = None,
        n_variants: int = 1,
        p_values: list[float] | None = None,
        judge_kappa: float | None = None,
        experiment_name: str = "unnamed",
    ) -> AuditReport:
        """
        Post-hoc audit of a completed experiment.

        Parameters
        ----------
        n_examples:
            Total examples evaluated.
        accuracy:
            Observed accuracy (used for achieved-power calculation).
        label_distribution:
            Dict mapping label → count. Used to detect class imbalance.
        n_variants:
            Number of prompt variants / models that were compared.
        p_values:
            Raw (unadjusted) p-values from comparisons. If n_variants > 1
            and these are provided, BH correction applicability is checked.
        judge_kappa:
            Measured inter-rater agreement (κ or α) for LLM-as-judge.
        """
        findings: list[AuditFinding] = []

        if accuracy is not None and not (0.0 <= accuracy <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        if judge_kappa is not None and not (-1.0 <= judge_kappa <= 1.0):
            raise ValueError(f"judge_kappa must be in [-1, 1], got {judge_kappa}")
        if n_examples < 0:
            raise ValueError(f"n_examples must be non-negative, got {n_examples}")

        # Check 1: Absolute minimum - always run regardless of other checks.
        findings.extend(self._check_absolute_minimum(n_examples))

        # Check 2: Power analysis - only if we have observed accuracy to work with.
        if accuracy is not None and n_examples >= self.min_n:
            findings.extend(self._check_achieved_power(n_examples, accuracy))

        # Check 3: Class imbalance
        if label_distribution:
            findings.extend(self._check_class_imbalance(label_distribution))

        # Check 4: Multiple testing correction
        if n_variants > 1 and p_values is not None:
            findings.extend(self._check_multiple_testing_results(n_variants, p_values))
        elif n_variants > 1:
            findings.append(
                AuditFinding(
                    code="MULTIPLE_TESTING_NO_PVALUES",
                    severity=Severity.WARNING,
                    message=(
                        f"You compared {n_variants} variants, but no p-values were provided "
                        "for multiple testing correction review."
                    ),
                    action=(
                        "Pass p_values to RigorChecker.audit() to check whether BH correction "
                        "changes your conclusions."
                    ),
                )
            )

        # Check 5: Judge agreement
        if judge_kappa is not None:
            findings.extend(self._check_judge_agreement(judge_kappa))

        return AuditReport(findings=findings, experiment_name=experiment_name)

    # ------------------------------------------------------------------ #
    # Individual checks                                                   #
    # ------------------------------------------------------------------ #

    def _check_absolute_minimum(self, n: int) -> list[AuditFinding]:
        if n < self.min_n:
            return [
                AuditFinding(
                    code="SAMPLE_TOO_SMALL",
                    severity=Severity.ERROR,
                    message=(
                        f"Sample size n={n} is below the absolute minimum ({self.min_n}). "
                        "No metric is meaningful at this scale."
                    ),
                    action=(
                        f"Collect at least {self.min_n} examples before reporting any results. "
                        "For accuracy estimates to be useful, aim for N≥200."
                    ),
                    details={"n": n, "minimum": self.min_n},
                )
            ]
        return []

    def _check_ci_precision(
        self,
        n: int,
        expected_accuracy: float,
        desired_half_width: float,
    ) -> list[AuditFinding]:
        result = self.power.for_ci_precision(
            desired_half_width=desired_half_width,
            expected_accuracy=expected_accuracy,
            observed_n=n,
        )
        required = result.minimum_n

        if n < required:
            z = scipy_stats.norm.ppf(0.975)
            p = expected_accuracy
            actual_hw = z * (p * (1 - p) / n) ** 0.5

            return [
                AuditFinding(
                    code="UNDERPOWERED_CI",
                    severity=Severity.WARNING,
                    message=(
                        f"Your sample size (n={n}) gives a CI half-width of ±{actual_hw:.3f} "
                        f"(±{actual_hw * 100:.1f}%), not the ±{desired_half_width:.3f} you may "
                        "be implying by reporting to two decimal places."
                    ),
                    action=(
                        f"To achieve ±{desired_half_width:.3f} precision, you need n≥{required}. "
                        "Either collect more examples or report results to one decimal place "
                        "with explicit CI bounds."
                    ),
                    details={"n": n, "required_n": required, "actual_half_width": actual_hw},
                )
            ]
        return []

    def _check_achieved_power(
        self,
        n: int,
        accuracy: float,
        effect_size: float = 0.05,
    ) -> list[AuditFinding]:
        """Check whether n was sufficient to detect a 5pp accuracy difference."""
        # Clamp accuracy to open interval (0, 1) - exactly 0 or 1 makes the
        # power formula degenerate (variance term = 0). These edge cases are
        # real (perfect or zero accuracy) but not meaningful for power analysis.
        accuracy_clamped = max(1e-4, min(1 - 1e-4, accuracy))
        result = self.power.for_proportion_difference(
            effect_size=effect_size,
            p1=accuracy_clamped,
            observed_n=n,
        )

        if result.achieved_power is not None and result.achieved_power < self.desired_power:
            return [
                AuditFinding(
                    code="UNDERPOWERED_COMPARISON",
                    severity=Severity.WARNING,
                    message=(
                        f"Your sample size (n={n}) achieves only {result.achieved_power:.0%} power "
                        f"to detect a {effect_size:.0%} accuracy difference. "
                        f"Target is {self.desired_power:.0%}. "
                        "A negative result here is inconclusive, not evidence of equivalence."
                    ),
                    action=(
                        f"To achieve {self.desired_power:.0%} power, "
                        f"you need n≥{result.minimum_n}. "
                        "If models appear similar in accuracy, increase N "
                        "before concluding equivalence."
                    ),
                    details={
                        "n": n,
                        "required_n": result.minimum_n,
                        "achieved_power": result.achieved_power,
                    },
                )
            ]
        return []

    def _check_class_imbalance(self, label_distribution: dict[str, int]) -> list[AuditFinding]:
        counts = np.array(list(label_distribution.values()), dtype=float)
        total = counts.sum()
        majority_frac = counts.max() / total
        majority_label = list(label_distribution.keys())[int(np.argmax(counts))]

        if majority_frac >= 0.90:
            return [
                AuditFinding(
                    code="SEVERE_CLASS_IMBALANCE",
                    severity=Severity.ERROR,
                    message=(
                        f"Your test set is {majority_frac:.0%} class '{majority_label}'. "
                        "A trivial model predicting the majority class achieves "
                        f"{majority_frac:.0%} accuracy. Accuracy is a meaningless metric here."
                    ),
                    action=(
                        "Report balanced accuracy, macro-F1, or AUC instead of accuracy. "
                        "Consider stratified sampling to balance the test set."
                    ),
                    details={"distribution": label_distribution, "majority_frac": majority_frac},
                )
            ]
        elif majority_frac >= 0.75:
            return [
                AuditFinding(
                    code="CLASS_IMBALANCE",
                    severity=Severity.WARNING,
                    message=(
                        f"Your test set is {majority_frac:.0%} class '{majority_label}'. "
                        "Accuracy will be inflated relative to minority-class performance."
                    ),
                    action=(
                        "Report macro-F1 alongside accuracy, and note the class distribution "
                        "when presenting results."
                    ),
                    details={"distribution": label_distribution, "majority_frac": majority_frac},
                )
            ]
        return []

    def _check_multiple_testing_design(self, n_variants: int) -> list[AuditFinding]:
        expected_fp = n_variants * 0.05
        prob_at_least_one_fp = 1 - (1 - 0.05) ** n_variants

        return [
            AuditFinding(
                code="MULTIPLE_TESTING_RISK",
                severity=Severity.WARNING,
                message=(
                    f"You are comparing {n_variants} variants. At α=0.05, the probability "
                    f"of at least one false positive is {prob_at_least_one_fp:.0%}, "
                    f"and you'd expect {expected_fp:.1f} false discoveries under H0."
                ),
                action=(
                    "Apply Benjamini-Hochberg FDR correction to all p-values before reporting. "
                    "Use evalkit.metrics.BHCorrection. Always report adjusted p-values."
                ),
                details={"n_variants": n_variants, "expected_fp": expected_fp},
            )
        ]

    def _check_multiple_testing_results(
        self, n_variants: int, p_values: list[float]
    ) -> list[AuditFinding]:
        bh = BHCorrection(alpha=0.05)
        result = bh.correct(p_values)

        findings = []
        if result.false_positive_warning:
            n_changed = sum(
                1
                for raw, adj_reject in zip(result.unadjusted_p_values, result.reject_null)
                if raw < 0.05 and not adj_reject
            )
            findings.append(
                AuditFinding(
                    code="MULTIPLE_TESTING_UNCORRECTED",
                    severity=Severity.ERROR,
                    message=(
                        f"{n_changed} comparison(s) appear significant without FDR correction "
                        "but are NOT significant after Benjamini-Hochberg correction. "
                        "Reporting uncorrected results would be misleading."
                    ),
                    action=(
                        "Report only BH-adjusted p-values. "
                        "The apparent significant results are likely false positives."
                    ),
                    details={"n_changed": n_changed, "adjusted_pvalues": result.adjusted_p_values},
                )
            )

        return findings

    def _check_judge_agreement(self, kappa: float) -> list[AuditFinding]:
        if kappa < self.min_judge_agreement:
            return [
                AuditFinding(
                    code="LOW_JUDGE_AGREEMENT",
                    severity=Severity.ERROR,
                    message=(
                        f"LLM-as-judge agreement (κ={kappa:.2f}) is below the minimum "
                        f"threshold (κ={self.min_judge_agreement:.2f}). "
                        "Your evaluation scores are unreliable."
                    ),
                    action=(
                        "Improve your judge prompt with clearer rubrics and examples. "
                        "Consider using an ensemble of judges and taking the majority vote. "
                        "Re-validate agreement before reporting results."
                    ),
                    details={"kappa": kappa, "threshold": self.min_judge_agreement},
                )
            ]
        return []
