"""
Tests for RigorChecker.

These tests verify that the right findings are raised in the right situations.
The invariant: RigorChecker must never silently pass a bad experiment.
"""

import pytest

from evalkit.analysis.rigour import RigorChecker, Severity


@pytest.fixture
def checker():
    return RigorChecker()


# ── Sample size checks ─────────────────────────────────────────────────────────


def test_raises_error_on_tiny_sample(checker):
    report = checker.audit(n_examples=5)
    codes = [f.code for f in report.findings]
    assert "SAMPLE_TOO_SMALL" in codes
    assert not report.passed


def test_raises_error_on_tiny_sample_even_with_accuracy(checker):
    """n=5 should be flagged as SAMPLE_TOO_SMALL even when accuracy is given."""
    report = checker.audit(n_examples=5, accuracy=0.80)
    codes = [f.code for f in report.findings]
    assert "SAMPLE_TOO_SMALL" in codes
    assert not report.passed


def test_no_error_for_adequate_sample(checker):
    report = checker.audit(n_examples=200, accuracy=0.75)
    sample_errors = [f for f in report.findings if f.code == "SAMPLE_TOO_SMALL"]
    assert not sample_errors


def test_underpowered_comparison_warning(checker):
    """n=50 is underpowered to detect a 5% accuracy difference."""
    report = checker.audit(n_examples=50, accuracy=0.75)
    codes = [f.code for f in report.findings]
    assert "UNDERPOWERED_COMPARISON" in codes or "UNDERPOWERED_CI" in codes


# ── Class imbalance checks ─────────────────────────────────────────────────────


def test_severe_imbalance_raises_error(checker):
    dist = {"positive": 950, "negative": 50}
    report = checker.audit(n_examples=1000, label_distribution=dist)
    codes = [f.code for f in report.findings]
    assert "SEVERE_CLASS_IMBALANCE" in codes
    assert not report.passed


def test_moderate_imbalance_raises_warning(checker):
    dist = {"positive": 800, "negative": 200}
    report = checker.audit(n_examples=1000, label_distribution=dist)
    codes = [f.code for f in report.findings]
    assert "CLASS_IMBALANCE" in codes
    # Warning, not error - experiment can proceed
    imbalance_findings = [f for f in report.findings if f.code == "CLASS_IMBALANCE"]
    assert all(f.severity == Severity.WARNING for f in imbalance_findings)


def test_balanced_dataset_no_imbalance_warning(checker):
    dist = {"positive": 100, "negative": 100}
    report = checker.audit(n_examples=200, label_distribution=dist)
    codes = [f.code for f in report.findings]
    assert "CLASS_IMBALANCE" not in codes
    assert "SEVERE_CLASS_IMBALANCE" not in codes


# ── Multiple testing checks ────────────────────────────────────────────────────


def test_multiple_testing_warning_in_preflight(checker):
    report = checker.pre_flight(n_examples=200, n_variants=8)
    codes = [f.code for f in report.findings]
    assert "MULTIPLE_TESTING_RISK" in codes


def test_single_variant_no_multiple_testing_warning(checker):
    report = checker.pre_flight(n_examples=200, n_variants=1)
    codes = [f.code for f in report.findings]
    assert "MULTIPLE_TESTING_RISK" not in codes


def test_uncorrected_false_positives_raises_error(checker):
    """p=0.04 with k=10 is a false positive after BH correction."""
    p_values = [0.04] + [0.80] * 9
    report = checker.audit(n_examples=200, n_variants=10, p_values=p_values)
    codes = [f.code for f in report.findings]
    assert "MULTIPLE_TESTING_UNCORRECTED" in codes
    assert not report.passed


def test_corrected_significant_result_no_error(checker):
    """Very small p-values should survive BH correction without error."""
    p_values = [0.0001, 0.0002, 0.80, 0.90]
    report = checker.audit(n_examples=200, n_variants=4, p_values=p_values)
    uncorr = [f for f in report.findings if f.code == "MULTIPLE_TESTING_UNCORRECTED"]
    assert not uncorr


# ── Judge agreement checks ─────────────────────────────────────────────────────


def test_low_judge_agreement_raises_error(checker):
    report = checker.audit(n_examples=200, judge_kappa=0.35)
    codes = [f.code for f in report.findings]
    assert "LOW_JUDGE_AGREEMENT" in codes
    assert not report.passed


def test_acceptable_judge_agreement_no_error(checker):
    report = checker.audit(n_examples=200, judge_kappa=0.75)
    codes = [f.code for f in report.findings]
    assert "LOW_JUDGE_AGREEMENT" not in codes


# ── Audit report properties ────────────────────────────────────────────────────


def test_clean_experiment_passes(checker):
    report = checker.audit(
        n_examples=300,
        accuracy=0.75,
        label_distribution={"pos": 150, "neg": 150},
        n_variants=1,
        judge_kappa=0.72,
    )
    assert report.passed


def test_audit_report_str_contains_status():
    checker = RigorChecker()
    report = checker.audit(n_examples=5)
    s = str(report)
    assert "FAIL" in s or "PASS" in s
    assert "RigorChecker" in s


def test_audit_report_sort_order_errors_before_warnings():
    """Errors must appear before warnings in the report string."""
    from evalkit.analysis.rigour import AuditFinding, AuditReport, Severity

    findings = [
        AuditFinding("WARN_CODE", Severity.WARNING, "a warning", "fix it"),
        AuditFinding("INFO_CODE", Severity.INFO, "an info", "consider it"),
        AuditFinding("ERR_CODE", Severity.ERROR, "an error", "fix this first"),
    ]
    report = AuditReport(findings=findings, experiment_name="test")
    s = str(report)
    err_pos = s.index("ERR_CODE")
    warn_pos = s.index("WARN_CODE")
    info_pos = s.index("INFO_CODE")
    assert err_pos < warn_pos < info_pos, (
        "Expected errors before warnings before info in report output"
    )


def test_preflight_llm_judge_gives_info(checker):
    report = checker.pre_flight(n_examples=200, judge_type="llm")
    codes = [f.code for f in report.findings]
    assert "JUDGE_AGREEMENT_REQUIRED" in codes


def test_preflight_deterministic_judge_no_agreement_notice(checker):
    report = checker.pre_flight(n_examples=200, judge_type="deterministic")
    codes = [f.code for f in report.findings]
    assert "JUDGE_AGREEMENT_REQUIRED" not in codes


# ── Input validation ────────────────────────────────────────────────────────────


def test_audit_invalid_accuracy_raises(checker):
    with pytest.raises(ValueError, match="accuracy"):
        checker.audit(n_examples=200, accuracy=1.5)


def test_audit_invalid_accuracy_negative_raises(checker):
    with pytest.raises(ValueError, match="accuracy"):
        checker.audit(n_examples=200, accuracy=-0.1)


def test_audit_invalid_judge_kappa_raises(checker):
    with pytest.raises(ValueError, match="judge_kappa"):
        checker.audit(n_examples=200, judge_kappa=1.5)


def test_audit_negative_n_examples_raises(checker):
    with pytest.raises(ValueError, match="n_examples"):
        checker.audit(n_examples=-1)


def test_preflight_raises_on_negative_n(checker):
    """n_examples < 0 is a programming error - should raise immediately."""
    with pytest.raises(ValueError):
        checker.pre_flight(n_examples=-1)


def test_multiple_testing_no_pvalues_warning(checker):
    """
    n_variants > 1 without p_values triggers MULTIPLE_TESTING_NO_PVALUES warning.
    The user is comparing variants but hasn't provided their p-values for correction.
    """
    report = checker.audit(n_examples=300, accuracy=0.75, n_variants=5)
    codes = [f.code for f in report.findings]
    assert "MULTIPLE_TESTING_NO_PVALUES" in codes


def test_ci_precision_check_passes_when_n_adequate(checker):
    """
    _check_ci_precision returns [] when n is sufficient.
    This exercises the 'return []' branch at line 352 in rigour.py.
    With n=1000 and desired_half_width=0.05 (requires ~300), we expect no finding.
    """
    report = checker.audit(n_examples=1000, accuracy=0.75)
    ci_findings = [f for f in report.findings if f.code == "UNDERPOWERED_CI"]
    assert len(ci_findings) == 0


def test_achieved_power_check_passes_when_adequate(checker):
    """
    _check_achieved_power returns [] when N is large enough.
    Line 392 in rigour.py: the 'return []' branch when power is adequate.
    n=2000 at accuracy=0.75 should have adequate power for 5pp detection.
    """
    report = checker.audit(n_examples=2000, accuracy=0.75)
    power_findings = [f for f in report.findings if f.code == "UNDERPOWERED_COMPARISON"]
    assert len(power_findings) == 0
