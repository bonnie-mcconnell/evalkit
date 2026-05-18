"""
Integration tests for Experiment.

These verify that the full pipeline - dataset → runner → metrics → audit -
works correctly end-to-end, and that the RigorChecker fires at the right times.
"""

from pathlib import Path

import pytest

from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.experiment import Experiment, ExperimentResult
from evalkit.core.judge import ExactMatchJudge
from evalkit.core.runner import ExampleResult, MockRunner, RunResult
from evalkit.metrics.accuracy import BalancedAccuracy, F1Score


@pytest.fixture
def adequate_dataset():
    """300 examples - adequately powered by any reasonable standard."""
    records = [{"id": str(i), "question": f"q{i}", "label": f"answer_{i % 10}"} for i in range(300)]
    return EvalDataset.from_list(records, name="adequate_ds")


@pytest.fixture
def tiny_dataset():
    """10 examples - deliberately underpowered."""
    records = [{"id": str(i), "question": f"q{i}", "label": "yes"} for i in range(10)]
    return EvalDataset.from_list(records, name="tiny_ds")


@pytest.fixture
def runner(adequate_dataset):
    template = PromptTemplate("{{ question }}")
    return MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)


@pytest.fixture
def simple_result(adequate_dataset, runner):
    """A complete ExperimentResult on adequate_dataset with default runner."""
    return Experiment("simple_test", adequate_dataset, runner, n_resamples=500).run()


def test_experiment_returns_experiment_result(adequate_dataset, runner):
    exp = Experiment("test", adequate_dataset, runner)
    result = exp.run()
    assert isinstance(result, ExperimentResult)


def test_experiment_metrics_contain_accuracy(adequate_dataset, runner):
    result = Experiment("test", adequate_dataset, runner).run()
    assert "Accuracy" in result.metrics


def test_experiment_accuracy_has_valid_ci(adequate_dataset, runner):
    result = Experiment("test", adequate_dataset, runner).run()
    acc = result.metrics["Accuracy"]
    assert 0.0 <= acc.ci_lower <= acc.value <= acc.ci_upper <= 1.0
    assert acc.n == 300


def test_experiment_accuracy_close_to_mock_accuracy(adequate_dataset):
    """Observed accuracy should be within 10% of the mock accuracy parameter."""
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)
    result = Experiment("test", adequate_dataset, runner).run()
    acc = result.metrics["Accuracy"].value
    assert 0.65 <= acc <= 0.95, f"Expected ~0.80, got {acc:.3f}"


def test_experiment_posthoc_audit_present(adequate_dataset, runner):
    result = Experiment("test", adequate_dataset, runner).run()
    assert result.posthoc_audit is not None


def test_experiment_adequate_dataset_passes_audit(adequate_dataset):
    """A well-powered experiment with balanced labels should pass the RigorChecker."""
    template = PromptTemplate("{{ question }}")
    # Balanced dataset: half 'a', half 'b'
    records = [
        {"id": str(i), "question": f"q{i}", "label": "a" if i % 2 == 0 else "b"} for i in range(300)
    ]
    ds = EvalDataset.from_list(records, name="balanced")
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)
    result = Experiment("test", ds, runner).run()
    assert result.posthoc_audit.passed


def test_experiment_tiny_dataset_fails_audit(tiny_dataset):
    """
    n=10 triggers SAMPLE_TOO_SMALL. With strict=True (default), this raises
    PreFlightError before the run. With strict=False, the post-hoc audit flags it.
    This test verifies both behaviours.
    """
    from evalkit import PreFlightError

    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template)

    # Default: strict=True → PreFlightError raised before any runner calls
    with pytest.raises(PreFlightError) as exc_info:
        Experiment("test", tiny_dataset, runner).run()
    assert "SAMPLE_TOO_SMALL" in str(exc_info.value)
    assert exc_info.value.audit is not None
    assert any(f.code == "SAMPLE_TOO_SMALL" for f in exc_info.value.audit.findings)

    # strict=False → runs, post-hoc audit catches it
    result = Experiment("test", tiny_dataset, runner, strict=False).run()
    assert not result.posthoc_audit.passed
    codes = [f.code for f in result.posthoc_audit.findings]
    assert "SAMPLE_TOO_SMALL" in codes


def test_experiment_imbalanced_dataset_triggers_imbalance_warning():
    """91% of one class should trigger class imbalance finding."""
    records = [{"id": str(i), "label": "yes"} for i in range(91)]
    records += [{"id": str(i + 91), "label": "no"} for i in range(9)]
    # Add required 'question' field
    for r in records:
        r["question"] = "q"
    ds = EvalDataset.from_list(records, name="imbalanced")
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template)
    result = Experiment("test", ds, runner).run()
    codes = [f.code for f in result.posthoc_audit.findings]
    assert "SEVERE_CLASS_IMBALANCE" in codes


def test_experiment_additional_metric_is_included(adequate_dataset):
    """Additional metrics passed in should appear in results."""
    from evalkit.metrics.accuracy import Accuracy

    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)

    class DoubleAccuracy(Accuracy):
        @property
        def name(self) -> str:
            return "DoubleAccuracy"

    result = Experiment(
        "test", adequate_dataset, runner, additional_metrics=[DoubleAccuracy()]
    ).run()
    assert "DoubleAccuracy" in result.metrics


def test_experiment_run_result_model_set(adequate_dataset, runner):
    result = Experiment("test", adequate_dataset, runner).run()
    assert result.run_result.model == "mock-model-v1"


def test_experiment_name_appears_in_audit(adequate_dataset, runner):
    result = Experiment("my_special_experiment", adequate_dataset, runner).run()
    assert result.posthoc_audit.experiment_name == "my_special_experiment"


def test_experiment_print_summary_runs_without_error(adequate_dataset, runner, capsys):
    result = Experiment("test", adequate_dataset, runner).run()
    result.print_summary()  # Should not raise
    captured = capsys.readouterr()
    assert "Accuracy" in captured.out


# ── v0.2.0: additional_metrics receive outputs and references ──────────────────


def _make_biased_run_result(
    majority_label: str,
    minority_label: str,
    n_majority: int,
    n_minority: int,
    majority_error_rate: float,
    minority_error_rate: float,
) -> RunResult:
    """
    Build a RunResult where errors are class-biased - the model makes far more
    mistakes on the minority class than on the majority class.  This gives
    Accuracy, BalancedAccuracy, and F1Score meaningfully different values.

    Majority examples: output = majority_label (correct) or "WRONG" (incorrect).
    Minority examples: output = minority_label (correct) or "WRONG" (incorrect).
    """
    judge = ExactMatchJudge()
    results = []
    seed = 0

    def _make_example(label: str, error: bool, idx: int) -> ExampleResult:
        output = "WRONG" if error else label
        judgment = judge.judge(output, label)
        return ExampleResult(
            example_id=str(idx),
            prompt=f"q{idx}",
            output=output,
            reference=label,
            judgment=judgment,
            latency_ms=0.0,
        )

    for i in range(n_majority):
        error = (i / n_majority) < majority_error_rate
        results.append(_make_example(majority_label, error, seed))
        seed += 1

    for i in range(n_minority):
        error = (i / n_minority) < minority_error_rate
        results.append(_make_example(minority_label, error, seed))
        seed += 1

    return RunResult(
        example_results=results,
        model="mock-biased",
        dataset_name="biased-test",
    )


def test_balanced_accuracy_differs_from_accuracy_on_imbalanced_data():
    """
    On genuinely imbalanced data where the minority class has a much higher
    error rate, BalancedAccuracy must differ from Accuracy.

    Setup:
      - 180 majority examples, 10% error rate  → 162 correct
      - 20 minority examples, 80% error rate   → 4 correct
      - Overall Accuracy ≈ (162 + 4) / 200 = 0.83
      - BalancedAccuracy = mean(0.90, 0.20) = 0.55

    If the fix is NOT in place, both would equal ~0.83 (the overall accuracy).
    """
    run_result = _make_biased_run_result(
        majority_label="yes",
        minority_label="no",
        n_majority=180,
        n_minority=20,
        majority_error_rate=0.10,
        minority_error_rate=0.80,
    )

    class _FixedRunner:
        judge = ExactMatchJudge()

        def run(self, dataset: EvalDataset) -> RunResult:
            return run_result

    records = [{"id": str(i), "question": f"q{i}", "label": "yes"} for i in range(200)]
    ds = EvalDataset.from_list(records, name="biased")

    result = Experiment(
        "biased_test",
        ds,
        _FixedRunner(),
        additional_metrics=[BalancedAccuracy(n_resamples=500)],
        n_resamples=500,
    ).run()

    acc = result.metrics["Accuracy"].value
    bal_acc = result.metrics["BalancedAccuracy"].value

    assert 0.75 <= acc <= 0.90, f"Accuracy={acc:.3f} out of expected range"
    assert bal_acc < acc - 0.10, (
        f"BalancedAccuracy ({bal_acc:.3f}) should be substantially below "
        f"Accuracy ({acc:.3f}) on class-biased data. "
        "Fix may not be applied - additional_metrics may still receive binary correct array."
    )


def test_f1score_differs_from_accuracy_on_imbalanced_data():
    """
    On imbalanced data with class-biased errors, macro F1 must differ from Accuracy.

    Same setup as test_balanced_accuracy_differs_from_accuracy_on_imbalanced_data.
    F1(macro) should also be substantially lower than overall Accuracy because
    the minority class has very low recall.
    """
    run_result = _make_biased_run_result(
        majority_label="yes",
        minority_label="no",
        n_majority=180,
        n_minority=20,
        majority_error_rate=0.10,
        minority_error_rate=0.80,
    )

    class _FixedRunner:
        judge = ExactMatchJudge()

        def run(self, dataset: EvalDataset) -> RunResult:
            return run_result

    records = [{"id": str(i), "question": f"q{i}", "label": "yes"} for i in range(200)]
    ds = EvalDataset.from_list(records, name="biased_f1")

    result = Experiment(
        "biased_f1_test",
        ds,
        _FixedRunner(),
        additional_metrics=[F1Score(average="macro", n_resamples=500)],
        n_resamples=500,
    ).run()

    acc = result.metrics["Accuracy"].value
    f1_key = "F1Score(macro)"
    assert f1_key in result.metrics, f"Expected '{f1_key}' in metrics, got {list(result.metrics)}"

    f1 = result.metrics[f1_key].value

    assert f1 < acc - 0.10, (
        f"F1Score(macro) ({f1:.3f}) should be substantially below "
        f"Accuracy ({acc:.3f}) on class-biased data. "
        "Fix may not be applied - additional_metrics may still receive binary correct array."
    )


# ── repr and save/report methods ───────────────────────────────────────────────


def test_experiment_result_repr_is_concise(simple_result: ExperimentResult) -> None:
    """__repr__ should be a one-liner, not the full dataclass dump."""
    r = repr(simple_result)
    assert "ExperimentResult(" in r
    assert "accuracy=" in r
    assert "audit=" in r
    # Crucially: should NOT be hundreds of characters showing full nested structure
    assert len(r) < 200, f"repr too verbose ({len(r)} chars): {r}"


def test_experiment_result_repr_contains_audit_status(simple_result: ExperimentResult) -> None:
    r = repr(simple_result)
    assert "PASS" in r or "FAIL" in r


def test_metric_result_repr_matches_str(simple_result: ExperimentResult) -> None:
    acc = simple_result.metrics["Accuracy"]
    assert repr(acc) == f"MetricResult({acc})"


def test_eval_dataset_repr_informative() -> None:
    from evalkit import EvalDataset

    ds = EvalDataset.from_list(
        [{"id": str(i), "q": "Q", "label": "pos" if i < 7 else "neg"} for i in range(10)],
        reference_field="label",
    )
    r = repr(ds)
    assert "EvalDataset(" in r
    assert "n=10" in r
    assert "neg" in r
    assert "pos" in r


def test_experiment_result_save_and_reload(simple_result: ExperimentResult, tmp_path: Path) -> None:
    """save() writes a JSON file that can be loaded and used for comparison."""
    import json

    out = tmp_path / "result.json"
    returned_path = simple_result.save(out)

    assert returned_path == out
    assert out.exists()

    data = json.loads(out.read_text())
    assert data["status"] == "complete"
    assert data["experiment_name"] == simple_result.experiment_name
    assert "accuracy" in data
    assert "metrics" in data
    assert "example_ids" in data
    assert "correct" in data
    assert "audit_passed" in data
    assert isinstance(data["audit_findings"], list)


def test_experiment_result_save_creates_parent_dirs(
    simple_result: ExperimentResult, tmp_path: Path
) -> None:
    """save() should create parent directories if they don't exist."""
    out = tmp_path / "deep" / "nested" / "result.json"
    simple_result.save(out)
    assert out.exists()


def test_experiment_result_generate_report_returns_html(
    simple_result: ExperimentResult,
) -> None:
    """generate_report() with no path should return an HTML string."""
    html = simple_result.generate_report()
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert simple_result.experiment_name in html


def test_experiment_result_generate_report_writes_file(
    simple_result: ExperimentResult, tmp_path: Path
) -> None:
    """generate_report(path) should write the file and return the path."""
    out = tmp_path / "report.html"
    result = simple_result.generate_report(out)
    assert result == str(out)
    assert out.exists()
    assert "<!DOCTYPE html>" in out.read_text()


def test_sample_size_table_ci_mode_single_column() -> None:
    """CI mode should show a single-column table, not repeated identical columns."""
    from evalkit import PowerAnalysis

    pa = PowerAnalysis()
    table = pa.sample_size_table(
        effect_sizes=[0.05, 0.10],
        test="ci",
        print_table=False,
    )
    # Should mention half-width, not power columns
    assert "±5%" in table
    assert "±10%" in table
    # Should NOT have multiple power level headers for CI mode
    assert "Power 70%" not in table
    assert "Power 90%" not in table


def test_comparison_result_save(simple_result: ExperimentResult, tmp_path: Path) -> None:
    """ComparisonResult.save() writes JSON with all comparison fields."""
    import json

    comparison = simple_result.compare(simple_result)
    out = tmp_path / "comparison.json"
    returned = comparison.save(out)

    assert returned == out
    assert out.exists()

    data = json.loads(out.read_text())
    assert "experiment_a" in data
    assert "experiment_b" in data
    assert "p_value" in data
    assert "reject_null" in data
    assert "accuracy_a" in data
    assert "accuracy_b" in data


def test_comparison_result_save_creates_parents(
    simple_result: ExperimentResult, tmp_path: Path
) -> None:
    """ComparisonResult.save() creates parent directories automatically."""
    comparison = simple_result.compare(simple_result)
    out = tmp_path / "deep" / "nested" / "comparison.json"
    comparison.save(out)
    assert out.exists()


def test_audit_comparisons_empty_when_no_comparisons_made(runner, adequate_dataset):
    """audit_comparisons() returns clean pass when no .compare() calls made."""
    from evalkit.analysis.rigour import AuditReport

    result = Experiment("control", adequate_dataset, runner, strict=False, n_resamples=500).run()
    audit = result.audit_comparisons()
    assert isinstance(audit, AuditReport)
    assert audit.passed
    assert len(audit.findings) == 0


def test_audit_comparisons_after_compare_calls(runner, adequate_dataset):
    """audit_comparisons() with exactly one comparison returns INFO (not applicable).

    A single comparison has no family-wise error rate issue so BH correction is not
    needed. The audit returns an INFO finding explicitly communicating this rather
    than a clean empty pass - which would be ambiguous between "checked and passed"
    and "nothing to check".
    """
    from evalkit.analysis.rigour import AuditReport, Severity

    result_a = Experiment("exp_a", adequate_dataset, runner, strict=False, n_resamples=500).run()
    result_b = Experiment("exp_b", adequate_dataset, runner, strict=False, n_resamples=500).run()
    result_a.compare(result_b)

    # After one comparison, _comparison_p_values should have 1 entry
    assert len(result_a._comparison_p_values) == 1
    assert len(result_b._comparison_p_values) == 1

    audit = result_a.audit_comparisons()
    assert isinstance(audit, AuditReport)
    # Single comparison → INFO finding explaining BH was not applicable
    assert audit.passed  # INFO does not fail the audit
    assert len(audit.findings) == 1
    assert audit.findings[0].severity == Severity.INFO
    assert audit.findings[0].code == "MULTIPLE_TESTING_NOT_APPLICABLE"


def test_experiment_expected_accuracy_propagates_to_preflight(runner, adequate_dataset):
    """expected_accuracy is forwarded to the pre_flight CI precision check.

    This ensures the pre-flight N recommendation is calibrated to the user's
    actual model expectations rather than always assuming p=0.70. A model
    expected to achieve p=0.95 requires far fewer examples for the same CI
    width than p=0.70, so using 0.70 for a high-accuracy model would give
    an overly conservative (too large) N requirement.

    We verify that passing expected_accuracy=0.95 stores it correctly, meaning
    the pre_flight call receives it. We also verify default is 0.70.
    """
    exp_default = Experiment("e1", adequate_dataset, runner, strict=False)
    assert exp_default.expected_accuracy == 0.70

    exp_custom = Experiment("e2", adequate_dataset, runner, strict=False, expected_accuracy=0.95)
    assert exp_custom.expected_accuracy == 0.95

    # Both should run without error - the parameter only affects pre-flight CI
    # warning thresholds, not whether the experiment executes.
    result = exp_custom.run()
    assert "Accuracy" in result.metrics


def test_audit_comparisons_multiple_comparisons_runs_bh(runner, adequate_dataset):
    """audit_comparisons() with ≥2 comparisons runs the BH-FDR check.

    This exercises the code path where len(p_vals) >= 2, which calls
    RigorChecker.audit() with the collected p-values. We need this path
    covered because it is the primary purpose of audit_comparisons() -
    the single-comparison INFO path is the degenerate case.
    """
    from evalkit.analysis.rigour import AuditReport

    result_a = Experiment("exp_a", adequate_dataset, runner, strict=False, n_resamples=500).run()
    result_b = Experiment("exp_b", adequate_dataset, runner, strict=False, n_resamples=500).run()
    result_c = Experiment("exp_c", adequate_dataset, runner, strict=False, n_resamples=500).run()

    # Two comparisons against result_a - this fills _comparison_p_values with 2 entries
    result_a.compare(result_b)
    result_a.compare(result_c)

    assert len(result_a._comparison_p_values) == 2

    audit = result_a.audit_comparisons()
    assert isinstance(audit, AuditReport)
    # With adequate dataset and only 2 comparisons, audit should pass
    # (no MULTIPLE_TESTING_NOT_APPLICABLE INFO - that is the single-comparison path)
    assert audit.passed
    # The multi-comparison path does not return the NOT_APPLICABLE finding
    not_applicable = [f for f in audit.findings if f.code == "MULTIPLE_TESTING_NOT_APPLICABLE"]
    assert len(not_applicable) == 0
