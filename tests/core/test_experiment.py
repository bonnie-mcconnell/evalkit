"""
Integration tests for Experiment.

These verify that the full pipeline - dataset → runner → metrics → audit -
works correctly end-to-end, and that the RigorChecker fires at the right times.
"""

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
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template)
    result = Experiment("test", tiny_dataset, runner).run()
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
