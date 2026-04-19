"""
Integration tests for Experiment.

These verify that the full pipeline - dataset → runner → metrics → audit -
works correctly end-to-end, and that the RigorChecker fires at the right times.
"""

import pytest

from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.experiment import Experiment, ExperimentResult
from evalkit.core.judge import ExactMatchJudge
from evalkit.core.runner import MockRunner


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
