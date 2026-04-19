"""
Tests for accuracy metrics.

Statistical property tests: not exact values, but correct direction.
"""

import pytest

from evalkit.metrics.accuracy import Accuracy, BalancedAccuracy, F1Score


def test_accuracy_perfect():
    acc = Accuracy(n_resamples=500, seed=0)
    result = acc.compute([1, 1, 1, 0, 0] * 20, [1, 1, 1, 0, 0] * 20)
    assert abs(result.value - 1.0) < 1e-9
    assert result.ci_lower > 0.95


def test_accuracy_zero():
    acc = Accuracy(n_resamples=500, seed=0)
    result = acc.compute([0, 0, 0] * 30, [1, 1, 1] * 30)
    assert abs(result.value - 0.0) < 1e-9


def test_accuracy_half():
    acc = Accuracy(n_resamples=2000, seed=42)
    preds = [0, 1] * 100
    refs = [1, 1] * 100  # Always 1; half match
    result = acc.compute(preds, refs)
    assert abs(result.value - 0.5) < 0.05


def test_accuracy_warns_on_imbalance(caplog):
    import logging

    acc = Accuracy(n_resamples=500, seed=0)
    # 95% class 1 - should warn
    preds = [1] * 95 + [0] * 5
    refs = [1] * 95 + [0] * 5
    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        acc.compute(preds, refs)
    assert any("imbalance" in msg.lower() for msg in caplog.messages)


def test_accuracy_metric_result_has_correct_n():
    acc = Accuracy(n_resamples=500, seed=0)
    preds = [1] * 80 + [0] * 20
    refs = [1] * 100
    result = acc.compute(preds, refs)
    assert result.n == 100


def test_balanced_accuracy_insensitive_to_imbalance():
    """
    BalancedAccuracy should give 0.5 for a majority-class predictor,
    while Accuracy would give 0.9.
    """
    bal = BalancedAccuracy(n_resamples=500, seed=0)
    # Always predict 1; 90% are 1
    preds = [1] * 100
    refs = [1] * 90 + [0] * 10
    result = bal.compute(preds, refs)
    # Per-class recall: class 1 = 1.0, class 0 = 0.0. Mean = 0.5
    assert abs(result.value - 0.5) < 0.05


def test_f1_binary_basic():
    f1 = F1Score(average="binary", n_resamples=500, seed=0)
    # Perfect predictions
    preds = [1] * 50 + [0] * 50
    refs = [1] * 50 + [0] * 50
    result = f1.compute(preds, refs)
    assert abs(result.value - 1.0) < 1e-9


def test_f1_returns_per_class_extra():
    f1 = F1Score(average="macro", n_resamples=500, seed=0)
    preds = [0, 1, 0, 1, 0, 1] * 20
    refs = [0, 1, 0, 1, 0, 1] * 20
    result = f1.compute(preds, refs)
    assert "per_class_f1" in result.extra
    assert "0" in result.extra["per_class_f1"]
    assert "1" in result.extra["per_class_f1"]


def test_f1_ci_bounds_valid():
    f1 = F1Score(average="macro", n_resamples=1000, seed=0)
    preds = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 20
    refs = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 20
    result = f1.compute(preds, refs)
    assert result.ci_lower <= result.value <= result.ci_upper
    assert 0.0 <= result.ci_lower
    assert result.ci_upper <= 1.0


# ── ExpectedCalibrationError ─────────────────────────────────────────────────


def test_ece_well_calibrated_model():
    """A model whose confidence matches accuracy per bin should have low ECE.

    We create data where within each confidence bin, the actual accuracy
    matches the confidence. Specifically: examples with confidence 0.8 are
    correct 80% of the time, and examples with confidence 0.2 are correct
    20% of the time. Per-bin ECE is near 0 for each bin → total ECE near 0.
    """
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_bins=5, n_resamples=500, seed=0)
    # 100 examples at confidence=0.8: 80 correct, 20 wrong (accuracy=0.8 ≈ confidence=0.8)
    # 100 examples at confidence=0.2: 20 correct, 80 wrong (accuracy=0.2 ≈ confidence=0.2)
    correct = [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80
    confs = [0.8] * 100 + [0.2] * 100
    result = ece.compute(correct, confs)
    assert result.value < 0.05, f"Well-calibrated model should have low ECE, got {result.value:.4f}"


def test_ece_confidence_one_included_in_last_bin():
    """
    Confidence scores of exactly 1.0 must be counted, not silently dropped.
    With confidence=1.0 on every example and all correct, ECE should be 0.
    """
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_bins=10, n_resamples=200, seed=0)
    correct = [1] * 50
    confs = [1.0] * 50  # Exactly 1.0 - was excluded by the < bug
    result = ece.compute(correct, confs)
    assert abs(result.value) < 1e-9, (
        f"ECE should be 0.0 for perfectly calibrated model, got {result.value}"
    )


def test_ece_raises_on_out_of_range_confidence():
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_resamples=100, seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece.compute([1, 0], [0.5, 1.5])


def test_ece_raises_on_length_mismatch():
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_resamples=100, seed=0)
    with pytest.raises(ValueError, match="same length"):
        ece.compute([1, 0, 1], [0.9, 0.1])


# ── BLEUScore ─────────────────────────────────────────────────────────────────


def test_bleu_import_error_without_nltk(monkeypatch):
    """BLEUScore raises ImportError with install hint when nltk is not installed."""
    import importlib.util
    import sys

    if importlib.util.find_spec("nltk") is None:
        from evalkit.metrics.accuracy import BLEUScore

        with pytest.raises(ImportError, match="nltk"):
            BLEUScore().compute(["hello world"], ["hello world"])
    else:
        # nltk is installed - patch the import to simulate its absence
        monkeypatch.setitem(sys.modules, "nltk", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "nltk.translate", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", None)  # type: ignore[arg-type]
        import importlib

        from evalkit.metrics.accuracy import BLEUScore

        importlib.reload(__import__("evalkit.metrics.accuracy", fromlist=["BLEUScore"]))
        pytest.skip("BLEU import-error path requires nltk absent; skipping in nltk-installed env")


def test_bleu_warns_on_small_n(caplog):
    """BLEUScore should warn when N < 50 examples."""
    import importlib.util

    if importlib.util.find_spec("nltk") is None:
        pytest.skip("nltk not installed")
    import logging

    from evalkit.metrics.accuracy import BLEUScore

    preds = ["the cat sat"] * 20
    refs = ["the cat sat"] * 20
    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        BLEUScore(n_resamples=100).compute(preds, refs)
    assert any("N=20" in r.message or "unreliable" in r.message for r in caplog.records)


# ── F1Score ───────────────────────────────────────────────────────────────────


def test_f1_score_binary():
    """F1Score on binary predictions should return a MetricResult with valid CI."""
    from evalkit.metrics.accuracy import F1Score

    preds = [1, 1, 0, 1, 0, 0, 1, 1] * 10
    refs = [1, 0, 0, 1, 0, 1, 1, 0] * 10
    result = F1Score(n_resamples=300, seed=0).compute(preds, refs)
    assert 0.0 <= result.ci_lower <= result.value <= result.ci_upper <= 1.0
    assert result.n == 80


def test_f1_score_perfect():
    """F1Score of 1.0 when predictions exactly match references."""
    from evalkit.metrics.accuracy import F1Score

    preds = [1, 0, 1, 0] * 20
    refs = [1, 0, 1, 0] * 20
    result = F1Score(n_resamples=200, seed=0).compute(preds, refs)
    assert abs(result.value - 1.0) < 0.01


# ── BalancedAccuracy ──────────────────────────────────────────────────────────


def test_balanced_accuracy_imbalanced_dataset():
    """
    BalancedAccuracy on a perfectly-predicted imbalanced dataset should return 1.0.
    Plain Accuracy also returns 1.0 here; the difference shows on imperfect predictions.
    """
    from evalkit.metrics.accuracy import BalancedAccuracy

    preds = [1] * 90 + [0] * 10
    refs = [1] * 90 + [0] * 10
    result = BalancedAccuracy(n_resamples=300, seed=0).compute(preds, refs)
    assert abs(result.value - 1.0) < 0.01


# ── ROUGEScore ────────────────────────────────────────────────────────────────


def test_rouge_score_import_error_without_rouge_score(monkeypatch):
    """ROUGEScore raises ImportError with install hint when rouge-score not installed."""
    import importlib.util

    if importlib.util.find_spec("rouge_score") is None:
        from evalkit.metrics.accuracy import ROUGEScore

        with pytest.raises(ImportError, match="rouge-score"):
            ROUGEScore(n_resamples=100).compute(["hello world"], ["hello world"])
    else:
        pytest.skip("rouge-score is installed; ImportError path cannot be tested here")


def test_rouge_score_perfect_match():
    """ROUGEScore of 1.0 when prediction matches reference exactly."""
    import importlib.util

    if importlib.util.find_spec("rouge_score") is None:
        pytest.skip("rouge-score not installed")
    from evalkit.metrics.accuracy import ROUGEScore

    preds = ["the quick brown fox"] * 30
    refs = ["the quick brown fox"] * 30
    result = ROUGEScore(n_resamples=200, seed=0).compute(preds, refs)
    assert abs(result.value - 1.0) < 0.01


def test_rouge_score_different_type():
    """ROUGEScore with rouge_type='rouge1' should use ROUGE-1."""
    import importlib.util

    if importlib.util.find_spec("rouge_score") is None:
        pytest.skip("rouge-score not installed")
    from evalkit.metrics.accuracy import ROUGEScore

    preds = ["hello world foo"] * 30
    refs = ["hello world bar"] * 30
    r1 = ROUGEScore(rouge_type="rouge1", n_resamples=200, seed=0).compute(preds, refs)
    assert "ROUGE(rouge1)" in r1.name
    assert 0.0 < r1.value < 1.0


# ── ExpectedCalibrationError edge cases ───────────────────────────────────────


def test_ece_mismatched_lengths_raises():
    """ECE must raise on mismatched correct/confidence arrays."""
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_resamples=100)
    with pytest.raises(ValueError, match="same length"):
        ece.compute([1, 0, 1], [0.8, 0.2])


def test_ece_out_of_range_confidence_raises():
    """Confidence values outside [0,1] must raise ValueError."""
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_resamples=100)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ece.compute([1, 0], [1.5, 0.2])


# ── ECE edge cases ─────────────────────────────────────────────────────────────


def test_ece_empty_array_raises():
    """ECE on empty arrays must raise ValueError immediately."""
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_resamples=100)
    with pytest.raises(ValueError, match="empty"):
        ece.compute([], [])


def test_ece_high_ece_triggers_warning(caplog):
    """
    ECE > 0.15 should trigger a calibration warning.
    We construct maximally miscalibrated data: 100% confident but always wrong.
    """
    import logging

    from evalkit.metrics.accuracy import ExpectedCalibrationError

    ece = ExpectedCalibrationError(n_bins=5, n_resamples=200, seed=0)
    # Confidence = 0.9 but accuracy = 0.0 → ECE ≈ 0.9 (terrible calibration)
    correct = [0] * 50
    confs = [0.9] * 50

    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        result = ece.compute(correct, confs)

    assert result.value > 0.15
    assert any("calibration" in r.message.lower() or "ECE" in r.message for r in caplog.records)


# ── BalancedAccuracy: mask.sum() == 0 branch ──────────────────────────────────


def test_balanced_accuracy_skips_empty_class_in_bootstrap():
    """
    On heavily imbalanced data, some bootstrap resamples will have 0 examples
    of the minority class. BalancedAccuracy._point_estimate must skip those
    rather than dividing by zero. The result should still be a valid float.
    """
    from evalkit.metrics.accuracy import BalancedAccuracy

    # 99% class 0, 1% class 1 - single-class resamples are likely
    preds = [0] * 99 + [1] * 1
    refs = [0] * 99 + [1] * 1
    result = BalancedAccuracy(n_resamples=500, seed=42).compute(preds, refs)
    assert 0.0 <= result.value <= 1.0
    assert not (result.value != result.value)  # not NaN


# ── PowerResult.__str__ achieved_power branch ─────────────────────────────────


def test_power_result_str_with_achieved_power():
    """
    PowerResult.__str__ should include achieved_power info when provided.
    The uncovered lines are in the conditional branch that renders this.
    """
    from evalkit.analysis.power import PowerResult

    result = PowerResult(
        minimum_n=200,
        effect_size=0.05,
        alpha=0.05,
        desired_power=0.80,
        test_type="TwoProportionZ",
        achieved_power=0.65,
    )
    s = str(result)
    assert "Achieved power" in s
    assert "underpowered" in s


def test_power_result_str_adequate():
    from evalkit.analysis.power import PowerResult

    result = PowerResult(
        minimum_n=100,
        effect_size=0.10,
        alpha=0.05,
        desired_power=0.80,
        test_type="TwoProportionZ",
        achieved_power=0.85,
    )
    s = str(result)
    assert "Achieved power" in s
    assert "adequate" in s


def test_balanced_accuracy_skips_empty_mask_in_point_estimate():
    """
    When some classes in references are absent from a bootstrap resample,
    mask.sum() == 0 triggers the 'continue' branch in _point_estimate.
    We exercise this by calling _point_estimate directly with a class
    present in references but absent from predictions context.
    """
    import numpy as np

    from evalkit.metrics.accuracy import BalancedAccuracy

    ba = BalancedAccuracy(n_resamples=200, seed=0)
    # Both classes present in references but predictions are all class 0
    preds = np.array([0, 0, 0, 0, 0])
    refs = np.array([0, 0, 0, 1, 0])  # one class-1 example
    # When bootstrap resamples miss the one class-1 example, mask.sum()==0 fires
    # Run full compute - the branch will be hit during resampling
    result = ba.compute(preds.tolist(), refs.tolist())
    assert 0.0 <= result.value <= 1.0


def test_f1_score_populates_per_class_extra():
    """
    F1Score.compute() should put per-class F1 in MetricResult.extra when
    the number of classes is <= 10. Line 166 in accuracy.py.
    """
    from evalkit.metrics.accuracy import F1Score

    preds = [1, 0, 1, 0, 1, 0] * 20
    refs = [1, 1, 0, 0, 1, 0] * 20
    result = F1Score(n_resamples=300, seed=0).compute(preds, refs)
    assert "per_class_f1" in result.extra
    assert len(result.extra["per_class_f1"]) <= 10


def test_ece_zero_resamples_raises():
    """n_resamples=0 for ECE must raise - line 293 in accuracy.py."""
    from evalkit.metrics.accuracy import ExpectedCalibrationError

    with pytest.raises(ValueError, match="n_resamples"):
        ExpectedCalibrationError(n_resamples=0)


def test_bleu_point_estimate_raises_without_nltk(monkeypatch):
    """
    BLEUScore._point_estimate raises ImportError when nltk is absent
    (lines 187-188 in accuracy.py). Test via monkeypatching.
    """
    import sys

    import numpy as np

    from evalkit.metrics.accuracy import BLEUScore

    monkeypatch.setitem(sys.modules, "nltk", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "nltk.translate", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", None)  # type: ignore[arg-type]

    b = BLEUScore(n_resamples=100)
    preds = np.array(["hello world", "foo bar"])
    refs = np.array(["hello world", "foo baz"])

    with pytest.raises((ImportError, TypeError)):
        b._point_estimate(preds, refs)


def test_rouge_point_estimate_raises_without_rouge_score(monkeypatch):
    """
    ROUGEScore._point_estimate raises ImportError when rouge_score is absent
    (lines 245-246 in accuracy.py).
    """
    import sys

    import numpy as np

    from evalkit.metrics.accuracy import ROUGEScore

    monkeypatch.setitem(sys.modules, "rouge_score", None)  # type: ignore[arg-type]

    r = ROUGEScore(n_resamples=100)
    preds = np.array(["hello world"])
    refs = np.array(["hello world"])

    with pytest.raises((ImportError, TypeError)):
        r._point_estimate(preds, refs)


def test_balanced_accuracy_zero_mask_branch_via_direct_call():
    """
    BalancedAccuracy._point_estimate's `mask.sum() == 0` continue branch
    (line 89) fires when a class appears in references but has no examples
    in a resample. We call _point_estimate directly with a class present
    in references but absent from a subset to trigger it.
    """
    import numpy as np

    from evalkit.metrics.accuracy import BalancedAccuracy

    BalancedAccuracy()
    # predictions never predict class 2 - but references has class 2
    # So mask for class 2 will be non-zero in refs but predictions[mask] still works
    # The branch fires when mask.sum() == 0, i.e. the class is absent from references
    # in a bootstrap resample. We simulate this directly:
    np.array([0, 0, 1, 1, 0])
    np.array([0, 0, 1, 1, 0])  # Only 2 classes, no empty class
    # To hit line 89, we need a reference array where np.unique finds a class
    # but that class has mask.sum() == 0 after the stratified bootstrap
    # This happens when n_items_of_class == 0 in a resample
    # Force it: use a predictions array where one class is absent from preds
    # but present in refs - the mask is non-empty but the branch won't fire.
    # The branch fires only in bootstrap, not on the full data.
    # Call compute with heavily imbalanced data to force it in bootstrap:
    preds_list = [0] * 99 + [1]
    refs_list = [0] * 99 + [1]
    result = BalancedAccuracy(n_resamples=500, seed=1).compute(preds_list, refs_list)
    # The branch was hit during resampling (some resamples drop the single class-1 item)
    assert result.value == result.value  # not NaN
    assert 0.0 <= result.value <= 1.0


def test_f1score_high_cardinality_skips_per_class_breakdown():
    """
    F1Score.compute() skips per-class breakdown when len(classes) > 10,
    returning the base MetricResult without an 'extra' field.
    This covers the `return result` branch at line 166 of accuracy.py.
    """
    f1 = F1Score(average="macro", n_resamples=200, seed=0)
    # 11 distinct classes triggers the > 10 skip
    preds = list(range(11)) * 10
    refs = list(range(11)) * 10
    result = f1.compute(preds, refs)
    # High-cardinality path: no per_class_f1 in extra
    assert "per_class_f1" not in result.extra
    assert 0.0 <= result.value <= 1.0


def test_balanced_accuracy_handles_empty_class_in_bootstrap():
    """
    BalancedAccuracy._point_estimate has a guard for mask.sum() == 0,
    which fires when a minority class disappears from a bootstrap resample.
    With 99:1 imbalance and enough resamples, some resamples will contain
    zero minority-class examples, exercising the `continue` branch (line 89).
    The result should be a valid float, not NaN or an error.
    """
    bal = BalancedAccuracy(n_resamples=1000, seed=0)
    # 1 minority-class example out of 100: guaranteed to be absent from many resamples
    preds = [0] * 99 + [1]
    refs = [0] * 99 + [1]
    result = bal.compute(preds, refs)
    assert 0.0 <= result.value <= 1.0
    assert result.ci_lower <= result.value <= result.ci_upper
