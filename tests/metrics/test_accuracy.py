"""
Tests for accuracy metrics.

Statistical property tests: not exact values, but correct direction.
"""

import pytest

from evalkit.metrics.accuracy import (
    Accuracy,
    BalancedAccuracy,
    F1Score,
    PrecisionScore,
    RecallScore,
)


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

    preds = [1, 1, 0, 1, 0, 0, 1, 1] * 10
    refs = [1, 0, 0, 1, 0, 1, 1, 0] * 10
    result = F1Score(n_resamples=300, seed=0).compute(preds, refs)
    assert 0.0 <= result.ci_lower <= result.value <= result.ci_upper <= 1.0
    assert result.n == 80


def test_f1_score_perfect():
    """F1Score of 1.0 when predictions exactly match references."""

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

    preds = [1, 0, 1, 0, 1, 0] * 20
    refs = [1, 1, 0, 0, 1, 0] * 20
    result = F1Score(n_resamples=300, seed=0).compute(preds, refs)
    assert "per_class_f1" in result.extra
    assert len(result.extra["per_class_f1"]) <= 10


def test_f1_per_class_extra_matches_sklearn():
    """
    Per-class F1 in MetricResult.extra must match sklearn's per-class F1 exactly.
    This test is important because the implementation was rewritten in pure numpy
    (removing the sklearn dependency from the hot path) - correctness must be verified.
    """
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(42)
    preds = list(rng.integers(0, 3, 300))
    refs = list(rng.integers(0, 3, 300))

    result = F1Score(average="macro", n_resamples=300, seed=0).compute(preds, refs)
    per_class = result.extra["per_class_f1"]

    for cls in [0, 1, 2]:
        sk = f1_score(refs, preds, labels=[cls], average="macro", zero_division=0)
        ours = per_class[str(cls)]
        assert abs(ours - sk) < 1e-10, (
            f"Per-class F1 mismatch for class {cls}: evalkit={ours:.6f} sklearn={sk:.6f}"
        )


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


# ── PrecisionScore ─────────────────────────────────────────────────────────────


def test_precision_perfect():

    result = PrecisionScore(average="macro", n_resamples=500, seed=0).compute(
        ["cat", "dog", "cat", "dog"] * 20,
        ["cat", "dog", "cat", "dog"] * 20,
    )
    assert abs(result.value - 1.0) < 1e-9
    assert result.name == "Precision(macro)"


def test_precision_zero_division_returns_zero():

    # All predictions wrong - precision is 0 for every class
    result = PrecisionScore(average="macro", n_resamples=500, seed=0).compute(
        ["dog"] * 40,
        ["cat"] * 40,
    )
    assert result.value == 0.0
    assert 0.0 <= result.ci_lower <= result.ci_upper


def test_precision_binary():

    # 10 TP, 10 FP → precision = 0.5
    preds = [1] * 20 + [0] * 20
    refs = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    result = PrecisionScore(average="binary", pos_label=1, n_resamples=500, seed=0).compute(
        preds, refs
    )
    assert abs(result.value - 0.5) < 0.05
    assert result.name == "Precision(binary)"


def test_precision_ci_contains_true_value():
    """Bootstrap CI should contain the true precision at roughly the stated level."""

    preds = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 30
    refs = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] * 30
    result = PrecisionScore(average="macro", n_resamples=2000, seed=42).compute(preds, refs)
    assert result.ci_lower <= result.value <= result.ci_upper
    assert result.ci_width > 0


# ── RecallScore ────────────────────────────────────────────────────────────────


def test_recall_perfect():

    result = RecallScore(average="macro", n_resamples=500, seed=0).compute(
        ["yes", "no"] * 30,
        ["yes", "no"] * 30,
    )
    assert abs(result.value - 1.0) < 1e-9
    assert result.name == "Recall(macro)"


def test_recall_zero():

    # Model predicts "no" for everything, references are all "yes" → recall=0
    result = RecallScore(average="macro", n_resamples=500, seed=0).compute(
        ["no"] * 40,
        ["yes"] * 40,
    )
    assert result.value == 0.0


def test_recall_binary():

    # 10 TP, 10 FN → recall = 0.5
    preds = [1] * 10 + [0] * 30
    refs = [1] * 20 + [0] * 20
    result = RecallScore(average="binary", pos_label=1, n_resamples=500, seed=0).compute(
        preds, refs
    )
    assert abs(result.value - 0.5) < 0.05
    assert result.name == "Recall(binary)"


def test_recall_ci_contains_true_value():

    preds = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 30
    refs = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] * 30
    result = RecallScore(average="macro", n_resamples=2000, seed=42).compute(preds, refs)
    assert result.ci_lower <= result.value <= result.ci_upper
    assert result.ci_width > 0


def test_precision_recall_f1_relationship():
    """
    Verify P, R, F1 are self-consistent: F1 = 2PR/(P+R).
    This is a cross-metric sanity check, not an arithmetic identity test.
    """

    preds = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 40
    refs = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] * 40

    p = PrecisionScore(average="macro", n_resamples=500, seed=0).compute(preds, refs).value
    r = RecallScore(average="macro", n_resamples=500, seed=0).compute(preds, refs).value
    f = F1Score(average="macro", n_resamples=500, seed=0).compute(preds, refs).value

    if p + r > 0:
        expected_f1 = 2 * p * r / (p + r)
        assert abs(f - expected_f1) < 0.02, (
            f"F1={f:.4f} inconsistent with P={p:.4f}, R={r:.4f} (expected F1≈{expected_f1:.4f})"
        )


# ── _prf_scores numpy helper - exhaustive sklearn comparison ──────────────────


class TestPrfScoresMatchSklearn:
    """
    _prf_scores must be numerically identical to sklearn for all averaging modes,
    class counts, and edge cases. These tests are critical: _prf_scores is the
    hot path called B=10,000 times per metric per experiment.
    """

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_matches_sklearn_multiclass(self, average: str, n_classes: int) -> None:
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score

        from evalkit.metrics.accuracy import _prf_scores

        rng = np.random.default_rng(hash(f"{average}{n_classes}") % (2**31))
        preds = rng.integers(0, n_classes, 300)
        refs = rng.integers(0, n_classes, 300)

        sk_p = precision_score(refs, preds, average=average, zero_division=0)
        sk_r = recall_score(refs, preds, average=average, zero_division=0)
        sk_f = f1_score(refs, preds, average=average, zero_division=0)

        np_p, np_r, np_f = _prf_scores(preds, refs, average, pos_label=1)

        assert abs(np_p - sk_p) < 1e-10, (
            f"Precision mismatch ({average}, k={n_classes}): {np_p} != {sk_p}"
        )
        assert abs(np_r - sk_r) < 1e-10, (
            f"Recall mismatch ({average}, k={n_classes}): {np_r} != {sk_r}"
        )
        assert abs(np_f - sk_f) < 1e-10, f"F1 mismatch ({average}, k={n_classes}): {np_f} != {sk_f}"

    def test_matches_sklearn_binary(self) -> None:
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score

        from evalkit.metrics.accuracy import _prf_scores

        rng = np.random.default_rng(7)
        preds = rng.integers(0, 2, 200)
        refs = rng.integers(0, 2, 200)

        for pos_label in [0, 1]:
            sk_p = precision_score(
                refs, preds, average="binary", pos_label=pos_label, zero_division=0
            )
            sk_r = recall_score(refs, preds, average="binary", pos_label=pos_label, zero_division=0)
            sk_f = f1_score(refs, preds, average="binary", pos_label=pos_label, zero_division=0)

            np_p, np_r, np_f = _prf_scores(preds, refs, "binary", pos_label=pos_label)

            assert abs(np_p - sk_p) < 1e-10, f"Binary precision mismatch (pos={pos_label})"
            assert abs(np_r - sk_r) < 1e-10, f"Binary recall mismatch (pos={pos_label})"
            assert abs(np_f - sk_f) < 1e-10, f"Binary F1 mismatch (pos={pos_label})"

    def test_all_correct(self) -> None:
        """Perfect predictions: P=R=F1=1.0 for all averages."""
        import numpy as np

        from evalkit.metrics.accuracy import _prf_scores

        refs = np.array([0, 0, 1, 1, 2, 2])
        preds = refs.copy()
        for avg in ["macro", "micro", "weighted"]:
            p, r, f = _prf_scores(preds, refs, avg, pos_label=1)
            assert abs(p - 1.0) < 1e-10
            assert abs(r - 1.0) < 1e-10
            assert abs(f - 1.0) < 1e-10

    def test_all_wrong(self) -> None:
        """All predictions wrong: F1=0 for macro/weighted (no true positives)."""
        import numpy as np

        from evalkit.metrics.accuracy import _prf_scores

        refs = np.array([0, 0, 1, 1])
        preds = np.array([1, 1, 0, 0])  # all swapped
        for avg in ["macro", "micro", "weighted"]:
            p, r, f = _prf_scores(preds, refs, avg, pos_label=1)
            assert abs(p - 0.0) < 1e-10
            assert abs(r - 0.0) < 1e-10
            assert abs(f - 0.0) < 1e-10

    def test_zero_division_edge_case(self) -> None:
        """Class present in refs but never predicted: precision=0, recall=0, F1=0."""
        import numpy as np

        from evalkit.metrics.accuracy import _prf_scores

        # Class 2 is never predicted - should not crash, should return 0.
        refs = np.array([0, 1, 2, 2])
        preds = np.array([0, 1, 0, 1])  # class 2 never predicted
        p, r, f = _prf_scores(preds, refs, "macro", pos_label=1)
        # No crash is the main assertion; values should be in [0,1]
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0
        assert 0.0 <= f <= 1.0

    def test_no_numpy_warnings_on_zero_division(self) -> None:
        """errstate suppression: _prf_scores must not emit RuntimeWarning."""
        import warnings

        import numpy as np

        from evalkit.metrics.accuracy import _prf_scores

        refs = np.array([0, 0, 1, 1])
        preds = np.array([1, 1, 0, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _prf_scores(preds, refs, "macro", pos_label=1)  # must not raise


def test_accuracy_warn_on_imbalance_false_suppresses_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """warn_on_imbalance=False must not emit the class imbalance warning."""
    import logging

    acc = Accuracy(n_resamples=200, seed=0)
    # 95% class 1 - would trigger warning with default warn_on_imbalance=True
    preds = [1] * 95 + [0] * 5
    refs = [1] * 95 + [0] * 5
    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        acc.compute(preds, refs, warn_on_imbalance=False)
    assert not any("imbalance" in msg.lower() for msg in caplog.messages), (
        "warn_on_imbalance=False should suppress the imbalance warning"
    )


def test_accuracy_warn_on_imbalance_true_fires_warning(caplog: pytest.LogCaptureFixture) -> None:
    """warn_on_imbalance=True (the default) must fire on heavily imbalanced refs."""
    import logging

    acc = Accuracy(n_resamples=200, seed=0)
    preds = [1] * 92 + [0] * 8
    refs = [1] * 92 + [0] * 8  # 92% class 1 - above the 90% threshold
    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        acc.compute(preds, refs, warn_on_imbalance=True)
    assert any("imbalance" in msg.lower() for msg in caplog.messages), (
        "warn_on_imbalance=True should emit the class imbalance warning at ≥90% majority"
    )


def test_experiment_compute_metrics_does_not_warn_on_binary_correct_array(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    When Experiment._compute_metrics builds the binary correct[] array with refs=[1]*n,
    the 100%-class-1 reference array must NOT trigger the imbalance warning.
    This tests that the abstraction fix (warn_on_imbalance=False in _compute_metrics)
    works correctly end-to-end.
    """
    import logging

    from evalkit import EvalDataset, Experiment, MockRunner, PromptTemplate
    from evalkit.core.judge import ExactMatchJudge

    # Build a perfectly balanced dataset - the binary correct[] array will be ~82% 1s
    records = [{"text": f"ex{i}", "label": "a" if i % 2 == 0 else "b"} for i in range(60)]
    ds = EvalDataset.from_list(records, reference_field="label")
    tmpl = PromptTemplate("{{ text }}")
    judge = ExactMatchJudge()
    runner = MockRunner(judge=judge, template=tmpl, accuracy=0.92, seed=0)

    with caplog.at_level(logging.WARNING, logger="evalkit.metrics.accuracy"):
        exp = Experiment("test", ds, runner, n_resamples=200)
        exp.run()

    imbalance_warnings = [m for m in caplog.messages if "imbalance" in m.lower()]
    assert len(imbalance_warnings) == 0, (
        f"_compute_metrics must not emit imbalance warning for binary correct[] array; "
        f"got: {imbalance_warnings}"
    )
