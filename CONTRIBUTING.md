# Contributing to evalkit

## Development setup

```bash
git clone https://github.com/bonnie-mcconnell/evalkit
cd evalkit
pip install -e ".[dev]"
```

Verify everything works before making changes:

```bash
pytest          # fast - no coverage overhead
python examples/full_workflow.py
```

## Running the tests

```bash
# Full suite, no coverage (fast - for normal development)
pytest

# Full suite with coverage (what CI runs - slower)
pytest --cov=evalkit --cov-fail-under=85

# Skip slow bootstrap tests during quick iteration
pytest -m "not slow"

# Single module
pytest tests/metrics/test_comparison.py
```

The test suite is intentionally statistical: many tests check that CIs contain the true value "approximately" rather than asserting exact numbers. This is correct - the bootstrap is a stochastic procedure and exact values depend on seed. If you find a test that asserts an exact float from a bootstrap, that's a bug in the test, not something to preserve.

## Code conventions

**No point estimates without CIs.** Every metric function must return a `MetricResult`. Returning a bare float from a public function is not acceptable regardless of convenience.

**Statistical correctness over cleverness.** If you are unsure whether a statistical method is correct, write the dumb-but-obviously-right version first, then add a reference.

**Imports.** All imports at the top of the file. No local imports inside functions except in four cases: optional heavy dependencies (`sentence_transformers`, `krippendorff`, `nltk`, `rouge_score`), avoiding circular imports, FastAPI route handlers (lazy loading to keep app startup fast), and CLI command handlers (`evalkit/cli.py` - lazy imports keep `evalkit version` from loading scipy). Any other local import is a code smell.

**No `print()` in library code.** Use `logging` with appropriate levels. `DEBUG` for internals, `INFO` for user-facing progress, `WARNING` for things the user should know about. The CLI and examples may use `rich` for output. The one exception is `PowerAnalysis.sample_size_table()`, which intentionally prints to stdout by default (it is a planning tool designed for interactive use) but accepts `print_table=False` for programmatic callers.

**Type hints on all public functions.** Private helpers (`_point_estimate`, `_stratify_indices`) need type hints too.

**Docstrings.** Public classes and methods need docstrings explaining *why*, not *what*. If the docstring just restates the function signature, delete it and write something useful.

## Adding a new metric

1. Subclass `Metric` in `evalkit/metrics/`. `Metric` is importable from `evalkit` directly.
2. Implement `name` (property) and `_point_estimate(predictions, references)`.
3. Let the base class handle the bootstrap - don't reimplement it.
4. If your metric has a fundamentally different interface (like `ECE` or `KrippendorffAlpha`), do **not** subclass `Metric`. Write a standalone class with a `compute()` method that returns a `MetricResult`.
5. Export from `evalkit/metrics/__init__.py` and `evalkit/__init__.py`.
6. Add tests in `tests/metrics/`. Tests must check statistical properties, not exact values.

```python
from evalkit import Metric, MetricResult
import numpy as np

class TopKAccuracy(Metric):
    """Correct if the reference appears in the top-k candidates."""

    def __init__(self, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    @property
    def name(self) -> str:
        return f"Top{self.k}Accuracy"

    def _point_estimate(self, predictions: np.ndarray, references: np.ndarray) -> float:
        # predictions[i] is a list of k candidates; references[i] is the correct answer
        return float(np.mean([r in p[: self.k] for p, r in zip(predictions, references)]))
```

## Adding a new judge

1. Subclass `DeterministicJudge` or `StochasticJudge` from `evalkit.core.judge`.
   Both are importable directly from `evalkit` as of v0.1.0.
2. Implement `judge(output, reference) -> JudgmentResult`.
3. Return `JudgmentResult(score=..., is_correct=..., raw_output=output)`.
   `score` must be in [0, 1]. `is_correct` is the binary correctness flag.
4. For stochastic judges (LLM-based, embedding-based), override `is_stochastic`
   to return `True` - the RigorChecker will require inter-rater agreement validation.
5. Export from `evalkit/__init__.py` if it belongs in the public API.
6. Add tests in `tests/core/test_judge.py`.

```python
from evalkit import DeterministicJudge, JudgmentResult

class ContainsJudge(DeterministicJudge):
    """Correct if the reference string appears anywhere in the output."""

    def judge(self, output: str, reference: object) -> JudgmentResult:
        correct = str(reference).lower() in output.lower()
        return JudgmentResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=output,
        )
```

## Adding a new provider

1. Subclass `ModelProvider` in `evalkit/providers/base.py`.
2. Implement `_call()` - no retry logic, that's handled by `complete()`.
3. Add pricing to the `_PRICING` class dict if the provider exposes token costs.
4. Add to `evalkit/providers/__init__.py`.
5. Add the optional dependency to `pyproject.toml` under a new extra (e.g., `[gemini]`).

## Pull request checklist

- [ ] `pytest` passes with no new failures
- [ ] `mypy evalkit/` reports no errors (strict mode)
- [ ] `ruff check evalkit/ tests/ examples/` passes
- [ ] `ruff format --check evalkit/ tests/ examples/` passes
- [ ] Coverage remains ≥ 85% (`pytest --cov=evalkit --cov-fail-under=85`)
- [ ] New public API is exported from `evalkit/__init__.py` and the appropriate sub-`__init__.py`
- [ ] Statistical methods have a reference comment (author, year, paper/book name)
- [ ] No bare floats returned from metric functions - always `MetricResult`
- [ ] `python examples/full_workflow.py` runs end-to-end without error

## Statistical methods: what we use and why

| Need | Method | Why not the alternative |
|---|---|---|
| CI on accuracy | Percentile bootstrap, stratified | Wilson interval only applies to proportions; bootstrap works for F1, BalancedAccuracy, ECE. Stratification prevents CI collapse on imbalanced data |
| Model comparison (binary) | McNemar's with continuity correction | Paired t-test assumes normality; McNemar is exact for paired binary outcomes |
| Model comparison (continuous) | Wilcoxon signed-rank | Paired t-test assumes normality; LLM scores are bounded and skewed |
| Multiple comparison correction | Benjamini-Hochberg FDR | Bonferroni is too conservative for k > 5; BH controls expected false discovery rate |
| Two-rater agreement | Cohen's kappa | Raw percent agreement doesn't correct for chance; kappa does |
| Multi-rater agreement | Krippendorff's alpha | Generalises kappa to n raters, ordinal/continuous scales, and missing data |

## Reporting a bug

Open a GitHub issue with:
1. The minimal code that reproduces it
2. What you expected
3. What you got (including the full traceback)
4. Your Python version and `pip show evalkit-research` output

Statistical bugs (wrong CI coverage, incorrect p-values) are taken especially seriously. Include the mathematical argument for why the result is wrong, not just "the number looks off."
