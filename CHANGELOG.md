# Changelog

All notable changes to evalkit are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `evalkit run --judge llm` - LLM-as-judge from the CLI. Uses the same provider
  as `--model`, so `evalkit run data.jsonl --model gpt-4o-mini --judge llm` runs
  the evaluation and scores each response with the same model. A reminder to
  validate inter-rater agreement prints automatically.

### Fixed

- `SemanticSimilarityJudge._model` annotated as `SentenceTransformer | None` -
  fixes a mypy strict error on Python 3.12 where the inferred type was `None`.
- `# noqa: E501` comments removed from the HTML template string in `report.py`.
  These Python linter suppression comments were being written verbatim into every
  generated HTML report and rendered as visible text. The `pyproject.toml`
  per-file-ignores already suppresses E501 for `report.py` so the comments
  were never needed. A regression test now prevents this from recurring.
- `SECURITY.md` corrected contact email to `bonniep.mcconnell@gmail.com`.
- `.gitignore` extended to cover local scratch test files (`test_data.jsonl`,
  `test_api.py`, `test_dataset.py`, `test_checkpoint.py`, `report.html`).

---

## [0.1.0] - 2026-04-01

First public release.

### Core pipeline

- `EvalDataset` - typed dataset container with loaders for JSONL, CSV, list-of-dicts,
  and HuggingFace Datasets. Validates unique IDs (required for aligned paired tests).
  `split()` for stratified train/test splitting. `sample()` for cheap prototyping.
- `PromptTemplate` - Jinja2 rendering with `StrictUndefined` (missing fields raise
  immediately, never silently produce empty strings). `validate(dataset)` checks all
  variables exist before spending any API budget.
- `AsyncRunner` - async evaluation with concurrency control via `asyncio.Semaphore`,
  exponential-backoff retries, and atomic checkpointing (write-to-temp-then-rename).
  Synchronous provider calls run in a thread pool to avoid blocking the event loop.
- `MockRunner` - deterministic seeded runner for CI pipelines and examples. Operates
  at runner level (not provider level) so `ExactMatchJudge` scores correctly against
  the reference.
- `ExactMatchJudge`, `RegexMatchJudge` - deterministic judges. `LLMJudge`,
  `SemanticSimilarityJudge` - stochastic judges with inter-rater agreement validation
  required.
- `OpenAIProvider`, `AnthropicProvider`, `MockProvider` - provider abstraction with
  retry logic, cost tracking, and token counting.
- `Experiment` - single entry point that runs pre-flight audit, evaluation, metrics,
  and post-hoc audit in sequence.

### ExperimentResult methods

- `print_summary()` - concise human-readable output for scripts and notebooks.
- `compare(other)` - paired significance test (auto-selects McNemar vs Wilcoxon).
  Verifies example alignment. Returns `ComparisonResult` with plain-English
  interpretation including required N when result is not significant.
- `worst_examples(n)` - the N examples most confidently answered wrong, sorted by
  score. For error analysis.
- `to_dataframe()` - per-example results as a pandas DataFrame (requires pandas).

### Metrics with bootstrap CIs

- `Accuracy`, `BalancedAccuracy`, `F1Score` - classification metrics.
- `BLEUScore`, `ROUGEScore` - generation metrics (optional nltk/rouge-score deps).
- `ExpectedCalibrationError` (ECE) - with correct last-bin inclusive boundary
  (`confidence <= 1.0`), fixing a subtle bug in most ECE implementations.
- Stratified percentile bootstrap (10,000 resamples by default) in the `Metric`
  base class. Stratification prevents CI collapse on imbalanced datasets.
- `MetricResult` frozen dataclass - point estimate outside CI raises at construction
  time, catching bootstrap bugs immediately.

### Statistical comparison

- `McNemarTest` - paired binary outcomes. Edwards' continuity correction. Laplace-smoothed odds ratio.
- `WilcoxonTest` - paired continuous scores. Rank-biserial correlation effect size.
- `BHCorrection` - Benjamini-Hochberg FDR correction for K ≥ 2 comparisons. Flags
  when unadjusted conclusions differ from corrected ones.
- `CohenKappa`, `KrippendorffAlpha` - inter-rater agreement with bootstrap CIs and
  Landis & Koch interpretation.

### Analysis

- `RigorChecker` - two-pass automated statistical audit (pre-flight + post-hoc).
  Finding codes: `SAMPLE_TOO_SMALL`, `UNDERPOWERED_CI`, `UNDERPOWERED_COMPARISON`,
  `CLASS_IMBALANCE`, `SEVERE_CLASS_IMBALANCE`, `MULTIPLE_TESTING_RISK`,
  `MULTIPLE_TESTING_UNCORRECTED`, `MULTIPLE_TESTING_NO_PVALUES`,
  `JUDGE_AGREEMENT_REQUIRED`, `LOW_JUDGE_AGREEMENT`.
- `PowerAnalysis` - four methods: `for_proportion_difference`, `for_mcnemar`,
  `for_ci_precision`, `for_wilcoxon`. `sample_size_table()` prints a formatted
  planning grid.
- `ReportGenerator` - self-contained HTML tearsheet. Single-pass regex substitution
  prevents double-expansion of template tokens.

### Interfaces

- Python API - all public classes importable from `evalkit` root namespace.
- CLI - `evalkit run` (with `--format json` for scripting), `evalkit compare`,
  `evalkit power`, `evalkit table`, `evalkit version`.
- REST API - FastAPI with lifespan context manager, async background tasks, result
  persistence. Swagger UI at `/docs`.

### Infrastructure

- `py.typed` marker - full PEP 561 compliance.
- `mypy strict = true` - all public and private functions fully typed.
- 394 tests, 100% coverage across 20 source files. Statistical property tests verify
  CI coverage and width scaling with N. `stratify=False` where appropriate, documented
  in each test.
- GitHub Actions CI: test matrix (Python 3.11, 3.12), ruff lint, mypy, full workflow
  example, package build verification.
- Docker + docker-compose for the REST API.
- Makefile: `make install`, `make test`, `make fmt`, `make lint`, `make check`, `make demo`.

### Implementation notes

- `CostSummary` TypedDict on `ModelProvider.cost_summary()` - eliminates the
  `dict[str, object]` return type that required `type: ignore` suppressions in
  the runner.
- `PowerAnalysis.sample_size_table()` accepts `print_table: bool = True` -
  programmatic callers can suppress stdout output.
- `_severity_sort_key()` function in `rigour.py` - replaces direct dict import
  across module boundaries, cleaning up the internal API.
- `OpenAIProvider` and `AnthropicProvider` exported from the `evalkit` root namespace
  - `from evalkit import OpenAIProvider` now works directly.