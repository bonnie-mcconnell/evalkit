# Changelog

All notable changes to evalkit are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.1] - 2026-05-13

### Added

- **`examples/data/factual_qa_50.jsonl`** - updated to avoid substring collisions
  in `contains`-judge runs. All 50 answers are now multi-character strings; element
  symbol questions use the element name, math questions use shape names and
  word-form numbers, and `lang_001` now asks for "Cyrillic" instead of a near-
  duplicate language prompt.

- **`examples/openai_quickstart.py`** - complete runnable script for live API
  validation against OpenAI. Handles import errors, missing API key, dataset
  location, and produces a full audit report with cost summary and error analysis.
  Equivalent to running `evalkit run examples/data/factual_qa_50.jsonl --model
  gpt-4o-mini --template "Answer in one word or short phrase: {{ question }}"
  --ref-field answer --judge contains` but with richer Python-level output.

- **`docs/design_decisions.md`** - architectural and statistical trade-offs
  documented with rationale. Covers percentile bootstrap vs BCa, stratified
  resampling, McNemar vs t-test vs Wilcoxon, BH vs Bonferroni, sync provider
  wrapping in a thread pool, checkpoint type fidelity, REST API scope, frozen
  dataclass pattern, and 100% coverage rationale.

- **`examples/data/README.md` rewritten** - now documents the complete JSONL
  format specification: required fields, template field syntax, optional fields,
  judge selection guide, common mistakes, and all three included datasets with
  run commands.

- **README: real-provider quickstart section** - added before Step 0, showing
  the complete path from install to live API results in 3 commands (mock demo,
  OpenAI, Anthropic, and "your own data" with format guidance).

- **`MANIFEST.in`** - created. Without it, `python -m build` produces a `.tar.gz`
  that omits `examples/data/*.jsonl`, `examples/walkthrough.ipynb`, and `docs/*.md`.
  Verified by checking `tar -tzf dist/evalkit_research-*.tar.gz`.

- **`.github/workflows/ci.yml`: sdist verification step** - the `package` job
  now builds both wheel and sdist, then explicitly checks that
  `examples/data/balanced_demo.jsonl`, `examples/walkthrough.ipynb`, and
  `docs/statistical_methods.md` are present in the tar. MANIFEST.in breakage is
  now caught in CI rather than discovered post-publish.

- **`Experiment(expected_accuracy=0.70)`** - new parameter forwarded to the
  pre-flight CI precision check. Previously the check always assumed p=0.70,
  silently over-estimating the required N for high-accuracy models. The default
  is unchanged (backwards-compatible); pass the actual expected accuracy when
  your model is expected to perform well above 70%.

- **`docs/statistical_methods.md`: Post-hoc Sample Size Estimation section.**
  Documents why `_approx_required_n` uses a two-proportion z-test rather than
  the exact McNemar/Wilcoxon formula, proves the conservative direction (displayed
  N is an upper bound on the true required N), and quantifies the over-estimate
  (≈1.3–1.7× for typical evaluation settings). Closes an internal inconsistency:
  the document derived exact paired-test formulas but the implementation used an
  approximation with no explanation.

### Fixed

- **`ComparisonResult.winner` silently returned wrong answer for unknown test
  names.** The property had a bare `else` clause that applied Wilcoxon semantics
  to any test name that was not "McNemar". Fixed: explicit `elif self.test_name
  == "Wilcoxon"` with a final `else: raise ValueError(...)`.

- **`_approx_required_n` output showed no approximation caveat.** Fixed: output
  now labels the number as "conservative estimate via two-proportion z-test; actual
  required N for a paired {test_name} test will be equal or lower."

- **`audit_comparisons()` returned an ambiguous empty `AuditReport` for a single
  comparison.** Fixed: returns an `AuditReport` with one `INFO` finding
  (`MULTIPLE_TESTING_NOT_APPLICABLE`).

- **CLI `--judge` help text omitted `contains`.** Fixed.

- **`frozen=True` dataclass with mutable `_comparison_p_values` was undocumented.**
  Added detailed comment explaining the pattern and the design rationale.

- **README Binder badge URLs used deprecated `filepath=` parameter.** Fixed to
  `urlpath=lab/tree/`.

- **README "What I'd do differently" section removed.** The rationale now lives
  in `docs/design_decisions.md`.

- **`evalkit version` printed BibTeX citation on every invocation.** This is
  aggressive and unexpected behaviour for a version command. Fixed: `evalkit
  version` now prints only the version. `evalkit version --cite` prints BibTeX.

- **`.gitignore` did not include `.env`.** Fixed before any API keys were created.
  `.env`, `.env.*`, `*.pem`, and `*.key` are now all excluded.

- **CI workflow `package` job built wheel only.** If `MANIFEST.in` was broken,
  the sdist would silently omit files and CI would not catch it. Fixed: now builds
  both wheel and sdist, with explicit file presence checks.

- **`openai_quickstart.py` cost display used `provider.cost_summary()`** which
  accumulates across all experiments sharing a provider instance. Switched to
  `result.run_result.total_cost_usd` and `.total_tokens`, which are always scoped
  to exactly this run. Wall-clock time added to the cost line.

- **`openai_quickstart.py` `except Exception as e: raise`** printed then re-raised,
  producing duplicate output. Fixed to bare `except Exception: raise`. ruff F841
  (unused variable `e`) also resolved.

- **`openai_quickstart.py` docstring** claimed `"✓ 50/50 complete"` which the
  script never prints. Fixed: docstring now shows actual output structure with
  honest X placeholders, not fabricated numbers.

- **README CLI section** had a stale provider-validation blockquote interrupting
  the reference section. Removed.

- **README "Running the demo"** referenced mock examples only. Added
  `openai_quickstart.py` as the recommended first step for users with an API key.

- **`docs/design_decisions.md` 100% coverage rationale** was circular. Rewritten
  to explain: statistical bugs are silent, wrong numbers look plausible, the cost
  of bad science exceeds the cost of slow development.

---

## [0.5.0] - 2026-05-08

### Fixed

- **`examples/data/README.md`** - corrected the reported CI width for
  `underpowered_imbalanced_demo.jsonl` from ±18pp to ±16pp.

- **`examples/full_workflow.py`** - corrected the sample-size guidance and fixed
  the numbering in the failure-mode list.

- **`docs/statistical_methods.md`** - corrected the sample-size example from
  n=854 to n=897.

- **README and docs** - corrected the reported 95% CI width for the 74%
  accuracy example from ±14pp to ±12pp.

- **`pyproject.toml` missing `Intended Audience :: Developers` classifier.**
  The library is explicitly designed for production CI pipelines and ML
  engineering teams, not only researchers. The PyPI classifier now reflects both
  audiences.

### Changed

- **README opening** now explicitly names the production problem alongside the
  research problem. Previously: "I kept reading ML papers..." which framed the
  project as academic. Now the second paragraph names the real-world failure mode:
  teams shipping prompt changes based on n=50 results, making model selection
  decisions where CIs overlap, running 20 variants and treating the winner as real.

- **README Architecture section** now surfaces the 20× numpy speedup in the
  bootstrap inner loop with concrete numbers: ~2.6ms/call overhead in sklearn ×
  10,000 resamples = 26s total; the numpy path runs in ~0.1ms/call, making the
  default 10,000-resample bootstrap complete in under 200ms on n=1,000 examples.
  Previously this was documented only in a code comment in `accuracy.py`.

### Added

- **`evalkit run --fail-on-errors`** - exits with code 1 when the RigorChecker
  post-hoc audit finds ERROR-level findings (underpowered sample, severe class
  imbalance, uncorrected multiple testing). Without the flag, behaviour is
  unchanged: audit results are printed but the process always exits 0. This flag
  is designed for CI pipelines that need to gate deployments on statistical quality.

- **`evalkit compare --format json`** - machine-readable output for scripting.
  Keys: `test`, `statistic`, `p_value`, `effect_size`, `reject_null`,
  `b_is_significantly_worse`, `n_pairs`, `alpha`, `note`, `files`.
  Example: `evalkit compare a.json b.json --format json | jq .b_is_significantly_worse`

- **`evalkit compare --fail-on-regression`** - exits with code 2 if the second
  run (`result_b`) is statistically significantly *worse* than the first (`result_a`).
  Exits 0 when there is no significant difference or `result_b` is better.
  Directionality for McNemar's test: `effect_size > 1` means model A won more
  discordant pairs, so B is worse. Designed for CI deployment gates.

- **README: GitHub Actions CI pipeline example** showing `--fail-on-errors` and
  `--fail-on-regression` wired together in a real workflow. The "Using evalkit
  in CI pipelines" section explains the exit code semantics.

- **`evalkit compare` now shows directional panels** in text output: green
  "Improvement Confirmed" when `result_b` is significantly better, red
  "Regression Detected" when it is significantly worse.

 The generated report shows
  up to 10 wrong examples sorted by score descending (most confidently incorrect first),
  with columns for prompt preview, model output, expected reference, score, and reasoning.
  Previously the tearsheet showed only aggregate metrics and audit findings - it had no
  per-example error visibility. `worst_examples()` was already accessible via the Python
  API; this change makes it visible in the default shareable output.

### Fixed

- **CLI `--api-key` now correctly forwarded to `AnthropicProvider`.** Previously, passing
  `--api-key sk-ant-...` when using a `claude-*` model was silently ignored -
  `AnthropicProvider` was constructed without the key argument and fell back to whatever
  `ANTHROPIC_API_KEY` was in the environment. Fixed to pass `api_key` to both providers.
  The `envvar="OPENAI_API_KEY"` annotation is also removed from the flag definition since
  it was wrong for Anthropic models.

- **Exponential backoff in provider retry now uses full jitter.** The previous
  implementation used `wait = 2**attempt` (deterministic: 1s, 2s, 4s), which causes a
  thundering herd when multiple concurrent requests all fail simultaneously and retry at
  identical intervals. Fixed to `random.uniform(0, 2**attempt)` - the AWS-recommended
  full-jitter pattern. Error message updated to show actual wait time to one decimal place.

- **`docs/statistical_methods.md` threshold corrected: 85% → 90% in three places.**
  `SEVERE_CLASS_IMBALANCE` fires at ≥90% majority class (matching the code). The docs
  previously said 85%. The two-tier system is now fully documented: `CLASS_IMBALANCE`
  (WARNING at ≥75%) and `SEVERE_CLASS_IMBALANCE` (ERROR at ≥90%), with rationale for both.

- **Dead `test_compare_with_incomplete_run_returns_400` test removed.** The test body
  ended with `pass` - it asserted nothing. Replaced by
  `test_compare_with_still_running_run_returns_400`, which actually verifies the 400/404
  response. Five new API input validation tests added covering empty `dataset_records`,
  oversized records, and out-of-range `n_resamples` and `mock_accuracy`.

- **REST API `RunRequest` now validates inputs synchronously.** `dataset_records` must
  have 1–10,000 items; `n_resamples` must be 100–100,000. Invalid requests return 422
  immediately instead of 202 → silent background failure.

- **`MetricResult.n_resamples` now accurately reflects what was used.** When
  `Experiment._compute_metrics` built the Accuracy `MetricResult` directly, it
  omitted `n_resamples=` from the constructor - so the stored value defaulted to
  `10_000` regardless of the actual resamples used. A run with `n_resamples=500`
  would store `n_resamples=10000` in the result. Fixed to pass `self.n_resamples`
  explicitly.

- **`SECURITY.md` updated to show `0.4.x` as the supported version.** It still
  listed `0.3.x` after the version bump.

- **README Step 2 Precision CI lower bound corrected** from `0.7999` to `0.8000`
  (actual output; the previous value was a rounding artefact from an earlier run).

- **`CITATION.cff` version corrected** to `0.4.0`. GitHub uses
  `CITATION.cff` for the "Cite this repository" sidebar button. It previously said
  `0.3.1`, making it disagree with `pyproject.toml` and `evalkit.__version__`.

- **`evalkit version` BibTeX year is now dynamic.** Previously hardcoded to `2026`.
  Now derives from `datetime.datetime.now().year` so it stays correct without manual
  updates. Test updated to match.

- **README comparison table year corrected** from `May 2025` to `May 2026`.

- **CLI `--judge contains` now has a CLI-level test** (`test_run_contains_judge`).
  Previously only a judge unit test existed; the CLI integration path was untested.

---

## [0.4.0] - 2026-05-06

### Added

- **`ContainsJudge`** - deterministic judge scoring 1.0 when the output contains
  the reference string as a substring. Most common real-world evaluation pattern
  ("does the answer mention Paris?"). Case-insensitive and whitespace-stripped by
  default. Available from the top-level package and from `evalkit run --judge contains`.

- **`EvalDataset.to_jsonl(path, reference_field="label")`** - saves any dataset to
  JSONL (inverse of `from_jsonl`). Makes it straightforward to commit a stratified
  train/test split to disk so it never changes between runs.

- **`ExperimentResult.save(path)`** - saves results to JSON in the same schema used
  by the REST API and CLI. Files can be passed directly to `evalkit compare`.

- **`ExperimentResult.generate_report(path=None)`** - shorthand for
  `ReportGenerator().generate(result, path)`. Returns HTML string or writes to file.

- **`ComparisonResult.save(path)`** - saves model comparison results to JSON.
  Useful for attaching to GitHub PRs and CI artefacts.

- **`MultipleComparisonResult.save(path)`** - saves BH-FDR correction results to
  JSON. All three result types now have a first-class save path.

- **`__repr__` on `EvalDataset`, `ExperimentResult`, `MetricResult`** - informative
  one-liners instead of memory addresses or full dataclass dumps. Useful in REPL
  and notebooks.

- **`evalkit version`** now prints a BibTeX `@software` citation entry alongside
  the version string.

- **`examples/benchmark_audit.py`** - five-scenario demonstration of common LLM
  evaluation failure modes on realistic synthetic data.

- **`docs/blog_post_devto.md`** and **`docs/linkedin_post.md`** - ready-to-publish
  posts explaining the library's thesis.

- **`docs/statistical_methods.md` judge selection guide** - "Choosing a judge"
  section explains when to use each of the five judge types with concrete examples
  and a decision rule.

- **`make audit`** target, **`make notebook`** target.

### Performance

- **Bootstrap CI 4× faster for F1Score, PrecisionScore, RecallScore.** Replaced
  sklearn per-call overhead (~2.6ms/call × 10,000 = 26s) with a pure-NumPy
  `_prf_scores` helper (~0.1ms/call). Results are numerically identical to sklearn
  across all averaging modes (macro, micro, weighted, binary), verified by 14 tests.
  Measured: 5 metrics at B=10,000, n=1,000 - 22,000ms → 5,700ms.

- **Bootstrap index pre-generation ~3× faster.** Indices are now pre-generated in
  chunks of 50 resamples at a time rather than one call per resample, amortising
  RNG overhead without materialising the full (B × n) matrix.

- **Test suite time halved**: ~57s → ~21s (same 457 tests, same coverage).

### Fixed

- **All `ImportError` messages** reference the correct `evalkit-research[extra]`
  install path instead of bare `pip install <package>`.

- **`Experiment._compute_metrics` exception handling** tightened: `ImportError`
  (optional dep) is logged as a warning; all other exceptions surface immediately
  as `RuntimeError` so bugs aren't silently hidden.

- **`ExperimentResult.compare(test=...)` raises `ValueError`** for invalid test
  strings (e.g. `"mann-whitney"`). Previously silently treated as `"auto"`.

- **`sample_size_table(test='ci')`** now shows a clean single-column table instead
  of repeating identical power-level columns with a disclaimer.

- **HTML report supports light mode and printing.** Added
  `@media (prefers-color-scheme: light)` and `@media print` CSS blocks.
  Dark-mode-only report was previously unprintable and broken in light browsers.

- **`EvalDataset.label_distribution()` cached.** O(n) computation on first call,
  O(1) thereafter. Affects `__repr__`, `RigorChecker`, and `split(stratify=True)`.

- **`F1Score` per-class breakdown** no longer depends on sklearn at runtime.
  Replaced with a direct one-vs-rest TP/FP/FN implementation (numerically
  identical, verified to 1e-10).

- **`MockRunner` MD5 comment** explains the choice (speed, compactness, not a
  security context) to answer the obvious interview follow-up.

- **Walkthrough notebook** (`examples/walkthrough.ipynb`) populated with real
  executed outputs and verified to match the numbers in the README.

- **`evalkit compare`** now accepts files from both `evalkit run --save-results`
  and `ExperimentResult.save()` (schema unified).

- **README comparison table** has a verification date and an invitation to open
  an issue if any framework has added the listed features.

- **Blog post call to action** explains the required data format (JSONL with input
  fields + `label` field).

### Added (tests, 346 → 457)

- **`TestPrfScoresMatchSklearn`** - 14 tests verifying numpy F1/Precision/Recall
  implementation is numerically identical to sklearn across all averaging modes.
- **`test_f1_per_class_extra_matches_sklearn`** - regression test for per-class F1.
- **9 tests** for `ExperimentResult` repr, save, generate_report, CI table mode.
- **8 tests** for `ContainsJudge` - correct/incorrect, case, whitespace, export, CLI.
- **5 tests** for `to_jsonl` round-trip, parents, split-and-save workflow.
- **4 tests** for `ComparisonResult.save` and `MultipleComparisonResult.save`.
- **2 tests** for HTML report light-mode and print CSS.
- **1 test** for invalid `compare(test=...)` parameter.
- **1 test** for `evalkit version` BibTeX output.

---

## [0.3.1] - 2026-05-02

### Fixed

- `MockRunner` wrong outputs are now drawn from the dataset's actual label set
  rather than the opaque `__wrong_N__` strings. This means `F1Score`,
  `PrecisionScore`, and `RecallScore` via `additional_metrics` produce meaningful
  results when used with `MockRunner` - sklearn no longer sees spurious
  out-of-vocabulary prediction strings. Single-label datasets fall back to the
  previous behaviour. Three regression tests added.
- `RigorChecker` PASS label is now `PASS (N warning(s))` when warnings are
  present but no errors. Previously, a plain `PASS` label was shown even when
  the audit contained advisory warnings - users would read "PASS" and stop.
  Applies to both `AuditReport.__str__` and the CLI `_print_audit` display.
  A closing note "Review the warnings above before reporting results." is now
  appended. Three regression tests added.

---

## [0.3.0] - 2026-05-02

### Added

- `PrecisionScore` and `RecallScore` metrics with bootstrap CIs. Both support
  `average="macro"`, `"micro"`, `"weighted"`, and `"binary"`. Exported from
  `evalkit` directly alongside `F1Score` and `BalancedAccuracy`.
- `examples/data/` - two bundled demo datasets (`balanced_demo.jsonl` and
  `underpowered_imbalanced_demo.jsonl`) that demonstrate the RigorChecker
  without requiring API keys. The imbalanced dataset reliably triggers
  `SAMPLE_TOO_SMALL` and `SEVERE_CLASS_IMBALANCE` errors on every run.

---

## [0.2.0] - 2026-04-25

### Added

- `evalkit run --judge llm` - LLM-as-judge from the CLI. Uses the same provider
  as `--model`, so `evalkit run data.jsonl --model gpt-4o-mini --judge llm` runs
  the evaluation and scores each response with the same model. Prints a reminder
  to validate inter-rater agreement.

### Fixed

- `SemanticSimilarityJudge._model` annotated as `SentenceTransformer | None` -
  fixes a mypy strict error on Python 3.12 where the inferred type was `None`.
- `# noqa: E501` comments removed from the HTML template string in `report.py`.
  These linter suppression comments were written verbatim into every generated
  HTML report and rendered as visible text. A regression test now prevents this
  from recurring.
- `# noqa: E501` comments removed from `LLMJudge.DEFAULT_SYSTEM_PROMPT`. These
  were sent verbatim to the language model as part of the evaluation rubric,
  corrupting every evaluation that used the default judge prompt. A regression
  test now prevents recurrence.
- `Experiment.additional_metrics` now passes `run_result.outputs` and
  `run_result.references` to each metric instead of the binary correct/incorrect
  array. `BalancedAccuracy()` and `F1Score()` passed via `additional_metrics` now
  give correct, meaningful results on multi-class and imbalanced data. Previously
  they always equalled `Accuracy` because both predictions and references were
  the same all-ones binary array.
- `evalkit.__version__` now reads from installed package metadata via
  `importlib.metadata` instead of a hardcoded string. This eliminates version
  drift between `pyproject.toml`, `__init__.py`, and `evalkit version` CLI output.
- REST API `/health` endpoint and FastAPI `version` field now report the live
  package version rather than the hardcoded string `"0.1.0"`.
- `test_cli.py::test_version_prints_version_string` updated to compare against
  `evalkit.__version__` dynamically rather than a hardcoded version string.
- `.gitignore` extended to cover local scratch test files.

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
- 398 tests, 100% coverage across 20 source files. Statistical property tests verify
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