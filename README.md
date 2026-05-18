# evalkit

[![CI](https://github.com/bonnie-mcconnell/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/evalkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalkit-research.svg)](https://pypi.org/project/evalkit-research/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bonnie-mcconnell/evalkit/branch/main/graph/badge.svg)](https://codecov.io/gh/bonnie-mcconnell/evalkit)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bonnie-mcconnell/evalkit/main?urlpath=lab/tree/examples/walkthrough.ipynb)

I built this because I kept reading ML papers that reported accuracy improvements with no confidence intervals, no significance testing, and sample sizes that gave them less than 30% statistical power. The improvements were probably noise. Nobody could tell.

The same problem exists in production. Teams ship prompt changes based on n=50 eval results, make model selection decisions where the two models' CIs overlap entirely, and run 20 variants to pick the "best" one - not realising the winner is almost certainly a false positive. The eval tooling computes the numbers correctly. It just never asks whether those numbers are trustworthy.

---

## The problem in one line

```python
from evalkit import Accuracy

predictions = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 20   # 200 examples
references  = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0] * 20

# What everyone does
correct = sum(p == r for p, r in zip(predictions, references))
print(f"Model accuracy: {correct/len(predictions):.2f}")   # → 0.70  (what does this mean?)

# What evalkit does (seed for reproducibility)
print(Accuracy(seed=42).compute(predictions, references))
# → Accuracy: 0.7000 (95% CI: 0.6350–0.7600, n=200)
```

`0.70` at n=200 has a 95% CI of 63.5%–76.0% - a spread of 12.5 percentage points. At n=50, the same measurement gives a CI wider than ±13pp, meaning 70% could plausibly be anywhere from 57% to 83%. The CI tells you the actual precision of your measurement. If two models' CIs overlap substantially, claiming one is "better" is not a finding - it's noise.

```bash
pip install evalkit-research

# Try it immediately - no API keys required:
evalkit run examples/data/balanced_demo.jsonl \
  --model mock \
  --template "{{ text }}" \
  --ref-field label \
  --output report.html
```

---

## What evalkit does

Every metric returns a `MetricResult` - never a bare float. This is enforced at the architecture level: `MetricResult.__post_init__` raises if the point estimate falls outside its own CI bounds, catching bootstrap implementation bugs at the moment they occur rather than silently propagating wrong numbers.

**The `RigorChecker`** is the core feature. Every experiment runs through a two-pass statistical audit: pre-flight (before you spend API budget) and post-hoc (the audit trail you attach to results). It catches underpowered sample sizes, class imbalance inflating accuracy, uncorrected multiple testing, and low LLM judge agreement - the four most common ways LLM evaluation goes wrong.

The bootstrap is **stratified by reference class** - on imbalanced datasets, unstratified resampling occasionally produces resamples with zero minority-class examples, making CIs artificially narrow. The inner loop uses a pure-numpy implementation rather than sklearn (~2.6ms overhead per call × 10,000 resamples = 26s) - the numpy path runs in ~0.1ms per call, so the default 10,000-resample bootstrap completes in under 200ms.

> **For the technical details:** every statistical choice in evalkit - why percentile bootstrap over BCa, why McNemar's over a paired t-test, why BH-FDR over Bonferroni, why stratified bootstrap on imbalanced data - is documented with full derivations and design rationale in **[docs/statistical_methods.md](docs/statistical_methods.md)**.

Try it immediately without API keys using the bundled demo datasets:

```bash
# PASS (1 warning) - balanced classes, no errors; warning about comparison power
evalkit run examples/data/balanced_demo.jsonl \
  --model mock --template "{{ text }}" --ref-field label \
  --output balanced_report.html

# FAIL - fires SAMPLE_TOO_SMALL + SEVERE_CLASS_IMBALANCE errors
evalkit run examples/data/underpowered_imbalanced_demo.jsonl \
  --model mock --template "{{ text }}" --ref-field label \
  --output imbalanced_report.html
```

```
╔══════════════════════════════════════════════════════╗
║           evalkit  RigorChecker  Report              ║
╚══════════════════════════════════════════════════════╝
Experiment: my_prompt_experiment
Status: FAIL  (2 errors)

🔴 [SAMPLE_TOO_SMALL] Sample size n=28 is below the absolute minimum (30).
   No metric is meaningful at this scale.
   → Collect at least 30 examples. For useful accuracy estimates, aim for N≥200.

🔴 [SEVERE_CLASS_IMBALANCE] Your test set is 93% class 'positive'.
   A trivial model predicting the majority class achieves 93% accuracy.
   Accuracy is a meaningless metric here.
   → Report balanced accuracy, macro-F1, or AUC instead of accuracy.
```

| Feature | evalkit | lm-eval-harness | DeepEval | RAGAS | LangSmith |
|---------|:-------:|:---------------:|:--------:|:-----:|:---------:|
| Bootstrap CI on every metric (automatic, always-on) | ✅ | Opt-in¹ | ❌ | ❌ | ❌ |
| Paired significance test (McNemar / Wilcoxon) | ✅ | ❌ | ❌ | ❌ | ❌ |
| BH-FDR correction for prompt variants | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pre-flight power analysis (before spending budget) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Automated non-optional statistical audit | ✅ | ❌ | ❌ | ❌ | ❌ |
| Inter-rater agreement validation (κ, α) | ✅ | ❌ | Partial | ❌ | ❌ |
| CI-gated comparison (.compare() with required N) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Works offline / no vendor lock-in | ✅ | ✅ | ✅ | ✅ | ❌ |

¹ lm-evaluation-harness supports bootstrap resampling via `--bootstrap_iters` but it is opt-in, applied only to aggregate task scores (not individual metrics), and produces no power analysis, significance tests, or audit trail. evalkit's CIs are automatic on every `MetricResult` and cannot be turned off.

*Verified May 2026 against lm-evaluation-harness 0.4.x, DeepEval 1.x, RAGAS 0.2.x, LangSmith. If any framework has added these features since, please open an issue.*

---

## Quickstart

### Try it right now (60 seconds, no API key)

```bash
git clone https://github.com/bonnie-mcconnell/evalkit
cd evalkit
pip install -e .                              # core install, no extras needed for mock
evalkit run examples/data/balanced_demo.jsonl --model mock --template "{{ text }}" --ref-field label
evalkit table
```

You'll see a results table with 95% bootstrap CIs and a RigorChecker audit.
For real model evaluation, see the OpenAI and Anthropic sections below.

### Using a real model (OpenAI)

```bash
pip install "evalkit-research[openai]"
export OPENAI_API_KEY=sk-...

# Validate a factual QA dataset (~$0.01, 50 examples, ~20 seconds)
evalkit run examples/data/factual_qa_50.jsonl \
  --model gpt-4o-mini \
  --template "Answer in one word or short phrase: {{ question }}" \
  --ref-field answer \
  --judge contains

# Or run the full Python quickstart with error analysis and HTML report
python examples/openai_quickstart.py --output report.html
```

What to expect when you run it:

- **Accuracy** roughly 88–96% with a wide CI (±10–14pp at n=50) - the wide CI is
  correct behaviour, not a bug. 50 examples is not enough for a precise measurement.
- **RigorChecker** will warn about CI precision. It will tell you exactly how many
  more examples you need.
- **Cost** under $0.01 total.
- **Time** approximately 15–30 seconds.

`evalkit power 0.05 --test ci` shows the required N for any target precision.

### Using a real model (Anthropic)

```bash
pip install "evalkit-research[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...

evalkit run examples/data/factual_qa_50.jsonl \
  --model claude-haiku-4-5 \
  --template "Answer in one word or short phrase: {{ question }}" \
  --ref-field answer \
  --judge contains
```

### Your own data

Format your data as JSONL (one JSON object per line):

```jsonl
{"id": "q1", "question": "What is the capital of France?", "label": "Paris"}
{"id": "q2", "question": "What year did WWII end?", "label": "1945"}
```

Then run:

```bash
evalkit run my_data.jsonl \
  --model gpt-4o-mini \
  --template "{{ question }}" \
  --ref-field label \
  --judge contains
```

See [examples/data/README.md](examples/data/README.md) for the full JSONL format
specification, judge selection guide, and common mistakes.

---

### Step 0: figure out how many examples you need

Run this before labelling any data or spending any API budget.

```python
from evalkit import PowerAnalysis

pa = PowerAnalysis(alpha=0.05, power=0.80)
ci = pa.for_ci_precision(desired_half_width=0.05)
print(f"Report accuracy to ±5%: need n ≥ {ci.minimum_n}")  # → n ≥ 323

cmp = pa.for_proportion_difference(effect_size=0.05)
print(f"Detect 5pp difference: need n ≥ {cmp.minimum_n}")  # → n ≥ 1,251
```

Or get the full planning table:

```python
from evalkit import PowerAnalysis

pa = PowerAnalysis(alpha=0.05)
pa.sample_size_table()
```
```
Sample Size Requirements  (α=0.05, two-tailed)
Test: TwoProportionZ
Baseline accuracy: 0.70

 Effect size │  Power 70%   Power 80%   Power 90%
─────────────┼────────────────────────────────────
      Δ = 2% │    6,354     8,080    10,816
      Δ = 5% │      984     1,251     1,674
     Δ = 10% │      231       294       392
     Δ = 15% │       96       121       161
     Δ = 20% │       49        62        82
```

Or from the CLI: `evalkit table`

### Step 1: validate your setup before spending budget

```python
from evalkit import EvalDataset, PromptTemplate

dataset = EvalDataset.from_jsonl("my_data.jsonl")
template = PromptTemplate("Answer concisely: {{ question }}")

# Check all template variables exist in the dataset - zero API cost
errors = template.validate(dataset)
if errors:
    print("\n".join(errors))   # fix before running
```

Template validation catches field-name typos before you've made a single API call. Without this, you discover the problem at example 0 after the asyncio setup and retry logic have all initialised.

### Step 2: run the evaluation

> **Note:** On balanced data where the model makes errors uniformly across classes,
> `BalancedAccuracy` and `F1Score` equal `Accuracy` - this is correct, not a bug.
> They diverge on **imbalanced data or when the model has class-biased errors**.
> The `underpowered_imbalanced_demo.jsonl` dataset demonstrates this clearly.

```python
from evalkit import (
    EvalDataset, PromptTemplate, ExactMatchJudge, MockRunner, Experiment,
    BalancedAccuracy, F1Score, PrecisionScore, RecallScore,
    PreFlightError,
)

dataset = EvalDataset.from_jsonl("my_data.jsonl")
template = PromptTemplate("Answer concisely: {{ question }}")
runner = MockRunner(judge=ExactMatchJudge(), template=template)

# strict=True (default): pre-flight ERRORs raise PreFlightError before any
# API calls are made - no budget is spent on a broken experiment design.
# Set strict=False to run anyway and rely on the post-hoc audit instead.
try:
    result = Experiment(
        "my_eval", dataset, runner,
        additional_metrics=[
            BalancedAccuracy(),
            F1Score(average="macro"),
            PrecisionScore(average="macro"),
            RecallScore(average="macro"),
        ],
    ).run()
except PreFlightError as e:
    print(e.audit)      # full RigorChecker report
    raise               # or fix the issue and re-run

result.print_summary()
```

```
============================================================
Experiment: my_eval
============================================================
Dataset: my_data (n=1000)
Model:   mock-model-v1
Cost:    $0.0000 | Tokens: 0 | Time: 0.8s

Metrics (with 95% bootstrap CIs):
  Accuracy: 0.8230 (95% CI: 0.7990–0.8470, n=1000)
  BalancedAccuracy: 0.8230 (95% CI: 0.7990–0.8460, n=1000)
  F1Score(macro): 0.8229 (95% CI: 0.7989–0.8459, n=1000)
  Precision(macro): 0.8236 (95% CI: 0.8000–0.8467, n=1000)
  Recall(macro): 0.8230 (95% CI: 0.7990–0.8460, n=1000)

✅ RigorChecker: No issues found. Experiment appears statistically sound.
```

### Step 2b: save results and generate a tearsheet

```python
# Save to JSON - load later for comparisons or CI pipelines
result.save("results/my_eval.json")

# Generate a self-contained HTML tearsheet (no external deps)
# Includes: metrics table with CI bars, RigorChecker audit findings,
# error analysis (up to 10 worst examples with output vs expected), and run details.
result.generate_report("results/my_eval.html")

# Or get the HTML string directly (e.g. to embed in a notebook)
html = result.generate_report()

# Save a comparison result (useful for PR attachments)
comparison = result_a.compare(result_b)
comparison.save("results/comparison.json")

# Commit your train/test split to disk so it never changes
train, test = dataset.split(test_size=0.2, stratify=True)
train.to_jsonl("data/train.jsonl")
test.to_jsonl("data/test.jsonl")
```

### Step 3: compare models correctly

```python
result_a = Experiment("gpt-4o", dataset, runner_a).run()
result_b = Experiment("gpt-4o-mini", dataset, runner_b).run()

comparison = result_a.compare(result_b)
print(comparison)
```

```
Comparison: gpt-4o vs gpt-4o-mini
  gpt-4o:      accuracy = 0.8460
  gpt-4o-mini: accuracy = 0.7140
  McNemar: stat=24.28, p=0.0000, effect=2.21 → REJECT H₀ (α=0.05)
  ✓ gpt-4o is statistically better (p=0.0000)
```

`compare()` auto-selects McNemar's test for binary outcomes (exact match) or Wilcoxon signed-rank for continuous scores (LLM judge). It verifies both result sets contain the same examples in the same order before running - misaligned paired tests are the most common error in LLM model comparison.

When the result is not significant, it tells you exactly what to do:

```
The difference is NOT statistically significant. Increase N to ≥1,251 to detect this effect.
```

### Step 4: error analysis

```python
# The examples the model most confidently got wrong
worst = result_a.worst_examples(10)
for ex in worst:
    print(f"{ex['output']!r}  →  expected {ex['reference']!r}")

# Full per-example results as a pandas DataFrame
# Requires: pip install "evalkit-research[dataframe]"
df = result_a.to_dataframe()
wrong = df[~df["is_correct"]]
print(wrong[["example_id", "output", "reference", "reasoning"]].head())
```

Aggregate accuracy tells you how often the model is wrong. `worst_examples()` tells you *how* it's wrong - the confidently incorrect answers that reveal systematic failure modes.

### Step 5: multiple testing correction

When you're comparing K prompt variants, the probability of at least one false positive grows with K. At K=20 and α=0.05, you'd expect one false positive by chance even with identical prompts.

```python
from evalkit import BHCorrection

result = BHCorrection(alpha=0.05).correct(
    p_values=[0.003, 0.041, 0.068, 0.24, 0.51],
    comparison_names=["v1", "v2", "v3", "v4", "v5"],
)
# v2: p_raw=0.041 → p_adj=0.103  ← would be a false positive without correction
```

### Step 6: validate your LLM judge

```python
from evalkit import CohenKappa

# Run your judge twice on the same examples and measure agreement
result = CohenKappa().compute(judge_run_1_scores, judge_run_2_scores)
print(result)
# ✗ CohenKappa: 0.4100 (95% CI: 0.31–0.51) - below minimum threshold (0.60)
# → RigorChecker will flag this as LOW_JUDGE_AGREEMENT
```

κ < 0.60 means the judge is inconsistent enough that the scores are unreliable. The RigorChecker will flag it as an ERROR if you proceed.

---

## Dataset utilities

```python
# Split your labelled pool before evaluating
# stratify=True preserves class distribution in both subsets
train, test = dataset.split(test_size=0.2, stratify=True)
print(f"Train: {len(train)}, Test: {len(test)}")

# Sample for quick prototyping (RigorChecker will flag the small N)
small = dataset.sample(50)

# Load from different sources
dataset = EvalDataset.from_jsonl("data.jsonl")
dataset = EvalDataset.from_csv("data.csv")
dataset = EvalDataset.from_huggingface(  # requires: pip install "evalkit-research[huggingface]"
    "squad", split="validation", reference_field="answers"
)
dataset = EvalDataset.from_list(records, reference_field="label")
```

---

## CLI

```bash
# Evaluate with exact-match judge (output must exactly equal label)
evalkit run data.jsonl --model gpt-4o-mini \
  --template "Q: {{ question }}" \
  --output report.html

# Evaluate with contains judge (output must contain the label string)
evalkit run data.jsonl --model gpt-4o-mini \
  --judge contains \
  --template "Q: {{ question }}"

# Evaluate with LLM-as-judge (same model scores the responses)
evalkit run data.jsonl --model gpt-4o-mini \
  --judge llm \
  --template "Q: {{ question }}"

# Machine-readable output for scripting
evalkit run data.jsonl --model mock --format json | jq '.metrics.Accuracy.value'
evalkit run data.jsonl --model mock --format json > results.json

# Planning table (before labelling any data)
evalkit table
evalkit table --test ci --baseline 0.80

# Power analysis for a specific experiment
evalkit power 0.05                          # N needed to detect 5% accuracy gain
evalkit power 0.05 --observed-n 150        # power achieved at n=150
evalkit power 0.03 --test ci               # N for ±3% CI half-width

# Compare two saved runs with significance testing
evalkit run data.jsonl --model gpt-4o --save-results gpt4o.json
evalkit run data.jsonl --model gpt-4o-mini --save-results mini.json
evalkit compare gpt4o.json mini.json --test mcnemar

# Version
evalkit version
```

---

## REST API

```bash
docker compose up api
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

Endpoints: `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/report`, `POST /compare`, `POST /power`, `GET /health`. Results are written to `./results/` and persist across restarts.

> **Note:** The REST API currently supports `MockRunner` only - it is designed for integration testing and CI pipelines. For production evaluations with real model providers (OpenAI, Anthropic), use the Python API or CLI directly. Real provider support in the REST API is tracked in the [GitHub issues](https://github.com/bonnie-mcconnell/evalkit/issues).

---

## Using evalkit in CI pipelines

evalkit is designed to gate model deployments on statistical quality. Two flags make this work:

**`evalkit run --fail-on-errors`** - exits with code 1 if the RigorChecker audit finds ERROR-level findings (underpowered sample, severe class imbalance, uncorrected multiple testing). Use this to block a deployment when your eval data is too small to trust.

**`evalkit compare --fail-on-regression`** - exits with code 2 if the new model is statistically significantly *worse* than the baseline. Exit 0 means no significant difference or the new model is better. Use this to block a deployment when the new model regresses.

**`evalkit compare --format json`** - machine-readable output for scripting: `jq .reject_null`, `jq .b_is_significantly_worse`, `jq .p_value`.

Example GitHub Actions workflow:

```yaml
# .github/workflows/eval.yml
name: Eval gate

on:
  pull_request:
    paths: ["prompts/**", "models/**"]

jobs:
  eval:
    runs-on: ubuntu-latest
> **For the technical details:** every statistical choice in evalkit - why percentile bootstrap over BCa, why McNemar's over a paired t-test, why BH-FDR over Bonferroni, why stratified bootstrap on imbalanced data - is documented with full derivations and design rationale in **[docs/statistical_methods.md](docs/statistical_methods.md)**. This is the document to read before an interview.

    steps:
      - uses: actions/checkout@v4
      - name: Install evalkit
The README will be updated with real output numbers after live validation.
        run: pip install "evalkit-research[openai]"

      - name: Run eval on baseline (main branch result cached)
        run: |
          evalkit run data/eval.jsonl \
            --model gpt-4o-mini \
            --template "{{ question }}" \
            --save-results baseline.json \
            --fail-on-errors          # block if sample is too small to trust
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Run eval on new model/prompt
        run: |
          evalkit run data/eval.jsonl \
            --model gpt-4o-mini \
            --template "{{ question }} Think step by step." \
            --save-results new_model.json \
            --fail-on-errors
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Compare - block if new model regresses
        run: |
          evalkit compare baseline.json new_model.json \
            --fail-on-regression      # exit 2 if new model is significantly worse
          # exit 0 = no regression detected (safe to deploy)
```

The `--fail-on-errors` flag catches the most common mistake in LLM eval CI pipelines: running a comparison on a dataset that is too small to detect meaningful differences, and treating a non-significant result as evidence of equivalence.

---

## Architecture

The thing I spent the most time on was making sure the statistical methods couldn't be bypassed. The four-layer architecture enforces this:

**Data layer** (`EvalDataset`, `PromptTemplate`): Jinja2 with `StrictUndefined` - referencing an undefined template variable raises immediately rather than silently rendering empty. `PromptTemplate.validate(dataset)` lets you catch this before any API calls. Example IDs are not cosmetic; they're the mechanism by which `compare()` and McNemar's test verify that two result sets are actually aligned.

**Execution layer** (`Judge`, `AsyncRunner`, `MockRunner`): `MockRunner` sits at the runner level rather than the provider level because "correct" is only meaningful relative to the reference answer, which the provider layer doesn't have access to. `AsyncRunner` writes checkpoints atomically (write-to-temp-then-rename) so a killed process never leaves a corrupt checkpoint. Synchronous provider calls run in a thread pool via `run_in_executor` to avoid blocking the event loop.

**Analysis layer** (`Metric`, `RigorChecker`): The bootstrap is implemented once in `Metric.bootstrap_ci` and called by every subclass via `_point_estimate`. Stratified bootstrap samples within each class separately so all classes appear in every resample - on an imbalanced dataset, unstratified resampling sometimes produces zero examples of the minority class, making CIs artificially narrow. The `RigorChecker` clamps accuracy to the open interval (0, 1) before running power calculations - accuracy of exactly 0 or 1 makes the variance term p*(1-p) degenerate to zero, producing a misleading "adequately powered" result.

The inner bootstrap loop calls `_prf_scores` - a pure-numpy precision/recall/F1 implementation - rather than `sklearn.metrics`. sklearn adds ~2.6ms of Python overhead per call (input validation, type coercion, format dispatch). At 10,000 resamples that totals ~26 seconds. The numpy path runs in ~0.1ms per call, a 20× speedup that makes the default `n_resamples=10_000` practical: the full default bootstrap completes in under 200ms on n=1,000 examples. The benchmark is in `evalkit/metrics/accuracy.py` at the `_prf_scores` docstring.

**Interface layer** (Python API, CLI, REST API): Three interfaces over the same objects. The CLI uses lazy imports for fast startup - `evalkit version` doesn't load scipy. `evalkit run --format json` produces pipeable output for scripting.

---

## Statistical methods

Detailed derivations, design rationale, and interview-ready explanations: [docs/statistical_methods.md](docs/statistical_methods.md).

### Bootstrap confidence intervals

Percentile bootstrap with 10,000 resamples (configurable). Stratified by reference class for all classification metrics. With B=1,000 the Monte Carlo error on a 95% CI endpoint is roughly ±0.7%; with B=10,000 it's ±0.2%. For the quick-iteration case, pass `n_resamples=1000`.

### McNemar's test

The correct test for paired binary outcomes. Chi-squared with Edwards' continuity correction, which matters for small samples - without it the test is anti-conservative. Effect size is the odds ratio (Laplace-smoothed to handle zero counts). Concordant pairs (both models right or both wrong) are correctly ignored; only discordant pairs carry information about which model is better.

### Wilcoxon signed-rank test

For continuous quality scores (LLM judge ratings). Non-parametric - no normality assumption, which is appropriate for bounded, skewed LLM scores. Effect size is rank-biserial correlation r ∈ [−1, 1].

### Benjamini-Hochberg FDR correction

For K ≥ 2 prompt variants. Controls the expected false discovery rate at α, which is less conservative than Bonferroni (family-wise error rate) and has much higher power for large K. The implementation enforces monotonicity on adjusted p-values via the step-down procedure.

### Cohen's kappa / Krippendorff's alpha

For validating LLM-as-judge reliability. κ < 0.60 triggers a `RigorChecker` ERROR. Krippendorff's alpha handles multiple raters, missing data, and ordinal scales. The 0.60 threshold comes from Landis & Koch (1977).

### Power analysis

Four methods: CI precision (how wide will my CI be?), two-proportion z-test (can I detect a Δ% accuracy difference?), McNemar's (how many pairs do I need for the exact paired test?), Wilcoxon (Cohen's d for continuous scores). Always run pre-flight.

### Expected Calibration Error

The last confidence bin uses an inclusive upper boundary so scores of exactly 1.0 are always counted. This is a subtle but consistent bug in most ECE implementations.

---

## Installation

```bash
pip install evalkit-research                    # core
pip install "evalkit-research[openai]"          # + OpenAI provider
pip install "evalkit-research[anthropic]"       # + Anthropic provider
pip install "evalkit-research[api]"             # + REST API server
pip install "evalkit-research[generation]"      # + BLEU / ROUGE
pip install "evalkit-research[agreement]"       # + Krippendorff's alpha
pip install "evalkit-research[huggingface]"     # + HuggingFace Datasets
pip install "evalkit-research[dataframe]"       # + pandas (to_dataframe())
pip install "evalkit-research[semantic]"        # + SemanticSimilarityJudge (sentence-transformers)
pip install "evalkit-research[all]"             # everything
```

Requires Python 3.11+. Fully typed (`py.typed` marker, mypy strict).

---

## Running the demo

### With a real API key (recommended first step)

```bash
git clone https://github.com/bonnie-mcconnell/evalkit
cd evalkit
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
python examples/openai_quickstart.py
```

Runs 50 factual QA questions against `gpt-4o-mini`, produces bootstrap CIs, a
RigorChecker audit, cost summary, and error analysis. Total cost under $0.02.
See `examples/openai_quickstart.py` for full documentation and `--output report.html`
to generate an HTML tearsheet.

### Mock demos (no API key)

```bash
pip install -e ".[dev]"
python examples/full_workflow.py
```

Runs the complete pipeline - power analysis, evaluation, model comparison, FDR
correction, judge validation, RigorChecker audit, HTML tearsheet, error analysis,
dataset splitting, template validation - using a mock model.

```bash
python examples/benchmark_audit.py
```

Audits the five most common evaluation failure modes: underpowered measurements
(n=50, ±12pp CI), class imbalance inflation, multiple testing false positives
(3 variants look significant → 0 survive FDR), low judge agreement (κ=0.41),
and a well-designed passing experiment - all in one run.

Or open the **interactive notebook** in your browser - no install required:

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bonnie-mcconnell/evalkit/main?urlpath=lab/tree/examples/walkthrough.ipynb)

The notebook walks through the same pipeline step-by-step with explanations and live output.

---

## References

- Efron & Hastie (2016). *Computer Age Statistical Inference*. Ch. 11 - bootstrap.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS-B*.
- McNemar (1947). Sampling error of correlated proportions. *Psychometrika*.
- Wilcoxon (1945). Individual comparisons by ranking methods. *Biometrics*.
- Guo et al. (2017). On calibration of modern neural networks. *ICML*.
- Landis & Koch (1977). The measurement of observer agreement. *Biometrics*.

---

MIT License · [Changelog](CHANGELOG.md) · [Statistical methods](docs/statistical_methods.md) · [Design decisions](docs/design_decisions.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)
