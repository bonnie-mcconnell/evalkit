# evalkit

[![CI](https://github.com/bonnie-mcconnell/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/bonnie-mcconnell/evalkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalkit-research.svg)](https://pypi.org/project/evalkit-research/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bonnie-mcconnell/evalkit/branch/main/graph/badge.svg)](https://codecov.io/gh/bonnie-mcconnell/evalkit)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bonnie-mcconnell/evalkit/main?filepath=examples/walkthrough.ipynb)

I built this because I kept reading ML papers that reported accuracy improvements with no confidence intervals, no significance testing, and sample sizes that gave them less than 30% statistical power. The improvements were probably noise. Nobody could tell.

The specific thing that annoyed me enough to build a library: you can run 20 prompt variants, find one that achieves 74% vs 71%, ship it, and never notice that at n=50 examples, a 3-point difference is almost certainly random. Existing evaluation frameworks (lm-evaluation-harness, DeepEval, RAGAS) compute metrics correctly but don't ask whether those metrics are trustworthy. evalkit makes the statistical audit automatic and non-optional.

```bash
pip install evalkit-research
evalkit run data.jsonl --model mock --template "Q: {{ question }}"
```

---

## The problem in one line

```python
# What everyone does
print(f"Model accuracy: {correct/total:.2f}")          # → 0.73  (what does this mean?)

# What evalkit does
print(Accuracy().compute(predictions, references))     # → Accuracy: 0.7300 (95% CI: 0.6804–0.7785, n=200)
```

`0.73` without a CI could be anywhere from 0.60 to 0.88 on 50 examples. The CI tells you the actual precision of your measurement. If two models' CIs overlap substantially, claiming one is "better" is not a finding - it's noise.

---

## What evalkit does

Every metric returns a `MetricResult` - never a bare float. This is enforced at the architecture level: `MetricResult.__post_init__` raises if the point estimate falls outside its own CI bounds, catching bootstrap implementation bugs at the moment they occur rather than silently propagating wrong numbers.

**The `RigorChecker`** is the core feature. Every experiment runs through a two-pass statistical audit: pre-flight (before you spend API budget) and post-hoc (the audit trail you attach to results). It catches underpowered sample sizes, class imbalance inflating accuracy, uncorrected multiple testing, and low LLM judge agreement - the four most common ways LLM evaluation goes wrong.

```
╔══════════════════════════════════════════════════════╗
║           evalkit  RigorChecker  Report              ║
╚══════════════════════════════════════════════════════╝
Experiment: my_prompt_experiment
Status: FAIL  (2 errors, 1 warning)

🔴 [UNDERPOWERED_CI] Your sample size (n=47) gives a CI half-width of
   ±0.071 (±7.1%), not the ±0.050 you may be implying by reporting to
   two decimal places.
   → To achieve ±5% precision, you need n≥196.

🔴 [MULTIPLE_TESTING_UNCORRECTED] 2 comparison(s) appear significant
   without FDR correction but are NOT significant after Benjamini-Hochberg
   correction. Reporting uncorrected results would be misleading.
   → Report only BH-adjusted p-values.

🟡 [CLASS_IMBALANCE] Your test set is 82% class 'correct'. Accuracy is
   inflated relative to minority-class performance.
   → Report macro-F1 alongside accuracy.
```

| Feature | evalkit | lm-eval-harness | DeepEval | RAGAS | LangSmith |
|---------|:-------:|:---------------:|:--------:|:-----:|:---------:|
| Bootstrap CI on every metric | ✅ | ❌ | ❌ | ❌ | ❌ |
| McNemar's test for model comparison | ✅ | ❌ | ❌ | ❌ | ❌ |
| BH-FDR correction for prompt variants | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pre-flight power analysis | ✅ | ❌ | ❌ | ❌ | ❌ |
| Automated statistical audit | ✅ | ❌ | ❌ | ❌ | ❌ |
| Inter-rater agreement (κ, α) | ✅ | ❌ | Partial | ❌ | ❌ |
| Direct model comparison (.compare()) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Works offline / no vendor dependency | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## Quickstart

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

Or get the full planning table - screenshot this and put it in your design doc:

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

```python
from evalkit import EvalDataset, PromptTemplate, ExactMatchJudge, MockRunner, Experiment

dataset = EvalDataset.from_jsonl("my_data.jsonl")
template = PromptTemplate("Answer concisely: {{ question }}")
runner = MockRunner(judge=ExactMatchJudge(), template=template)

result = Experiment("my_eval", dataset, runner).run()
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
  Accuracy: 0.8230 (95% CI: 0.8010–0.8440, n=1000)

✅ RigorChecker: No issues found. Experiment appears statistically sound.
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
  gpt-4o:      accuracy = 0.8400
  gpt-4o-mini: accuracy = 0.7200
  McNemar: stat=9.08, p=0.0026, effect=2.71 → REJECT H₀ (α=0.05)
  ✓ gpt-4o is statistically better (p=0.0026)
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
# Evaluate with exact-match judge
evalkit run data.jsonl --model gpt-4o-mini \
  --template "Q: {{ question }}" \
  --output report.html

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

---

## Architecture

The thing I spent the most time on was making sure the statistical methods couldn't be bypassed. The four-layer architecture enforces this:

**Data layer** (`EvalDataset`, `PromptTemplate`): Jinja2 with `StrictUndefined` - referencing an undefined template variable raises immediately rather than silently rendering empty. `PromptTemplate.validate(dataset)` lets you catch this before any API calls. Example IDs are not cosmetic; they're the mechanism by which `compare()` and McNemar's test verify that two result sets are actually aligned.

**Execution layer** (`Judge`, `AsyncRunner`, `MockRunner`): `MockRunner` sits at the runner level rather than the provider level because "correct" is only meaningful relative to the reference answer, which the provider layer doesn't have access to. `AsyncRunner` writes checkpoints atomically (write-to-temp-then-rename) so a killed process never leaves a corrupt checkpoint. Synchronous provider calls run in a thread pool via `run_in_executor` to avoid blocking the event loop.

**Analysis layer** (`Metric`, `RigorChecker`): The bootstrap is implemented once in `Metric.bootstrap_ci` and called by every subclass via `_point_estimate`. Stratified bootstrap samples within each class separately so all classes appear in every resample - on an imbalanced dataset, unstratified resampling sometimes produces zero examples of the minority class, making CIs artificially narrow. The `RigorChecker` clamps accuracy to the open interval (0, 1) before running power calculations - accuracy of exactly 0 or 1 makes the variance term p*(1-p) degenerate to zero, producing a misleading "adequately powered" result.

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
pip install "evalkit-research[all]"             # everything
```

Requires Python 3.11+. Fully typed (`py.typed` marker, mypy strict).

---

## Running the demo

```bash
git clone https://github.com/bonnie-mcconnell/evalkit
cd evalkit
pip install -e ".[dev]"
python examples/full_workflow.py
```

This runs the complete pipeline - power analysis, evaluation, model comparison, FDR correction, judge validation, RigorChecker audit, HTML tearsheet, error analysis, dataset splitting, template validation - using a mock model with no API keys.

---

## What I'd do differently

The `Experiment._compute_metrics` method only computes accuracy by default. For real use you almost always want F1 or BalancedAccuracy too, especially on imbalanced datasets. I'd make those the default rather than opt-in. (As of v0.2.0, `additional_metrics` now correctly passes `outputs` and `references` to each metric, so `BalancedAccuracy()` and `F1Score()` work properly via that parameter.)

The REST API only supports `MockRunner`. Adding real provider support would require API key management in the API layer, which I deliberately deferred - the security surface is non-trivial and the CLI already handles real providers well.

The checkpoint format serialises references as `str(reference)`, losing type information across restarts. A proper implementation would use a typed schema. Judges that call `str()` on the reference are safe; others are not.

---

## References

- Efron & Hastie (2016). *Computer Age Statistical Inference*. Ch. 11 - bootstrap.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS-B*.
- McNemar (1947). Sampling error of correlated proportions. *Psychometrika*.
- Wilcoxon (1945). Individual comparisons by ranking methods. *Biometrics*.
- Guo et al. (2017). On calibration of modern neural networks. *ICML*.
- Landis & Koch (1977). The measurement of observer agreement. *Biometrics*.

---

MIT License · [Changelog](CHANGELOG.md) · [Statistical methods](docs/statistical_methods.md) · [Cite](CITATION.cff) · [Security](SECURITY.md)