# Statistical Methods

Every statistical choice in evalkit has a specific reason. This document records those reasons at a level of precision suitable for a methods section or code review.

---

## Confidence Intervals

### What a 95% CI actually means

A 95% CI is a statement about the *procedure*, not the parameter. If you ran this experiment many times and computed a CI each time using this method, 95% of those intervals would contain the true value. The true accuracy is a fixed unknown constant - it's the CI that is random.

**What it does not mean:** "There is a 95% probability the true accuracy is in this interval." Under the frequentist framework that evalkit uses, parameters are not random variables. That interpretation belongs to Bayesian credible intervals, which are a different thing.

### Why percentile bootstrap, not Wilson or BCa

For a proportion, the Wilson interval and the BCa (bias-corrected accelerated) bootstrap are both more accurate than the percentile bootstrap in small samples. evalkit uses the percentile bootstrap anyway, for one reason: **consistency across metrics**.

The Wilson interval applies only to proportions. F1, BalancedAccuracy, and ECE have no tractable analytic CI. Using Wilson for accuracy and bootstrap for everything else creates intervals with subtly different interpretations and different small-sample behaviours. A reader comparing Accuracy CI width to F1 CI width cannot tell whether differences are real or artefacts of different methodology.

Using one method for everything eliminates this ambiguity. The cost is slightly wider CIs for accuracy on small samples - a cost worth paying.

BCa would be a defensible improvement for all metrics. It is not used because it requires computing the jackknife influence function, which is O(n²) for some metrics and adds implementation complexity. For n ≥ 100 - above the sample-size floor used by RigorChecker - percentile bootstrap Monte Carlo error is well below 1%.

### Why stratified resampling

The standard percentile bootstrap samples n examples with replacement from the full dataset. On an imbalanced dataset (90% class A, 10% class B), some resamples will contain zero class-B examples. For BalancedAccuracy, which is the mean of per-class recalls, a resample with no class-B examples makes the class-B recall undefined and the bootstrap statistic meaningless.

Stratified bootstrap samples within each class separately, guaranteeing all classes appear in every resample. This is equivalent to the stratified random sample used in survey methodology.

The trade-off: for text generation metrics (BLEU, ROUGE), there is no meaningful notion of "class." Those use unstratified bootstrap.

### Monte Carlo error and the choice of B=10,000

The percentile bootstrap is itself a Monte Carlo estimator. The standard error of a percentile estimate is approximately σ / (2φ(zₚ)√B), where φ is the standard normal density. For a 95% CI endpoint (zₚ ≈ 1.96), this gives:

| B | MC error on CI endpoint |
|---|------------------------|
| 100 | ±2.3% |
| 1,000 | ±0.7% |
| 10,000 | ±0.2% |

For results reported to two decimal places, B=10,000 keeps MC error below the last reported digit. `n_resamples=1000` is offered for iteration speed; results should not be published at that setting.

---

## McNemar's Test

### Exact null hypothesis

H₀: the probability that model A is correct and model B is not equals the probability that model B is correct and model A is not. Formally: P(A correct, B wrong) = P(A wrong, B correct).

This is a test of **marginal homogeneity** in the 2×2 table of outcomes. It is *not* a test of whether the two accuracy proportions are equal - those tests (two-proportion z-test, Fisher's exact test) assume independence. McNemar's uses the pairing.

### Why McNemar's and not a paired t-test

The paired t-test requires the differences (score_A_i - score_B_i) to be approximately normally distributed. For binary outcomes (0 or 1), the differences can only be -1, 0, or +1. This is a three-point discrete distribution - not approximately normal for any sample size. McNemar's is the exact correct test.

### Why concordant pairs are discarded

Examples where both models are right, or both are wrong, carry no information about *which model is better*. They only tell you the overall difficulty of the examples. Including them in the test statistic would dilute the signal. McNemar's correctly conditions on discordant pairs.

Consequence: if two models have near-identical outputs, the test has very low power regardless of sample size. This is expected behaviour - if the models disagree on almost no examples, there is genuinely little information to distinguish them.

### Edwards' continuity correction

The chi-squared approximation to the exact binomial test is slightly anti-conservative without correction. Edwards' correction (subtracting 1 from |b - c|) brings the chi-squared approximation closer to the exact test, particularly at small discordant-pair counts.

Without the correction, the test rejects H₀ too often at small n, producing spurious significant results. We always apply it.

### Laplace smoothing on the odds ratio

The odds ratio effect size is (b + 0.5) / (c + 0.5). The +0.5 is Laplace smoothing, which handles the case where b=0 or c=0 (giving an infinite or zero odds ratio). This is standard practice in contingency table analysis.

---

## Wilcoxon Signed-Rank Test

### When to use it instead of McNemar's

Use Wilcoxon when quality scores are continuous - for example, LLM-as-judge scores on [0, 1]. McNemar's requires binary (correct/incorrect) outcomes. evalkit selects the test automatically based on whether scores are all in {0.0, 1.0}.

LLM scores are typically bounded, right-skewed, and non-normal. The Wilcoxon test makes no distributional assumption and has 95.5% asymptotic efficiency relative to the t-test under normality - and higher efficiency under non-normality.

### Effect size: rank-biserial correlation

The rank-biserial correlation r = 1 − (2W / n(n+1)) where W is the Wilcoxon test statistic. It lies in [−1, 1]:

- r > 0: model A tends to score higher
- r < 0: model B tends to score higher
- |r| ≈ 0.1: small, 0.3: medium, 0.5: large (Cohen's benchmarks)

Unlike Cohen's d, rank-biserial r makes no distributional assumption.

---

## Post-hoc Sample Size Estimation

### When evalkit shows a required-N recommendation

When a comparison is not statistically significant (`reject_null=False`), evalkit shows an estimated minimum N required to detect the observed effect at 80% power. This appears in `ComparisonResult.__str__()` and in JSON output as `approx_required_n`.

### Why a two-proportion z-test formula, not the exact paired test formula

The exact power formula for McNemar's test requires knowing the discordant-pair fraction φ = (n₁₂ + n₂₁) / n - the proportion of examples on which the two models *disagree*. The exact formula for Wilcoxon requires knowing the within-pair score correlation ρ. Neither quantity is stored in `ComparisonResult`; they are properties of the raw run data.

Rather than require the user to pass raw data into a post-hoc display method, evalkit uses the two-proportion z-test formula as a computationally available approximation:

```
n ≈ ((z_α √(2p̄(1-p̄)) + z_β √(p_A(1-p_A) + p_B(1-p_B))) / Δ)²
```

where Δ = |p_A − p_B|, p̄ = (p_A + p_B)/2, z_α = 1.96 (two-tailed α=0.05), z_β = 0.842 (80% power).

### Direction of the approximation error

The two-proportion z-test ignores the within-pair correlation of outcomes. For paired tests, this correlation *always* reduces variance - pairing never hurts. Therefore:

- The two-proportion formula *overestimates* the required N relative to the exact paired test.
- The displayed N is a **conservative upper bound**: the true required N for a paired McNemar or Wilcoxon test will be equal or lower.
- The approximation never tells you that you need *fewer* examples than you actually do.

This conservative direction is the correct failure mode for a sample size warning. Overcounting required N wastes budget; undercounting leads to under-powered experiments that produce false negatives.

### Magnitude of the conservatism

For typical evaluation settings (p_A ≈ 0.75, p_B ≈ 0.70, φ ≈ 0.20 discordant pairs), McNemar's exact required N is roughly 60–75% of the two-proportion z-test N. The displayed number is approximately 1.3–1.7× the true required N. For planning purposes, this means collecting the displayed N guarantees adequate power.

---



### The multiple testing problem, precisely stated

If you perform K independent tests each at significance level α, the probability of at least one false positive under the complete null hypothesis is 1 − (1 − α)^K. For K=20, α=0.05, this is 64%.

### FWER vs FDR

**Family-wise error rate (FWER)**: probability of *any* false positive. Bonferroni controls FWER by testing each hypothesis at α/K. With K=20, α=0.05, the per-comparison threshold is 0.0025 - extremely conservative.

**False discovery rate (FDR)**: expected proportion of false positives among all rejected hypotheses. BH controls FDR at level q. If you reject m hypotheses, the expected number of false discoveries is ≤ q × m.

For prompt engineering (comparing K=10-50 variants), FWER control is too conservative - it makes true improvements very hard to detect. FDR control at q=0.05 means that at most 5% of your declared winners are expected to be false positives. This is the right trade-off.

### The BH step-up procedure

1. Sort p-values ascending: p(1) ≤ p(2) ≤ ... ≤ p(K)
2. Find the largest i such that p(i) ≤ (i/K) × q
3. Reject H₀ for all j ≤ i

Adjusted p-values: p̃(i) = min_{j ≥ i} p(j) × (K/j), then enforce monotonicity step-down (p̃(i) = min(p̃(i), p̃(i+1))).

### What evalkit specifically flags

evalkit flags the case that changes conclusions: when unadjusted analysis says "this comparison is significant" but BH-corrected analysis says "it is not." This is the researcher-relevant failure mode.

---

## Cohen's Kappa

### Why not raw percent agreement

Raters who both label "correct" with probability p and "incorrect" with probability (1-p) will agree by chance with probability p² + (1-p)². For p=0.8 (80% accuracy), this is 0.64 + 0.04 = 68% - substantial apparent agreement with zero actual coordination. Kappa subtracts this chance baseline.

### Interpretation thresholds (Landis & Koch 1977)

| κ | Interpretation |
|---|---|
| < 0.00 | Poor (worse than chance) |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

evalkit uses 0.60 as the minimum threshold for a judge to be considered reliable. This is the widely-used threshold in NLP evaluation literature. Below κ=0.60, two runs of the same judge might produce different model rankings - the judge is the bottleneck, not the models.

### Why bootstrap CI on kappa

Kappa has an analytic asymptotic SE, but it requires the large-sample assumption and is inaccurate for small n or extreme kappa values. The bootstrap makes no distributional assumptions and is accurate for any n ≥ 30.

---

## Krippendorff's Alpha

### When to prefer it over kappa

Krippendorff's alpha (α_K) generalises kappa to:
- **More than two raters**: kappa is defined for exactly two raters. α_K handles any number.
- **Missing data**: some raters may not have scored all items. α_K handles this gracefully.
- **Ordinal and continuous scales**: kappa treats all disagreements equally. α_K uses a distance metric appropriate to the measurement level (nominal, ordinal, interval, ratio).

For LLM judge scores on [0, 1] with multiple runs, α_K with `level_of_measurement="interval"` is the correct choice.

### Bootstrap over items, not rater-item pairs

Items (evaluation examples) are the independent observations. Rater-item pairs within the same item are correlated. The bootstrap must resample over items (columns of the reliability matrix), not individual ratings. evalkit does this correctly; many implementations do not.

---

## Power Analysis

### Why power matters in LLM evaluation

A study with 30% power will miss a real effect 70% of the time. The researcher who concludes "the models are equivalent" after a 30% power study has not found equivalence - they've found nothing.

The conventional target of 80% power means accepting a 20% chance of missing a real effect. This is a deliberate trade-off between sample size cost and detection reliability.

### The four methods

**`for_ci_precision(desired_half_width w)`**

Solves for n in: w = z_{1−α/2} × √(p(1−p)/n)

n = (z_{1−α/2} / w)² × p(1−p)

This is a planning tool, not a hypothesis test. It answers: "how many examples do I need to report accuracy to ±w%?"

**`for_proportion_difference(effect_size Δ, p1)`**

Uses the two-proportion z-test formula (Lehr's approximation):
n ≈ (z_{1−α/2}√(2p̄(1−p̄)) + z_β√(p₁(1−p₁) + p₂(1−p₂)))² / Δ²

where p₂ = p₁ + Δ and p̄ = (p₁ + p₂)/2.

**`for_mcnemar(effect_size OR, discordant_proportion π)`**

The required number of discordant pairs is:
n_d = ((z_{α/2}√(0.25) + z_β√(p_A(1−p_A))) / |p_A − 0.5|)²

where p_A = OR / (1 + OR) is the probability that model A wins given a discordant pair. The total N then follows from n_d / π.

**`for_wilcoxon(cohens_d)`**

Wilcoxon has 95.5% asymptotic relative efficiency (ARE) compared to the t-test. So the required n for Wilcoxon is approximately n_t / 0.955, where n_t is the t-test requirement at the same power:

n_t = ((z_{α/2} + z_β) / d)²

---

## Expected Calibration Error

### Definition

ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|

where bins partition confidence scores into equal-width intervals, acc(B_b) is the accuracy of examples in bin b, and conf(B_b) is their mean confidence.

A well-calibrated model has ECE ≈ 0. ECE > 0.15 indicates the confidence scores are not useful for decision-making.

### The last-bin boundary bug

The standard binning scheme uses `confidence < upper_bound` for all bins. This means confidence scores of exactly 1.0 are dropped from all bins silently. evalkit uses `confidence <= upper_bound` for the last bin, ensuring 100%-confident predictions are counted.

This is a small correction but it matters: many models output exactly 1.0 for trivial inputs, and silently dropping those examples produces a systematically biased ECE estimate.

---

## Design decisions defensible under pressure

**"Why not use the BCa bootstrap?"** BCa is more accurate for small n, but it requires the jackknife influence function, which is O(n²) for some metrics. For n ≥ 100, the percentile bootstrap Monte Carlo error is below 1% - well within acceptable range. (The RigorChecker's hard minimum is n=30; UNDERPOWERED_CI fires at n<100 for CI precision tasks.)

**"Why McNemar's not a proportion test?"** Because the observations are paired. The two-proportion z-test and Fisher's exact test assume independent samples. McNemar's uses the pairing information and has higher power than any independence-assuming test on the same data.

**"Why BH not Bonferroni?"** Because Bonferroni controls FWER, which becomes very conservative for K > 5. At K=20, Bonferroni requires p < 0.0025 - power drops below 20% for any reasonable effect size. BH controls FDR, which is the right criterion when you care about the proportion of false discoveries among your claimed winners, not the probability of any false discovery.

**"Why is the kappa threshold 0.60 not 0.70 or 0.80?"** 0.60 is the "substantial agreement" threshold from Landis & Koch (1977), which is the standard reference in NLP evaluation. At κ=0.60, the bootstrap CI on kappa will typically include 0.55-0.65, which is still borderline reliable. We use it as the minimum bar, not the target. For publication-quality results, κ ≥ 0.70 is a better target.

**"Why does `_approx_required_n()` use a normal approximation in `ComparisonResult`?"** When two models are not significantly different, we want to tell the user how many examples would be needed to detect the observed accuracy gap at 80% power. The normal approximation (Lehr's formula) is used here, not McNemar's formula, because we don't know the true discordant proportion - we're providing a rough estimate, not a precise sample size. The note in the output makes this clear.

---

## Practical interview questions answered

**"What does your RigorChecker actually check, specifically?"**

Eight distinct failure modes in priority order:

1. `SAMPLE_TOO_SMALL` - n < 30. Below this the bootstrap is unreliable (too few resamples to estimate tail quantiles) and the CI is uninformative regardless of width.
2. `UNDERPOWERED_CI` - the CI half-width is > 0.05 (5 percentage points). You're reporting accuracy to two decimal places but the measurement is not that precise.
3. `CLASS_IMBALANCE` - the majority class exceeds 75% of examples. Accuracy is inflated relative to minority-class performance. Warning level: results require caveats.
4. `SEVERE_CLASS_IMBALANCE` - the majority class exceeds 90% of examples. A trivial majority-class predictor achieves that accuracy; accuracy is essentially meaningless. Error level: results should not be reported without switching metrics.
5. `MULTIPLE_TESTING_UNCORRECTED` - K ≥ 2 comparisons were made and at least one unadjusted p-value is < alpha, but the BH-adjusted p-value is not. Reporting the unadjusted result is a false positive.
6. `UNDERPOWERED_COMPARISON` - two models are compared but the sample size is below the minimum needed to detect the observed accuracy gap at 80% power.
7. `LOW_JUDGE_AGREEMENT` - Cohen's κ or Krippendorff's α for the judge is below 0.60. Every score this judge produces is unreliable.
8. `JUDGE_AGREEMENT_REQUIRED` - an LLM judge (stochastic) was used but no inter-rater agreement measurement has been recorded in the audit. Running the judge twice on a sample and computing κ is recommended before publishing results.


**"Why 90% as the SEVERE_CLASS_IMBALANCE threshold, and why is there a two-tier system?"**

The two-tier system reflects different degrees of harm:

- **75% threshold (`CLASS_IMBALANCE`, WARNING)**: A dataset that is 75% majority class means a constant-positive predictor achieves 75% accuracy. This is inflated but not necessarily higher than a real model. Results need caveats and `macro-F1` alongside accuracy, but accuracy is not meaningless - report both.

- **90% threshold (`SEVERE_CLASS_IMBALANCE`, ERROR)**: At 90% majority, a zero-effort majority classifier scores 90%. A real model that achieves 88% is technically *worse* than the trivial baseline, yet would be reported as impressive without this check. Accuracy here is actively misleading. The threshold is deliberately set at 90% rather than 80% or 85% - to avoid firing on naturally skewed but tractable datasets (product review sentiment is typically 70-80% positive; that should be a warning, not a hard stop).

**"What does `_point_estimate` vs `compute` separate in the Metric abstraction?"**

`_point_estimate(predictions, references)` computes the statistic on one set of arrays - this is the method subclasses implement. `compute(predictions, references)` calls `_point_estimate` for the observed value, then calls it B times on bootstrap resamples, and wraps everything in a `MetricResult`. The separation means implementing a new metric requires writing one method (not the bootstrap loop), and the bootstrap is guaranteed to be applied consistently across all metrics.

**"Why are `predictions` and `references` passed as `np.ndarray` to `_point_estimate` if users pass `list[Any]`?"**

Because `_point_estimate` needs to accept indexing for bootstrap resampling (`predictions[idx]`). Plain lists don't support fancy indexing. The conversion happens once at the top of `compute()` so subclasses don't have to think about it.

**"If you were at Atlassian evaluating a new LLM-powered feature, what would your eval setup look like?"**

This is the translation to production. The setup I'd use:

1. **Sample size first**: run `pa.for_ci_precision(desired_half_width=0.03)` with my expected accuracy. At 0.70 accuracy, that's n=897 to report accuracy to ±3%. I'd label that many examples before writing any evaluation code.

2. **Stratified split**: use `dataset.split(test_size=0.2, stratify=True)` so class distribution matches production. Commit the test split - never change it.

3. **Judge validation**: if using an LLM judge, run it twice on 100 examples and compute κ. If κ < 0.60, fix the judge prompt before any evaluation. This step is almost always skipped in practice and almost always matters.

4. **Baseline first**: evaluate the current production model on the full test set. This is the number you need to beat, with its CI.

5. **McNemar's not accuracy comparison**: `result_new.compare(result_old)`. Same 854 examples, paired test, verified alignment. A new model needs to win on the *same* examples, not just have higher overall accuracy.

6. **BH correction**: if comparing multiple prompt variants, collect all p-values and pass them to `BHCorrection`. Report only adjusted p-values.

7. **Attach the `AuditReport` to the PR**: the `posthoc_audit` from `Experiment.run()` goes into the PR description. Green board means the result is defensible. Red board means the experiment doesn't ship.

**"What's the difference between statistical significance and practical significance?"**

Statistical significance says: given the null hypothesis (no difference), this result is unlikely to have occurred by chance. Practical significance says: the difference is large enough to matter in production.

You can have statistical significance without practical significance: with n=10,000, a 0.3pp accuracy difference is statistically significant but operationally irrelevant. You can also have practical significance without statistical significance: with n=30, a 12pp difference might be practically huge but the CI is so wide you can't tell if it's real.

evalkit's power analysis addresses both. `for_ci_precision` tells you whether your CI is narrow enough to make practical claims. McNemar's tests whether the difference is real. Run both.

---

## Choosing a judge

**"When should I use each judge type?"**

The correct judge depends on what "correct" means for your task:

`ExactMatchJudge` - the output must exactly equal the reference (after optional case normalisation). Use for tasks with a single canonical answer: multiple-choice (A/B/C/D), classification labels (positive/negative), closed-form factual answers (a specific date or name). This is the most conservative judge and should be the default for any task where exact equality is meaningful.

`ContainsJudge` - the output must contain the reference as a substring. Use when the model's output is a sentence or paragraph that should include a key phrase, but the exact wording around it is acceptable. Example: the reference is "Paris" and the model says "The capital of France is Paris, population 2.1M" - that is correct under ContainsJudge, wrong under ExactMatchJudge. More lenient; can inflate accuracy if the reference is a very short string.

`RegexMatchJudge` - the output must match a regex pattern, with optional group extraction. Use for structured outputs where the format matters: "Answer: A", "Confidence: 0.95", "Label: <category>". More precise than ContainsJudge and handles format validation in one step.

`SemanticSimilarityJudge` - cosine similarity of sentence embeddings, correct if similarity ≥ threshold. Use for tasks where paraphrase is acceptable: summarisation, translation, open-ended QA where "Paris" and "the French capital" should both be counted correct. Requires `pip install "evalkit-research[semantic]"`. The threshold (default 0.85) needs calibration on your specific domain - validate with CohenKappa before trusting scores.

`LLMJudge` - calls an LLM to score the response against the reference and returns a structured score + reasoning. Use for tasks where human-level semantic understanding is required: instruction following, reasoning quality, factual accuracy in long-form answers. Most expensive, most powerful, most unreliable without validation. Always measure inter-rater agreement (κ ≥ 0.60) before reporting results.

**Rule of thumb:** start with ExactMatchJudge. If too strict, try ContainsJudge. If output format is structured, try RegexMatchJudge. If semantic equivalence matters, try SemanticSimilarityJudge. Only use LLMJudge when the task genuinely requires human-level judgment and you have budget for validation.
