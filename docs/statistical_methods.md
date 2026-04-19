# Statistical Methods

Every statistical choice in evalkit has a specific reason. This document records those reasons at a level of precision that lets you defend them in an interview or a methods section.

---

## Confidence Intervals

### What a 95% CI actually means

A 95% CI is a statement about the *procedure*, not the parameter. If you ran this experiment many times and computed a CI each time using this method, 95% of those intervals would contain the true value. The true accuracy is a fixed unknown constant - it's the CI that is random.

**What it does not mean:** "There is a 95% probability the true accuracy is in this interval." Under the frequentist framework that evalkit uses, parameters are not random variables. That interpretation belongs to Bayesian credible intervals, which are a different thing.

### Why percentile bootstrap, not Wilson or BCa

For a proportion, the Wilson interval and the BCa (bias-corrected accelerated) bootstrap are both more accurate than the percentile bootstrap in small samples. evalkit uses the percentile bootstrap anyway, for one reason: **consistency across metrics**.

The Wilson interval applies only to proportions. F1, BalancedAccuracy, and ECE have no tractable analytic CI. Using Wilson for accuracy and bootstrap for everything else creates intervals with subtly different interpretations and different small-sample behaviours. A reader comparing Accuracy CI width to F1 CI width cannot tell whether differences are real or artefacts of different methodology.

Using one method for everything eliminates this ambiguity. The cost is slightly wider CIs for accuracy on small samples - a cost worth paying.

BCa would be a defensible improvement for all metrics. It is not used because it requires computing the jackknife influence function, which is O(n²) for some metrics and adds implementation complexity. For n ≥ 100 (which the RigorChecker enforces), percentile bootstrap error is well below 1%.

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

## Benjamini-Hochberg FDR Correction

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

**"Why not use the BCa bootstrap?"** BCa is more accurate for small n, but it requires the jackknife influence function, which is O(n²) for some metrics. For n ≥ 100 (which the RigorChecker enforces), the percentile bootstrap error is below 1% - well within acceptable range.

**"Why McNemar's not a proportion test?"** Because the observations are paired. The two-proportion z-test and Fisher's exact test assume independent samples. McNemar's uses the pairing information and has higher power than any independence-assuming test on the same data.

**"Why BH not Bonferroni?"** Because Bonferroni controls FWER, which becomes very conservative for K > 5. At K=20, Bonferroni requires p < 0.0025 - power drops below 20% for any reasonable effect size. BH controls FDR, which is the right criterion when you care about the proportion of false discoveries among your claimed winners, not the probability of any false discovery.

**"Why is the kappa threshold 0.60 not 0.70 or 0.80?"** 0.60 is the "substantial agreement" threshold from Landis & Koch (1977), which is the standard reference in NLP evaluation. At κ=0.60, the bootstrap CI on kappa will typically include 0.55-0.65, which is still borderline reliable. We use it as the minimum bar, not the target. For publication-quality results, κ ≥ 0.70 is a better target.

**"Why does `_approx_required_n()` use a normal approximation in `ComparisonResult`?"** When two models are not significantly different, we want to tell the user how many examples would be needed to detect the observed accuracy gap at 80% power. The normal approximation (Lehr's formula) is used here, not McNemar's formula, because we don't know the true discordant proportion - we're providing a rough estimate, not a precise sample size. The note in the output makes this clear.
