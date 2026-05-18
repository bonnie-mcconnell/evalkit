# evalkit: Design Decisions

This document records the significant architectural and statistical choices made in
evalkit, with the reasoning behind each decision and the trade-offs accepted. It is
intended for contributors, researchers evaluating the library for adoption, and anyone
who wants to understand why the library works the way it does rather than just what it does.

The decisions documented here were choices, not oversights. Where a different approach
would be clearly better in a future version, that is noted explicitly.

---

## Statistical choices

### Percentile bootstrap over bias-corrected accelerated (BCa)

**Decision:** evalkit uses the percentile bootstrap for all confidence intervals.

**Why not BCa?** BCa is more accurate than the percentile bootstrap for small samples and
skewed distributions - particularly for accuracy measured on n < 100. It corrects for both
bias (the bootstrap distribution's centre differs from the true parameter) and skewness
(the bootstrap distribution is not symmetric). In those settings, BCa intervals have better
coverage probability than percentile intervals.

**Why percentile bootstrap instead:**

1. *Consistency across metrics.* evalkit computes bootstrap CIs for Accuracy,
   BalancedAccuracy, F1Score, PrecisionScore, RecallScore, and any user-defined metric.
   BCa requires computing influence functions via jackknife resampling - an O(n²) operation
   that is well-defined for simple statistics but requires careful implementation for
   multi-class metrics. Using the same method for all metrics makes the library's behaviour
   predictable and testable.

2. *Computational cost.* The jackknife for BCa runs n leave-one-out resamples on top of
   the B bootstrap resamples. At n=1000, B=10000, that's 1000 additional full-dataset
   metric computations. For the numpy-vectorised inner loop this is manageable, but for
   custom metrics that call into sklearn it could multiply runtime by 10×.

3. *Coverage at production scale.* At n ≥ 200 - the minimum evalkit's RigorChecker
   recommends for any serious measurement - the coverage difference between BCa and
   percentile bootstrap is small (typically < 1 percentage point). The practical argument
   for BCa is strongest at small n, exactly where the RigorChecker is already flagging
   that your measurement is not precise enough to act on.

**Trade-off accepted:** Users running evalkit on n < 100 (which the RigorChecker flags as
an error) will get slightly overconfident intervals. This is acceptable because the
RigorChecker tells them the measurement is not trustworthy regardless.

**Future direction:** Offer BCa as an opt-in for users who explicitly need it on small
samples and are willing to accept the performance cost.

---

### Stratified bootstrap resampling

**Decision:** Bootstrap resamples are stratified by reference class.

**Why:** On imbalanced datasets, unstratified resampling occasionally produces resamples
that contain no minority-class examples. When this happens, multi-class metrics
(BalancedAccuracy, F1Score, RecallScore) are undefined or degenerate - sklearn raises
a warning and returns 0 for the missing class. The CI computed from these resamples is
artificially narrow because the missing-class resamples produce anomalous metric values,
not because the estimate is actually precise.

Stratified resampling preserves the class distribution in every resample. The CI width
is then determined by genuine sampling uncertainty, not by the bootstrap occasionally
producing pathological resamples.

**Trade-off accepted:** Stratified resampling uses slightly more memory (must store the
class index before resampling). The computational overhead is negligible.

---

### McNemar's test as the default paired comparison

**Decision:** When two models are evaluated on the same dataset, the default significance
test is McNemar's test.

**Why McNemar's over a paired t-test:**
The paired t-test on binary correctness vectors assumes normally distributed differences.
Binary outcomes (correct/incorrect) cannot be normally distributed. This assumption
violation inflates Type I error - you will reject the null hypothesis too often. McNemar's
test operates on the 2×2 contingency table of discordant pairs and makes no distributional
assumption. For LLM evaluation on classification tasks, it is the correct test.

**Why offer Wilcoxon as an alternative:**
For continuous-score judges (semantic similarity, ROUGE, etc.) where correctness is not
binary, McNemar's test does not apply. Wilcoxon signed-rank operates on the ranks of the
differences and makes no normality assumption, making it appropriate for continuous scores
from soft judges.

**Why not Mann-Whitney U:**
Mann-Whitney U tests unpaired samples. If two models are run on the same examples, the
outcomes are paired - the correct test uses the pairing. Mann-Whitney discards this
structure and loses statistical power.

---

### Benjamini-Hochberg FDR over Bonferroni FWER correction

**Decision:** Multiple testing correction uses Benjamini-Hochberg false discovery rate
control rather than Bonferroni family-wise error rate control.

**Why:** Bonferroni controls the probability that any null hypothesis is incorrectly
rejected. For k comparisons, it divides α by k - at k=10 and α=0.05, each test must
achieve p < 0.005 to be declared significant. This is extremely conservative. In the
context of prompt engineering, where teams routinely test 5-20 variants, Bonferroni
correction makes it nearly impossible to detect any real difference, even at n=1000.

Benjamini-Hochberg controls the expected proportion of false discoveries among rejected
hypotheses. At k=10, it allows some false discoveries but keeps the false discovery rate
at α=0.05. In practice: if you test 10 variants and find 3 significant differences, you
expect at most 0.15 of those to be false positives - you expect at least ~2.85 of your
findings to be real.

For prompt engineering with many variants, controlling FDR is more appropriate than
controlling FWER. The cost of missing a real improvement is higher than the cost of
occasionally including one false discovery in a batch of significant results.

**Trade-off accepted:** In confirmatory settings where one pre-specified hypothesis is
being tested (e.g., a regulatory submission), FWER control is more appropriate.
evalkit does not prevent users from applying Bonferroni correction manually - `BHCorrection`
is a named class they can replace with their own correction function.

---

## Architecture choices

### Synchronous provider calls wrapped in thread pool (not async SDK)

**Decision:** `AsyncRunner` wraps synchronous provider `_call()` methods in
`loop.run_in_executor(None, ...)` rather than using the providers' native async clients.

**Why:** This decision keeps the provider abstraction clean. A `ModelProvider` subclass
only needs to implement `_call()` - one synchronous function that takes messages and
returns a `ProviderResponse`. The async concurrency logic lives entirely in `AsyncRunner`,
not scattered across every provider implementation. Adding a new provider (Cohere, Mistral,
a local Ollama endpoint) requires implementing one synchronous function, not an async one.

**Trade-off accepted:** The OpenAI and Anthropic SDKs both have native async clients
(`openai.AsyncOpenAI`, `anthropic.AsyncAnthropic`) that use true async I/O. Using those
directly would avoid thread-pool overhead and allow the SDK's own async retry logic. For
the concurrency levels evalkit uses (5-20 concurrent calls), the thread-pool overhead is
negligible - the network latency dominates. At very high concurrency (50+ calls), native
async would be meaningfully faster.

**Future direction:** Add `AsyncModelProvider` as an optional interface for providers
that offer native async clients, allowing `AsyncRunner` to call them directly without a
thread pool.

---

### Checkpoint format uses `str(reference)`, losing type information

**Decision:** The checkpoint format serialises reference labels as `str(reference)`.

**Why:** Checkpoint files are JSONL. JSON has no native type system for arbitrary Python
objects - a reference that is an integer `1` and a reference that is the string `"1"` both
serialise to `1` in JSON and round-trip as `"1"` when loaded. Implementing a full typed
schema (storing `{"value": "1", "dtype": "int"}` alongside each reference) would
increase checkpoint complexity significantly.

**What this means in practice:** Judges that call `str()` on the reference are safe.
`ExactMatchJudge` compares `str(output) == str(reference)`, so round-trip through string
is transparent. `ContainsJudge` and `RegexMatchJudge` also work on strings. Only custom
judges that rely on the reference's Python type (e.g., a judge that does
`isinstance(reference, int)`) would behave incorrectly after a checkpoint resume.

**Trade-off accepted:** Users with custom typed judges must be aware that checkpoint
resume re-runs their judge with string references. This is documented in the `AsyncRunner`
docstring.

**Future direction:** Add optional typed serialisation (`{"v": ..., "t": "int|float|list"}`)
for users who need round-trip type fidelity.

---

### REST API supports MockRunner only

**Decision:** The REST API (`evalkit/api/app.py`) does not accept real provider credentials
and cannot run evaluations against OpenAI or Anthropic.

**Why:** API key management is a security surface that does not belong in a stateless HTTP
endpoint without authentication. A REST API that accepts `{"model": "gpt-4o-mini",
"api_key": "sk-..."}` in a request body is storing or proxying a credential - which
requires TLS enforcement, key rotation handling, audit logging, and rate limit attribution.
The CLI is the right interface for real provider evaluations: it runs on the user's machine,
reads keys from the environment, and the key never leaves the user's control.

The REST API exists for integration testing and CI pipelines that use the mock provider -
it lets you run evalkit in a Docker container without environment variable management.

**Future direction:** Accept provider credentials via environment variables set at container
startup, not per-request. Never accept keys in request bodies.

---

### `Accuracy` is the only default metric (not `BalancedAccuracy`)

**Decision:** `Experiment` computes `Accuracy` by default. `BalancedAccuracy`, `F1Score`,
`PrecisionScore`, and `RecallScore` are opt-in via `additional_metrics`.

**Why not default to BalancedAccuracy:**
Not all evaluation tasks are classification tasks. Some use continuous-score judges
(semantic similarity, ROUGE) where class-balance concepts do not apply. Making
`BalancedAccuracy` the default would produce confusing output for non-classification tasks.

**Why not auto-detect and prompt:**
Auto-detection based on label cardinality (e.g., "if fewer than 20 unique labels, assume
classification") would be heuristic and occasionally wrong. Explicit opt-in is more honest.

**Trade-off accepted:** Users evaluating imbalanced classification tasks without explicitly
adding `BalancedAccuracy()` will see only `Accuracy`, which can be inflated by predicting
the majority class. The `RigorChecker` fires `SEVERE_CLASS_IMBALANCE` when the majority
class exceeds 80% of examples, which will direct the user to consider `BalancedAccuracy`.

**Future direction:** When `SEVERE_CLASS_IMBALANCE` fires, suggest adding
`BalancedAccuracy()` in the audit finding's `action` field rather than just warning about
the imbalance.

---

### frozen=True dataclass with mutable `_comparison_p_values`

**Decision:** `ExperimentResult` is a `frozen=True` dataclass, but it contains a
`list[float]` field (`_comparison_p_values`) that is mutated after construction via
`.append()`.

**Why:** `frozen=True` prevents attribute *reassignment* (`self._comparison_p_values = []`)
but does not prevent mutation of the object an attribute references
(`self._comparison_p_values.append(x)` is always valid). This is standard Python
semantics. The pattern is used here because `audit_comparisons()` needs to accumulate
p-values from every `.compare()` call the user makes after the experiment runs - those
calls happen at user-controlled times that cannot be known at construction.

The alternative - a separate `ComparisonTracker` object that the user must pass to every
`.compare()` call - would require changing the public API of `.compare()` and introducing
an object whose lifecycle is not obvious. The current design keeps `.compare()` simple and
auditing automatic.

`default_factory=list` ensures each instance gets its own list. There is no shared mutable
state between `ExperimentResult` instances.

---

## Testing choices

### 100% coverage enforced at CI level

**Decision:** CI runs `pytest --cov-fail-under=100` on every push.

**Why:** evalkit makes statistical guarantees - the bootstrap CI, the pre-flight halt,
the BH correction. Statistical bugs are silent. A wrong CI does not throw an exception;
it returns a plausible-looking number that happens to be wrong. A user who acts on a
miscalculated CI designs a real experiment around a false precision estimate. The cost
of that bug is not a crashed program - it is bad science.

100% coverage does not guarantee correctness. It guarantees that every code path
executes under test. For a library making numerical guarantees, the minimum standard
is that every branch has been observed to run at least once. Branches that are never
executed are candidates for silent numerical errors that only surface under specific
input conditions - exactly the conditions a user is most likely to discover in production,
not in development.

The specific enforcement at CI level rather than as a developer guideline matters: it
makes coverage degradation impossible to merge accidentally. A new feature that adds
an untested branch fails CI. The coverage constraint is a forcing function for writing
tests at the time of writing code, not as an afterthought.

**Trade-off accepted:** 100% coverage is expensive to maintain and slows feature
development. This is intentional. A statistical library that ships fast with untested
numerical paths is worse than one that ships slowly with every path exercised. The
users of this library are making decisions about real model quality; they deserve
the slower, more careful approach.

---

## Decisions not yet made

The following are open questions that have not been resolved in the current version:

- **Async-native providers:** See above. The interface for this is not yet designed.
- **BCa bootstrap opt-in:** The API for opting in is not designed. A `bootstrap_method`
  parameter on `Accuracy` (and other metrics) is the obvious approach.
- **REST API real provider support:** Requires authentication design before implementation.
- **Typed checkpoint format:** Backwards-compatible with the current string format; the
  migration path is straightforward but not yet implemented.
