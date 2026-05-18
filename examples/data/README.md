# evalkit datasets

This directory contains datasets for demonstrating and validating evalkit.

---

## JSONL data format

evalkit reads datasets from JSONL files (one JSON object per line). Understanding
the format is necessary before running evalkit on your own data.

### Required fields

Every line must be a valid JSON object. The only required field is the **reference
field** - the ground truth label your model's output will be judged against.

By default, evalkit looks for a field named `label`. You can use any field name
with `--ref-field <name>` on the CLI or `reference_field=` in `EvalDataset.from_jsonl()`.

### Template fields

Your prompt template (`--template`) can reference any other field in the JSON object
using Jinja2 syntax: `{{ field_name }}`. evalkit validates that every field your
template references exists in every example before making any API calls.

### Optional fields

- **`id`**: A string identifier for the example. If missing, evalkit assigns a
  sequential integer ID. IDs are used in `worst_examples()` output and checkpoint
  filenames - stable IDs make debugging much easier.
- Any other fields are passed through to the template and stored in the run result.

### Minimal valid example

```jsonl
{"id": "q1", "question": "What is 2+2?", "label": "4"}
{"id": "q2", "question": "What is the capital of France?", "label": "Paris"}
```

Run with:
```bash
evalkit run my_data.jsonl \
  --model gpt-4o-mini \
  --template "Answer in one word: {{ question }}" \
  --ref-field label \
  --judge contains
```

### Multi-field templates

You can use multiple fields in a single template:

```jsonl
{"id": "1", "context": "The Eiffel Tower is in Paris.", "question": "Where is the Eiffel Tower?", "label": "Paris"}
```

Template: `"Context: {{ context }}\n\nQuestion: {{ question }}\n\nAnswer:"`

### What judge to use

| Situation | Judge | CLI flag |
|---|---|---|
| Model output should equal the reference exactly | ExactMatchJudge | `--judge exact` |
| Model output should contain the reference string | ContainsJudge | `--judge contains` |
| Model output should match a regex pattern | RegexMatchJudge | `--judge regex --regex "pattern"` |
| Open-ended output scored by another LLM | LLMJudge | `--judge llm` |

**Use `--judge contains` for most factual QA tasks.** Model output is rarely exactly
equal to a reference string - "The answer is Paris" does not match "Paris" with exact
match, but does match with contains. ContainsJudge checks whether the reference
appears anywhere in the model's output (case-insensitive by default).

### Common mistakes

**Wrong reference field name.** If your file has `{"answer": "Paris"}` but you run
with `--ref-field label`, evalkit raises a `KeyError`. Use `--ref-field answer`.

**Template references a missing field.** If your template is `{{ question }}` but your
file has `{"query": "..."}`, evalkit will catch this at validation time (before any
API calls) and tell you which fields are missing.

**Inconsistent field names across rows.** evalkit validates the template against every
example. If row 47 is missing a field that all other rows have, the validation error
will name the example ID.

---

## Included datasets

### factual_qa_50.jsonl

50 factual questions with unambiguous short answers across 5 categories: geography,
science, history, language, and mathematics.

**Purpose:** Live API validation. Run this with a real OpenAI or Anthropic key to
confirm the provider path works end-to-end. Expected accuracy with `gpt-4o-mini` and
`--judge contains`: approximately 88–96%.

All answers are multi-character strings designed to avoid ContainsJudge false positives:
element symbol questions ask for the element name ("gold", "iron") not the chemical
symbol ("Au", "Fe"); math questions use shape names ("octagon", "triangle", "pentagon")
and word-form numbers ("twelve"); questions where a short answer would appear as a
substring of unrelated words were replaced. Cost to run: under $0.01.

**Fields:** `id`, `question`, `answer`, `category`

**Reference field:** `answer`

```bash
# With a real API key (costs ~$0.01-0.02 total)
evalkit run examples/data/factual_qa_50.jsonl \
  --model gpt-4o-mini \
  --template "Answer in one word or short phrase: {{ question }}" \
  --ref-field answer \
  --judge contains

# With mock model (no API key needed)
evalkit run examples/data/factual_qa_50.jsonl \
  --model mock \
  --template "{{ question }}" \
  --ref-field answer
```

**Note on the RigorChecker output:** at n=50 the RigorChecker fires a warning
about CI precision. 50 examples gives a wide CI (around +/-12-14pp), which is
intentional - the dataset is sized for cheap API validation, not
publication-quality results. To get +/-5pp precision you need n>=200.

---

### balanced_demo.jsonl

200 examples, balanced 50/50 between `positive` and `negative` labels.

A mock-only demonstration of a reasonably designed evaluation. The RigorChecker
passes with one warning (comparison power at n=200 is ~33% for a 5pp difference
- honest and expected).

**Fields:** `id`, `text`, `label`

```bash
evalkit run examples/data/balanced_demo.jsonl \
  --model mock \
  --template "{{ text }}" \
  --ref-field label
```

---

### underpowered_imbalanced_demo.jsonl

28 examples, 93% `positive` / 7% `negative`.

Demonstrates two simultaneous evaluation failures: `SAMPLE_TOO_SMALL` (n=28 gives
+/-16pp CI) and `SEVERE_CLASS_IMBALANCE` (93% majority class inflates accuracy).
The RigorChecker will FAIL with errors. That is the point.

```bash
evalkit run examples/data/underpowered_imbalanced_demo.jsonl \
  --model mock \
  --template "{{ text }}" \
  --ref-field label \
  --no-strict
```

(`--no-strict` is required here because `SAMPLE_TOO_SMALL` is an ERROR-level finding
that aborts the run by default. Use `--no-strict` when you want to see results despite
known issues.)
