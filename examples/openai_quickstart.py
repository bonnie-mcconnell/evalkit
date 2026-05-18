"""
openai_quickstart.py - evalkit live API validation example
===========================================================

This script demonstrates evalkit running a real evaluation against the OpenAI API.
It uses the factual_qa_50.jsonl dataset (50 questions with known answers) and
produces a full audit report with bootstrap CIs and a RigorChecker assessment.

Requirements
------------
    pip install "evalkit-research[openai]"
    export OPENAI_API_KEY=sk-...   # or set in .env file

Usage
-----
    python examples/openai_quickstart.py

    # Save the HTML tearsheet
    python examples/openai_quickstart.py --output report.html

    # Use a different model
    python examples/openai_quickstart.py --model gpt-4o

    # Run via CLI instead (equivalent)
    evalkit run examples/data/factual_qa_50.jsonl \\
      --model gpt-4o-mini \\
      --template "Answer in one word or short phrase: {{ question }}" \\
      --ref-field answer \\
      --judge contains

Cost estimate
-------------
50 examples at gpt-4o-mini pricing: safely under $0.01 total.
(Input ~50 tokens/call including system overhead, output ~10-30 tokens/call.)

Expected output (approximate - model responses vary slightly between runs)
--------------------------------------------------------------------------
    ✓ Loaded 50 examples from factual_qa_50
    Running evaluation against gpt-4o-mini...
    (50 examples, ~5 concurrent requests, expect 15-30 seconds)

    ============================================================
    Results - factual_qa_50_gpt-4o-mini
    ============================================================
      Accuracy             0.XXXX   95% CI [0.XXX, 0.XXX]   ±0.XXX

    RigorChecker: ⚠  PASS with N warning(s)
      [CI_TOO_WIDE] Your n=50 gives a CI half-width of ±Xpp ...
      → Collect n≥200 for ±5pp precision.

    Cost:  $0.000X  |  Tokens: X,XXX  |  50 API calls  |  XX.Xs
    Model: gpt-4o-mini

Note: Exact numbers depend on the model run. The README will be updated with
real output after live validation. The X placeholders above are intentional -
this docstring shows the output *structure*, not fabricated numbers.

Note: The CI will be wide (±10-14pp) at n=50. This is correct - 50 examples is
not enough for a precise measurement. The RigorChecker will tell you exactly how
many more you need. This is the tool working as intended.

Note on accuracy variation: factual QA accuracy with gpt-4o-mini varies slightly
between runs due to temperature and model updates. The expected range is 88-96%.
If you see accuracy significantly outside this range, check that the JSONL file
is unmodified and that you are using the correct --ref-field.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(model: str = "gpt-4o-mini", output_path: Path | None = None) -> None:
    # --- Import checks ---
    try:
        import openai  # noqa: F401
    except ImportError:
        print(
            "ERROR: openai package is not installed.\n"
            'Install it with: pip install "evalkit-research[openai]"\n',
            file=sys.stderr,
        )
        sys.exit(1)

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "Set it with:\n"
            "  export OPENAI_API_KEY=sk-...        (macOS/Linux)\n"
            "  $env:OPENAI_API_KEY = 'sk-...'      (Windows PowerShell)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- evalkit imports ---
    from evalkit import (
        EvalDataset,
        Experiment,
        PromptTemplate,
    )
    from evalkit.core.experiment import PreFlightError
    from evalkit.core.judge import ContainsJudge
    from evalkit.core.runner import AsyncRunner
    from evalkit.providers.base import OpenAIProvider

    # --- Locate dataset ---
    # Works whether run from the repo root or from examples/
    this_dir = Path(__file__).parent
    dataset_path = this_dir / "data" / "factual_qa_50.jsonl"
    if not dataset_path.exists():
        print(
            f"ERROR: Dataset not found at {dataset_path}\n"
            "Make sure you are running this from the evalkit repo root or examples/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load dataset ---
    # reference_field="answer" because factual_qa_50.jsonl uses "answer" not "label"
    dataset = EvalDataset.from_jsonl(dataset_path, reference_field="answer")
    print(f"✓ Loaded {len(dataset)} examples from {dataset.name}")

    # --- Build provider and runner ---
    # ContainsJudge: scores 1.0 if the reference answer appears anywhere in the
    # model output (case-insensitive). More robust than ExactMatchJudge for
    # free-text model responses - "The capital is Paris." matches reference "Paris".
    provider = OpenAIProvider(model=model)
    judge = ContainsJudge()

    # Template instructs the model to answer concisely.
    # ContainsJudge does not require exact answers, but shorter answers are easier
    # to inspect and cost fewer output tokens.
    template = PromptTemplate("Answer in one word or short phrase: {{ question }}")

    runner = AsyncRunner(
        provider=provider,
        judge=judge,
        template=template,
        concurrency=5,  # 5 concurrent requests - well within OpenAI's rate limits
    )

    # --- Run experiment ---
    # strict=False here because n=50 will trigger a CI_TOO_WIDE warning.
    # In a real evaluation you would either collect more data (n≥200) or
    # use strict=True to force yourself to fix the design before running.
    # We use strict=False here so this example runs without modification
    # and demonstrates the RigorChecker output on a real (imperfect) setup.
    experiment = Experiment(
        name=f"{dataset.name}_{model.replace('-', '_')}",
        dataset=dataset,
        runner=runner,
        n_resamples=2000,  # 2000 resamples for demo speed; use 10000 for publication
        strict=False,
    )

    print(f"Running evaluation against {model}...")
    print("(50 examples, ~5 concurrent requests, expect 15-30 seconds)\n")

    try:
        result = experiment.run()
    except PreFlightError as e:
        # This should not trigger with strict=False, but guard anyway
        print(f"ERROR: Pre-flight audit failed:\n{e.audit}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        # Re-raise directly - the traceback is more informative than a summary message,
        # and printing then re-raising produces confusing duplicate output.
        raise

    # --- Display results ---
    print("=" * 60)
    print(f"Results - {result.experiment_name}")
    print("=" * 60)

    for name, m in result.metrics.items():
        ci_str = f"[{m.ci_lower:.3f}, {m.ci_upper:.3f}]"
        print(f"  {name:<20} {m.value:.4f}   95% CI {ci_str}   ±{m.margin_of_error:.4f}")

    print()

    # --- RigorChecker audit ---
    audit = result.posthoc_audit
    if audit.passed and not audit.warnings:
        print("RigorChecker: ✅ PASS - no issues found")
    elif audit.passed:
        print(f"RigorChecker: ⚠  PASS with {len(audit.warnings)} warning(s)")
        for f in audit.warnings:
            print(f"  [{f.code}] {f.message}")
            print(f"  → {f.action}")
    else:
        print(f"RigorChecker: ✗ FAIL - {len(audit.errors)} error(s)")
        for f in audit.errors:
            print(f"  [{f.code}] {f.message}")
            print(f"  → {f.action}")

    print()

    # --- Cost and token usage ---
    # Use run_result directly - provider.cost_summary() accumulates across all
    # experiments that share the provider instance, so it's only accurate here
    # because we create a fresh provider per run. Using run_result is always correct.
    run = result.run_result
    print(
        f"Cost:  ${run.total_cost_usd:.4f}  |  "
        f"Tokens: {run.total_tokens:,}  |  "
        f"{run.n} API calls  |  "
        f"{run.wall_time_seconds:.1f}s"
    )
    print(f"Model: {run.model}")

    print()

    # --- Worst examples ---
    # worst_examples() returns the examples with the lowest scores.
    # For ContainsJudge (binary), score=0.0 means wrong - sort is effectively
    # by insertion order among wrong answers, which is fine for inspection.
    # Keys: example_id, prompt, output, reference, score, reasoning
    n_total = result.run_result.n
    all_wrong = result.worst_examples(n=n_total)  # get all examples to filter wrong ones
    wrong_examples = [e for e in all_wrong if e["score"] == 0.0]
    if wrong_examples:
        sample = wrong_examples[:3]
        print(f"Sample of incorrect answers ({len(wrong_examples)} total wrong out of {n_total}):")
        for ex in sample:
            # Extract the question from the prompt (the template renders it)
            # The reference is the expected short answer
            print(f"  ID: {ex['example_id']}")
            print(f"    Expected:  {ex['reference']}")
            print(f"    Got:       {ex['output']!r}")
        if len(wrong_examples) > 3:
            print(f"  ... and {len(wrong_examples) - 3} more. See HTML report for full list.")
    else:
        print(f"✓ All {n_total} examples answered correctly.")

    # --- HTML report ---
    if output_path:
        result.generate_report(path=output_path)
        print(f"\n✓ HTML tearsheet written to {output_path}")
        print("  Open it in a browser for the full interactive report.")

    # --- Interpretation guidance ---
    print()
    print("─" * 60)
    print("Interpreting these results:")
    acc = result.metrics["Accuracy"]
    print(
        f"  Accuracy {acc.value:.1%} means the model answered {int(acc.value * len(dataset))}"
        f"/{len(dataset)} questions correctly"
    )
    print(
        f"  The 95% CI [{acc.ci_lower:.1%}, {acc.ci_upper:.1%}] is the range of plausible "
        "true accuracy values."
    )
    print(f"  CI half-width ±{acc.margin_of_error:.1%} - this is how precise your measurement is.")
    print()
    print(
        "  With n=50, the CI is wide. To get ±5pp precision, collect n≥200 examples."
        "\n  Run: evalkit power 0.05 --test ci"
    )
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evalkit live API quickstart - runs against OpenAI and shows a full audit."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to evaluate (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write an HTML tearsheet (optional)",
    )
    args = parser.parse_args()
    main(model=args.model, output_path=args.output)
