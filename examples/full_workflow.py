"""
evalkit full workflow example.

Demonstrates the complete pipeline with zero API keys required.
Run from the project root:

    pip install -e ".[dev]"
    python examples/full_workflow.py

What this covers:
  1. Power analysis - before spending any compute budget
  2. Dataset creation
  3. Evaluation with mock model (deterministic, reproducible)
  4. Metrics with bootstrap confidence intervals
  5. Model comparison with McNemar's test
  6. Multiple testing correction with Benjamini-Hochberg FDR
  7. Inter-rater agreement measurement (simulated LLM judges)
  8. RigorChecker audit - both a good and a bad experiment
  9. Full Experiment object (combines all of the above)
 10. HTML tearsheet generation
 11. Direct model comparison with compare()
 12. Error analysis - worst_examples()
 13. Dataset utilities - split() and sample()
 14. Template validation before spending budget
 15. Sample size planning table
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# rich is an optional dev dependency. Import gracefully so the example still
# runs in environments that only have the core package installed.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    _console = Console()

    def _rule(title: str) -> None:
        _console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))

    def _panel(body: str, title: str, colour: str = "yellow") -> None:
        _console.print(Panel(body, title=f"[{colour}]{title}[/{colour}]", border_style=colour))

    def _print(msg: str = "") -> None:
        _console.print(msg)

except ImportError:
    import re as _re

    def _rule(title: str) -> None:  # type: ignore[misc]
        print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")

    def _panel(body: str, title: str, colour: str = "yellow") -> None:  # type: ignore[misc]
        print(f"\n┌── {title} ──")
        for line in body.splitlines():
            print(f"│  {line}")
        print("└" + "─" * 40)

    def _print(msg: str = "") -> None:  # type: ignore[misc]
        print(_re.sub(r"\[/?[^\]]+\]", "", msg))


def main() -> None:
    _rule("evalkit - Full Workflow Demo")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Power analysis BEFORE running anything
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 1: Power analysis - how many examples do we need?[/bold]")

    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=0.05, power=0.80)
    ci_result = pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=0.75)
    cmp_result = pa.for_proportion_difference(effect_size=0.05, p1=0.75)

    _print(f"  To report accuracy to ±5%:          need n >= {ci_result.minimum_n}")
    _print(f"  To detect a 5pp accuracy difference: need n >= {cmp_result.minimum_n}")
    _print("  We will use n=400 -- adequately powered for both.")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Build the dataset
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 2: Create evaluation dataset (n=400)[/bold]")

    from evalkit.core.dataset import EvalDataset, PromptTemplate

    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is 12 x 12?", "144"),
        ("Who wrote Pride and Prejudice?", "Jane Austen"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many sides does a hexagon have?", "6"),
        ("What year did the Berlin Wall fall?", "1989"),
        ("What planet is closest to the Sun?", "Mercury"),
        ("What is the square root of 256?", "16"),
    ] * 50  # 400 examples total

    records = [{"id": str(i), "question": q, "label": a} for i, (q, a) in enumerate(qa_pairs)]
    dataset = EvalDataset.from_list(records, name="qa_benchmark")

    dist = dataset.label_distribution()
    _print(f"  [green]OK[/green] {len(dataset)} examples, {len(dist)} unique labels")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Evaluate Model A (~82% accuracy)
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 3: Evaluate Model A (mock, target accuracy ~82%)[/bold]")

    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    template = PromptTemplate("Answer concisely: {{ question }}")
    judge = ExactMatchJudge(case_sensitive=False)

    runner_a = MockRunner(judge=judge, template=template, accuracy=0.82, seed=42)
    result_a = runner_a.run(dataset)
    raw_a = sum(result_a.correct) / result_a.n
    _print(f"  [green]OK[/green] Model A done. Raw accuracy: {raw_a:.3f} (n={result_a.n})")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Metrics with bootstrap CIs
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 4: Compute metrics with 95% bootstrap confidence intervals[/bold]")

    from evalkit.metrics.accuracy import Accuracy

    acc_metric = Accuracy(n_resamples=10_000, seed=42)
    acc = acc_metric.compute(result_a.correct, [1] * result_a.n)
    _print(f"  {acc}")
    _print()

    _panel(
        f"Typical reporting:  accuracy = {raw_a:.2f}  <- point estimate, no uncertainty\n"
        f"evalkit reporting:  {acc}\n\n"
        f"The CI half-width is +-{acc.margin_of_error:.3f}. Reporting to two decimal\n"
        f"places implies precision you don't have unless CI half-width is < 0.005.",
        title="Why this matters",
        colour="yellow",
    )
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Evaluate Model B and compare with McNemar's test
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 5: Evaluate Model B (~72%) and compare with McNemar's test[/bold]")

    runner_b = MockRunner(judge=judge, template=template, accuracy=0.72, seed=99)
    result_b = runner_b.run(dataset)
    raw_b = sum(result_b.correct) / result_b.n
    _print(f"  [green]OK[/green] Model B done. Raw accuracy: {raw_b:.3f}")

    from evalkit.metrics.comparison import McNemarTest

    test_result = McNemarTest(alpha=0.05).test(result_a.correct, result_b.correct)
    decision = "REJECT H0" if test_result.reject_null else "fail to reject H0"
    _print(
        f"\n  McNemar's test: chi2={test_result.statistic:.3f}, "
        f"p={test_result.p_value:.4f}, OR={test_result.effect_size:.3f} "
        f"-> {decision} (alpha=0.05)"
    )
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Multiple testing correction - comparing 6 prompt variants
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 6: BH-FDR correction -- comparing 6 prompt variants[/bold]")

    from evalkit.metrics.comparison import BHCorrection

    # Simulate 6 prompt variants: 1 genuine signal, 5 noise.
    # Prompt B at p=0.041 looks significant raw -- after BH it becomes
    # p_adj=0.123, a false positive.
    p_values = [0.003, 0.041, 0.068, 0.24, 0.51, 0.78]
    names = [f"Prompt {chr(65 + i)}" for i in range(6)]

    bh = BHCorrection(alpha=0.05)
    bh_result = bh.correct(p_values, comparison_names=names)
    _print(str(bh_result))

    if bh_result.false_positive_warning:
        _panel(
            "Prompt B: p_raw=0.041 looks significant without correction,\n"
            "but p_adj=0.123 after BH correction. This is a false positive.\n\n"
            "Without FDR correction, you would incorrectly claim Prompt B is better.",
            title="False Positive Detected",
            colour="red",
        )
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Inter-rater agreement (simulated LLM judges)
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 7: Validate LLM judge with inter-rater agreement[/bold]")

    from evalkit.metrics.agreement import CohenKappa

    rng = np.random.default_rng(7)
    true_labels = rng.choice([0, 1], size=100, p=[0.4, 0.6]).tolist()

    # Good judge: agrees with reference 88% of the time (kappa ~ 0.74, substantial)
    judge1 = [lab if rng.random() > 0.12 else 1 - lab for lab in true_labels]
    # Bad judge: only agrees 68% of the time (kappa ~ 0.34, fair -- below threshold)
    judge_bad = [lab if rng.random() > 0.32 else 1 - lab for lab in true_labels]

    good_kappa = CohenKappa(n_resamples=5_000, seed=0).compute(judge1, true_labels)
    bad_kappa = CohenKappa(n_resamples=5_000, seed=0).compute(judge_bad, true_labels)

    _print(f"  Good judge: {good_kappa}")
    _print(f"  Bad judge:  {bad_kappa}")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: RigorChecker audit - the killer feature
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 8: RigorChecker -- automated statistical audit[/bold]")

    from evalkit.analysis.rigour import RigorChecker

    checker = RigorChecker()

    # Well-designed experiment.
    good_report = checker.audit(
        n_examples=400,
        accuracy=0.82,
        label_distribution=dist,
        n_variants=1,
        experiment_name="model_a_evaluation",
    )
    _print("Well-designed experiment:")
    _print(str(good_report))

    # Bad experiment: the kind that gets published without evalkit.
    bad_report = checker.audit(
        n_examples=47,
        accuracy=0.68,
        label_distribution={"correct": 43, "incorrect": 4},
        n_variants=8,
        p_values=[0.03, 0.04, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        judge_kappa=0.41,
        experiment_name="underpowered_eval",
    )
    _print("Poorly designed experiment:")
    _print(str(bad_report))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 9: Full Experiment object (combines all steps automatically)
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 9: Full Experiment -- everything in one object[/bold]")

    from evalkit.core.experiment import Experiment

    exp = Experiment(name="qa_benchmark_model_a", dataset=dataset, runner=runner_a)
    exp_result = exp.run()
    exp_result.print_summary()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 10: HTML tearsheet
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 10: Generate self-contained HTML tearsheet[/bold]")

    import tempfile

    from evalkit.analysis.report import ReportGenerator

    # Write to a temp file so the demo leaves no artifacts in the repo.
    # In real use: ReportGenerator().generate(result, output_path="report.html")
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(ReportGenerator().generate(exp_result))
        report_path = Path(f.name)
    _print(f"  [green]OK[/green] Report written to {report_path}")
    _print('  (temp file - in real use pass output_path="report.html" to persist)')
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 11: Direct model comparison with .compare()
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 11: Compare models directly with result_a.compare(result_b)[/bold]")

    exp_a = Experiment("model_a_82pct", dataset, runner_a, n_resamples=2000)
    exp_b = Experiment("model_b_72pct", dataset, runner_b, n_resamples=2000)
    res_a = exp_a.run()
    res_b = exp_b.run()

    comparison = res_a.compare(res_b)
    _print(str(comparison))
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 12: Error analysis with .worst_examples()
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 12: Error analysis - worst_examples()[/bold]")
    _print("  The 3 examples Model A most confidently got wrong:")
    _print()

    worst = res_a.worst_examples(3)
    for ex in worst:
        _print(
            f"  ID: {ex['example_id']}"
            f"  |  Output: {ex['output']!r}"
            f"  |  Reference: {ex['reference']!r}"
        )
    _print()
    _print("  This is more useful than aggregate accuracy. Fix these and re-evaluate.")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 13: Dataset split for held-out evaluation
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 13: Dataset utilities - split() and sample()[/bold]")

    train_ds, test_ds = dataset.split(test_size=0.2, stratify=True)
    _print(f"  split(test_size=0.2): train={len(train_ds)}, test={len(test_ds)}")

    small_ds = dataset.sample(50)
    _print(f"  sample(50): {len(small_ds)} examples for quick prototyping")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 14: Template validation before spending API budget
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 14: PromptTemplate.validate() - catch field errors before running[/bold]")

    good_template = PromptTemplate("Answer concisely: {{ question }}")
    bad_template = PromptTemplate("{{ question }} - context: {{ context }}")

    good_errors = good_template.validate(dataset)
    bad_errors = bad_template.validate(dataset)

    _print(f"  Good template errors: {good_errors}")  # []
    _print(f"  Bad template errors (first): {bad_errors[0] if bad_errors else 'none'}")
    _print("  Validation runs before any API calls - zero cost to catch this early.")
    _print()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 15: Sample size planning table
    # ──────────────────────────────────────────────────────────────────────────
    _print("[bold]Step 15: Sample size planning table - how many examples do I need?[/bold]")

    from evalkit.analysis.power import PowerAnalysis

    PowerAnalysis(alpha=0.05).sample_size_table(
        effect_sizes=[0.05, 0.10, 0.15, 0.20],
        powers=[0.70, 0.80, 0.90],
    )
    _print()

    _rule("Done")
    _print()
    _print(
        "Completed without any API keys. Demonstrated:\n"
        "  * Power analysis before spending budget\n"
        "  * Bootstrap CIs on every metric\n"
        "  * McNemar's test for paired model comparison\n"
        "  * BH-FDR correction catching a false positive\n"
        "  * Inter-rater agreement validating a judge\n"
        "  * RigorChecker audit surfacing statistical problems\n"
        "  * Self-contained HTML tearsheet\n"
        "  * result_a.compare(result_b) - direct comparison with significance test\n"
        "  * worst_examples() - error analysis on confident mistakes\n"
        "  * split() and sample() - dataset utilities\n"
        "  * validate() - template field checking before spending API budget\n"
        "  * sample_size_table() - planning grid for experiment design"
    )


if __name__ == "__main__":
    main()
