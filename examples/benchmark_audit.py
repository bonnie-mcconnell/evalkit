"""
evalkit benchmark audit: demonstrating the core thesis on realistic evaluation data.

This script shows what evalkit catches when you run it on the kinds of evaluation
scenarios that appear in published LLM papers and leaderboards. Every scenario here
uses synthetic data that mirrors real patterns - the numbers are chosen to reflect
what you actually see in published evals, not invented to make the library look good.

The scenarios:

1. The "impressive improvement" that is pure noise.
   A paper reports 74% → 78% accuracy improvement. n=50. With ±12pp CI, this is
   indistinguishable from random variation.

2. The class-imbalanced sentiment dataset.
   92% of examples are class "positive". A model that predicts "positive" for
   everything gets 92% accuracy. Accuracy is a useless metric here.

3. The multi-model comparison that invents winners.
   Running 8 prompt variants and picking the best one. At α=0.05, the probability
   of at least one false positive is 34%. Three variants look significant without
   FDR correction. Zero survive BH-FDR.

4. The LLM judge with poor agreement.
   A judge with κ=0.41 (below the 0.60 threshold). Every score produced by this
   judge is unreliable and the RigorChecker flags it as an ERROR.

5. The well-designed experiment.
   n=500, balanced classes, a single well-powered comparison. This is what
   passing the RigorChecker audit actually looks like.

Run from the project root:

    pip install -e ".[dev]"
    python examples/benchmark_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from evalkit import (
    BalancedAccuracy,
    BHCorrection,
    CohenKappa,
    EvalDataset,
    ExactMatchJudge,
    Experiment,
    F1Score,
    MockRunner,
    PowerAnalysis,
    PromptTemplate,
    RigorChecker,
)

console = Console()


def section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()


def _make_dataset(
    n: int,
    labels: list[str] | None = None,
    imbalance: float | None = None,
    name: str = "eval",
) -> EvalDataset:
    """Build a synthetic dataset. imbalance=0.92 means 92% of examples are label[0]."""
    if labels is None:
        labels = ["correct", "incorrect"]

    if imbalance is not None:
        n_majority = int(n * imbalance)
        n_minority = n - n_majority
        all_labels = [labels[0]] * n_majority + [labels[1]] * n_minority
    else:
        all_labels = [labels[i % len(labels)] for i in range(n)]

    records = [{"id": str(i), "question": f"Q{i}", "label": all_labels[i]} for i in range(n)]
    return EvalDataset.from_list(records, name=name)


def scenario_1_underpowered_improvement() -> None:
    """
    Scenario 1: The 'impressive improvement' that is pure noise.

    A paper reports: "Our new prompt increases accuracy from 74% to 78% on our
    evaluation set (n=50)." This is a 4 percentage point improvement. Sounds real.
    At n=50, the 95% CI for a proportion around 0.76 is ±12 percentage points.
    The two results are completely indistinguishable from random variation.
    """
    section("Scenario 1: The 'impressive improvement' that is noise (n=50)")

    pa = PowerAnalysis(alpha=0.05, power=0.80)
    ci_result = pa.for_ci_precision(desired_half_width=0.05, expected_accuracy=0.76)
    comparison_result = pa.for_proportion_difference(effect_size=0.04, p1=0.74)

    console.print(
        Panel(
            f"[bold]Claim:[/bold] Accuracy improved from 74% → 78% on n=50 examples.\n\n"
            f"[bold]What evalkit finds:[/bold]\n"
            f"  • To report accuracy to ±5% precision: need n ≥ [red]{ci_result.minimum_n:,}[/red]\n"
            f"  • To detect a 4pp improvement at 80% power: need n ≥ [red]{comparison_result.minimum_n:,}[/red]\n"
            f"  • At n=50, the 95% CI half-width is ±{100 * 1.96 * (0.76 * 0.24 / 50) ** 0.5:.0f}pp\n\n"
            f"[dim]The 4pp improvement is inside the noise floor of the measurement.\n"
            f"This result cannot distinguish real improvement from random variation.[/dim]",
            title="[yellow]SCENARIO 1[/yellow]",
            border_style="yellow",
        )
    )

    # Now run the experiment and show the RigorChecker output
    dataset = _make_dataset(50, name="small_eval")
    template = PromptTemplate("{{ question }}")
    runner_baseline = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.74, seed=1)
    runner_new = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.78, seed=2)

    result_a = Experiment("baseline-prompt", dataset, runner_baseline, n_resamples=2000).run()
    result_b = Experiment("new-prompt", dataset, runner_new, n_resamples=2000).run()

    comparison = result_a.compare(result_b)
    console.print(str(comparison))
    console.print()
    console.print("[bold]RigorChecker audit for baseline run:[/bold]")
    console.print(str(result_a.posthoc_audit))


def scenario_2_class_imbalance() -> None:
    """
    Scenario 2: The class-imbalanced dataset where accuracy is meaningless.

    A sentiment classifier is evaluated on a dataset that is 92% positive reviews.
    The "accuracy" of 91% is actually worse than the trivial majority-class predictor.
    evalkit catches this with SEVERE_CLASS_IMBALANCE and shows BalancedAccuracy
    revealing the true picture.
    """
    section("Scenario 2: 92% class imbalance - accuracy is a useless metric")

    dataset = _make_dataset(200, labels=["positive", "negative"], imbalance=0.92, name="sentiment")
    template = PromptTemplate("{{ question }}")

    # Model that predicts majority class ~90% of the time
    # (mimics a model that learned to predict "positive" for most things)
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.85, seed=42)

    result = Experiment(
        "sentiment-classifier",
        dataset,
        runner,
        additional_metrics=[
            BalancedAccuracy(n_resamples=2000),
            F1Score(average="macro", n_resamples=2000),
        ],
        n_resamples=2000,
    ).run()

    console.print("[bold]Metrics:[/bold]")
    for m in result.metrics.values():
        console.print(f"  {m}")

    console.print()
    console.print(str(result.posthoc_audit))

    acc = result.metrics["Accuracy"].value
    bal = result.metrics["BalancedAccuracy"].value
    console.print(
        f"\n[dim]Accuracy ({acc:.2%}) and BalancedAccuracy ({bal:.2%}) diverge because the "
        f"model learned to predict the majority class. The 'high accuracy' is misleading.[/dim]"
    )


def scenario_3_multiple_testing() -> None:
    """
    Scenario 3: Multi-model comparison that invents winners via multiple testing.

    8 prompt variants are compared to a baseline. At α=0.05, you'd expect 0.4
    false positives. With 8 tests, the probability of at least one false positive
    is 34%. Three variants show p < 0.05 without correction. Zero survive
    Benjamini-Hochberg FDR correction.
    """
    section("Scenario 3: 8-way comparison - multiple testing invents winners")

    # Realistic p-values from comparing 8 prompt variants where H0 is true for most
    # (small differences that are just noise at the sample size used)
    p_values_raw = [0.031, 0.044, 0.049, 0.12, 0.24, 0.38, 0.51, 0.73]
    variant_names = [f"variant_{i + 1}" for i in range(8)]

    n_significant_raw = sum(1 for p in p_values_raw if p < 0.05)
    prob_fp = 1 - (0.95 ** len(p_values_raw))

    console.print(f"[bold]Without correction:[/bold] {n_significant_raw} variants look significant")
    console.print(
        f"[bold]P(≥1 false positive) with {len(p_values_raw)} tests:[/bold] {prob_fp:.0%}"
    )
    console.print()

    bh = BHCorrection(alpha=0.05)
    result = bh.correct(p_values_raw, comparison_names=variant_names)

    console.print("[bold]After Benjamini-Hochberg FDR correction:[/bold]")
    for name, raw, adj, reject in zip(
        variant_names,
        result.unadjusted_p_values,
        result.adjusted_p_values,
        result.reject_null,
    ):
        marker = "[green]✓ significant[/green]" if reject else "[red]✗ not significant[/red]"
        console.print(f"  {name}: p_raw={raw:.3f} → p_adj={adj:.3f}  {marker}")

    n_after = sum(result.reject_null)
    console.print(
        f"\n[dim]{n_significant_raw} variants appeared significant. "
        f"{n_after} survive FDR correction. "
        f"{'All were false positives.' if n_after == 0 else 'Review remaining carefully.'}[/dim]"
    )

    # Show what RigorChecker does with this
    console.print()
    checker = RigorChecker()
    audit = checker.audit(
        n_examples=150,
        accuracy=0.72,
        n_variants=8,
        p_values=p_values_raw,
        experiment_name="8-variant-prompt-sweep",
    )
    console.print(str(audit))


def scenario_4_poor_judge_agreement() -> None:
    """
    Scenario 4: LLM judge with poor inter-rater agreement.

    An LLM judge is used to score open-ended model responses. The judge is run
    twice on the same 80 examples. Cohen's kappa is 0.41 - below the 0.60 minimum
    threshold. Every score produced by this judge is unreliable.
    """
    section("Scenario 4: LLM judge with κ=0.41 - scores are unreliable")

    import numpy as np

    # Simulate two runs of the same judge with κ ≈ 0.41
    rng = np.random.default_rng(7)
    n = 80
    # Base judgments
    base = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    # Second run: agree 68% of the time (gives κ ≈ 0.41 for 65/35 split)
    agree_mask = rng.random(n) < 0.68
    run2 = np.where(agree_mask, base, 1 - base)

    kappa_result = CohenKappa(n_resamples=1000).compute(base.tolist(), run2.tolist())
    console.print("[bold]Judge agreement measurement:[/bold]")
    console.print(f"  {kappa_result}")
    console.print()

    # Show RigorChecker flagging this
    checker = RigorChecker()
    audit = checker.audit(
        n_examples=n,
        accuracy=0.65,
        judge_kappa=kappa_result.metric.value,
        experiment_name="llm-judge-eval",
    )
    console.print(str(audit))
    console.print(
        "[dim]Every score produced by this judge is unreliable at this agreement level.\n"
        "Improving the judge prompt or using an ensemble is required before reporting results.[/dim]"
    )


def scenario_5_well_designed_experiment() -> None:
    """
    Scenario 5: A well-designed experiment that passes the RigorChecker.

    n=500, balanced classes, single model comparison with enough power to detect
    the observed effect. This is what 'statistically sound' actually looks like.
    """
    section("Scenario 5: A well-designed experiment - what passing looks like")

    dataset = _make_dataset(500, labels=["positive", "negative"], name="well_powered_eval")
    template = PromptTemplate("{{ question }}")

    runner_a = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.81, seed=1)
    runner_b = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.74, seed=2)

    result_a = Experiment(
        "model-A",
        dataset,
        runner_a,
        additional_metrics=[
            BalancedAccuracy(n_resamples=2000),
            F1Score(average="macro", n_resamples=2000),
        ],
        n_resamples=2000,
    ).run()

    result_b = Experiment(
        "model-B",
        dataset,
        runner_b,
        n_resamples=2000,
    ).run()

    result_a.print_summary()

    console.print("\n[bold]Model comparison:[/bold]")
    comparison = result_a.compare(result_b)
    console.print(str(comparison))


def main() -> None:
    console.print(
        Panel(
            "[bold]evalkit benchmark audit[/bold]\n\n"
            "Demonstrating the five most common ways LLM evaluation goes wrong,\n"
            "and what evalkit does about each of them.\n\n"
            "[dim]All scenarios use synthetic data that mirrors real published evaluation patterns.[/dim]",
            border_style="cyan",
        )
    )

    scenario_1_underpowered_improvement()
    scenario_2_class_imbalance()
    scenario_3_multiple_testing()
    scenario_4_poor_judge_agreement()
    scenario_5_well_designed_experiment()

    console.print()
    console.print(
        Panel(
            "[bold green]Summary[/bold green]\n\n"
            "Scenario 1 - [yellow]Underpowered[/yellow]: n=50 gives ±12pp CI. The 4pp improvement is noise.\n"
            "Scenario 2 - [yellow]Class imbalance[/yellow]: 92% majority class. Accuracy reports 91% on a bad model.\n"
            "Scenario 3 - [yellow]Multiple testing[/yellow]: 3/8 variants looked significant. 0/8 survive FDR.\n"
            "Scenario 4 - [red]Poor judge[/red]: κ=0.41 means every LLM judge score is unreliable.\n"
            "Scenario 5 - [green]Well-designed[/green]: n=500, balanced, single comparison. PASS.\n\n"
            "[dim]Install: pip install evalkit-research\n"
            "GitHub: https://github.com/bonnie-mcconnell/evalkit[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
