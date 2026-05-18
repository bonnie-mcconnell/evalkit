"""
evalkit CLI - command-line interface for the evalkit framework.

Commands:
  evalkit run      - run an evaluation from a JSONL dataset
  evalkit compare  - compare two run result files (McNemar or Wilcoxon)
  evalkit power    - compute sample size requirements before running
  evalkit table    - print a sample size planning grid
  evalkit version  - print version info

Design note: the CLI is a thin wrapper around the Python API, not a parallel
implementation. Every command builds the same objects you'd build in a script.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalkit.analysis.rigour import AuditReport
    from evalkit.core.experiment import ExperimentResult


import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evalkit.analysis.rigour import _severity_sort_key

app = typer.Typer(
    name="evalkit",
    help="Rigorous LLM evaluation: bootstrap CIs, significance testing, automated auditing.",
    add_completion=False,
)

# All human-readable output (status, progress, Rich tables) goes to stderr.
# This keeps stdout clean for machine-readable output (--format json).
# When format=text, stderr and stdout both go to the terminal so the user
# sees everything. When format=json and the user pipes to jq, stderr still
# goes to the terminal and stdout is pure JSON.
console = Console(stderr=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to JSONL dataset file."),
    model: str = typer.Option(
        "mock", "--model", "-m", help="Model to evaluate. Use 'mock' for demo."
    ),  # noqa: E501
    template: str = typer.Option(
        "{{ question }}",
        "--template",
        "-t",
        help="Jinja2 prompt template. Use {{ field }} for dataset fields.",
    ),
    reference_field: str = typer.Option(
        "label", "--ref-field", help="Dataset field containing ground truth."
    ),  # noqa: E501
    judge: str = typer.Option(
        "exact",
        "--judge",
        "-j",
        help=(
            "Judge type: exact, contains, regex, or llm. "
            "llm requires a real --model (gpt-* or claude-*)."
        ),
    ),
    regex_pattern: str | None = typer.Option(
        None, "--regex", help="Regex pattern for regex judge."
    ),  # noqa: E501
    output: Path | None = typer.Option(None, "--output", "-o", help="Path to write HTML report."),  # noqa: E501
    save_results: Path | None = typer.Option(
        None, "--save-results", help="Save run results as JSON for use with 'evalkit compare'."
    ),  # noqa: E501
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text (default) or json (pipeable)."
    ),  # noqa: E501
    n_resamples: int = typer.Option(10_000, "--resamples", help="Bootstrap resamples for CIs."),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Concurrent API calls."),
    checkpoint_dir: Path | None = typer.Option(
        None, "--checkpoint-dir", help="Directory for checkpoints."
    ),  # noqa: E501
    mock_accuracy: float = typer.Option(
        0.82, "--mock-accuracy", help="Accuracy for mock model (demo)."
    ),  # noqa: E501
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key. For OpenAI models: also reads OPENAI_API_KEY. "
        "For Anthropic models: also reads ANTHROPIC_API_KEY.",
    ),
    fail_on_errors: bool = typer.Option(
        False,
        "--fail-on-errors",
        help="Exit with code 1 if the RigorChecker audit finds ERROR-level findings. "
        "Useful for CI pipelines: gates deployment on statistical quality.",
    ),
    no_strict: bool = typer.Option(
        False,
        "--no-strict",
        help="Disable strict pre-flight mode. By default, pre-flight ERROR findings "
        "(e.g. SAMPLE_TOO_SMALL) abort the run before any API calls are made. "
        "--no-strict overrides this and runs anyway, relying on the post-hoc audit instead. "
        "Use when you know the issues and want results anyway.",
    ),
) -> None:
    """
    Run an evaluation on a JSONL dataset and produce a tearsheet with CIs.

    Examples:
      evalkit run data.jsonl --model mock --template "Q: {{ question }}"
      evalkit run data.jsonl --model gpt-4o-mini --judge llm --template "Q: {{ question }}"
      evalkit run data.jsonl --model gpt-4o-mini --format json | jq .metrics.Accuracy.value
    """
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.experiment import Experiment
    from evalkit.core.judge import ContainsJudge, ExactMatchJudge, LLMJudge, RegexMatchJudge
    from evalkit.core.runner import AsyncRunner, MockRunner

    if format not in ("text", "json"):
        console.print(f"[red]Unknown format '{format}'. Use: text, json[/red]")
        raise typer.Exit(1)

    # Load dataset
    with console.status(f"[cyan]Loading dataset from {dataset}..."):
        try:
            ds = EvalDataset.from_jsonl(dataset, reference_field=reference_field)
        except Exception as e:
            console.print(f"[red]Failed to load dataset:[/red] {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded {len(ds)} examples from [bold]{ds.name}[/bold]")

    # Build judge. For --judge llm the judge object is replaced after the
    # provider is constructed (we need the provider reference). Use a sentinel.
    judge_obj: ExactMatchJudge | ContainsJudge | RegexMatchJudge | LLMJudge
    if judge == "exact":
        judge_obj = ExactMatchJudge()
    elif judge == "contains":
        judge_obj = ContainsJudge()
    elif judge == "regex":
        if not regex_pattern:
            console.print("[red]--regex is required when --judge=regex[/red]")
            raise typer.Exit(1)
        judge_obj = RegexMatchJudge(pattern=regex_pattern)
    elif judge == "llm":
        if model == "mock":
            console.print(
                "[red]--judge llm requires a real model (gpt-* or claude-*), not mock.[/red]"
            )
            raise typer.Exit(1)
        judge_obj = ExactMatchJudge()  # placeholder - replaced below once provider is built
    else:
        console.print(f"[red]Unknown judge type '{judge}'. Use: exact, contains, regex, llm[/red]")
        raise typer.Exit(1)

    prompt_template = PromptTemplate(template)

    # Validate template against the dataset before constructing any provider.
    # Catches field-name typos at zero cost - before any API keys are used.
    prompt_errors = prompt_template.validate(ds)
    if prompt_errors:
        console.print("[red bold]Template validation failed - fix before running:[/red bold]")
        for err in prompt_errors:
            console.print(f"  [red]{err}[/red]")
        raise typer.Exit(1)

    # Build runner
    from evalkit.providers.base import ModelProvider

    runner: MockRunner | AsyncRunner
    provider: ModelProvider
    if model == "mock":
        console.print(f"[yellow]Using mock model (accuracy={mock_accuracy:.0%})[/yellow]")
        runner = MockRunner(
            judge=judge_obj,
            template=prompt_template,
            accuracy=mock_accuracy,
        )
    elif model.startswith("gpt"):
        from evalkit.providers.base import OpenAIProvider

        provider = OpenAIProvider(model=model, api_key=api_key)
        runner = AsyncRunner(
            provider=provider,
            judge=judge_obj,
            template=prompt_template,
            concurrency=concurrency,
            checkpoint_dir=checkpoint_dir,
        )
    elif model.startswith("claude"):
        from evalkit.providers.base import AnthropicProvider

        provider = AnthropicProvider(model=model, api_key=api_key)
        runner = AsyncRunner(
            provider=provider,
            judge=judge_obj,
            template=prompt_template,
            concurrency=concurrency,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        console.print(
            f"[red]Unknown model '{model}'. Supported prefixes: mock, gpt-*, claude-*[/red]\n"
            "[dim]Example: --model gpt-4o-mini  or  --model claude-3-5-sonnet-20241022[/dim]"
        )  # noqa: E501
        raise typer.Exit(1)

    # Now that we have a provider, wire up the LLM judge if requested.
    if judge == "llm":
        judge_obj = LLMJudge(provider=provider)
        runner = AsyncRunner(
            provider=provider,
            judge=judge_obj,
            template=prompt_template,
            concurrency=concurrency,
            checkpoint_dir=checkpoint_dir,
        )
        console.print(
            "[cyan]LLM judge active. Validate inter-rater agreement before reporting scores.[/cyan]"
        )

    # Run experiment
    # strict=True (default): pre-flight ERRORs raise PreFlightError before any
    # API calls are made. --no-strict overrides this for when you want results
    # despite known issues (e.g. small demo dataset).
    from evalkit.core.experiment import PreFlightError

    experiment = Experiment(
        name=f"{ds.name}_{model}",
        dataset=ds,
        runner=runner,
        n_resamples=n_resamples,
        strict=not no_strict,
    )

    try:
        with console.status("[cyan]Running evaluation..."):
            result = experiment.run()
    except PreFlightError as e:
        console.print("\n[red bold]✗ Pre-flight audit FAILED - experiment aborted[/red bold]")
        console.print(str(e.audit))
        console.print("\n[dim]Fix the issues above, or pass --no-strict to run anyway.[/dim]")
        raise typer.Exit(1) from None

    if format == "json":
        run = result.run_result
        output_payload: dict[str, object] = {
            "experiment": result.experiment_name,
            "model": run.model,
            "dataset": run.dataset_name,
            "n": run.n,
            "metrics": {
                name: {
                    "value": m.value,
                    "ci_lower": m.ci_lower,
                    "ci_upper": m.ci_upper,
                    "margin_of_error": m.margin_of_error,
                    "n": m.n,
                }
                for name, m in result.metrics.items()
            },
            "audit": {
                "passed": result.posthoc_audit.passed,
                "n_errors": len(result.posthoc_audit.errors),
                "n_warnings": len(result.posthoc_audit.warnings),
                "findings": [
                    {"code": f.code, "severity": f.severity.value, "message": f.message}
                    for f in result.posthoc_audit.findings
                ],
            },
            "cost_usd": run.total_cost_usd,
            "example_ids": run.example_ids,
            "correct": run.correct,
            "scores": run.scores,
        }
        print(json.dumps(output_payload, indent=2))
    else:
        # Print results table
        _print_metrics_table(result)

        # Print RigorChecker
        _print_audit(result.posthoc_audit)

        # Write report
        if output:
            result.generate_report(path=output)
            console.print(f"\n[green]✓[/green] Report written to [bold]{output}[/bold]")
        else:
            suggested = dataset.stem + "_report.html"
            console.print(
                f"\n[dim]Tip: add --output {suggested} to generate an HTML tearsheet.[/dim]"
            )

    # Save results JSON for use with evalkit compare
    if save_results:
        result.save(save_results)
        console.print(f"[green]✓[/green] Run results saved to [bold]{save_results}[/bold]")

    # CI gate: exit 1 if audit has errors and --fail-on-errors is set
    if fail_on_errors and result.posthoc_audit.errors:
        n_errors = len(result.posthoc_audit.errors)
        console.print(
            f"\n[red]✗ Exiting with code 1: audit found {n_errors} error(s). "
            "Use --no-fail-on-errors to suppress.[/red]"
        )
        raise typer.Exit(1)


@app.command()
def compare(
    result_a: Path = typer.Argument(..., help="First run result JSON file (baseline)."),
    result_b: Path = typer.Argument(..., help="Second run result JSON file (new model)."),
    test: str = typer.Option(
        "mcnemar", "--test", "-t", help="Statistical test: mcnemar or wilcoxon."
    ),  # noqa: E501
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level."),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text (default) or json (pipeable)."
    ),  # noqa: E501
    fail_on_regression: bool = typer.Option(
        False,
        "--fail-on-regression",
        help=(
            "Exit with code 2 if result_b is statistically significantly WORSE than result_a. "
            "Exit 0 = no significant difference or B is better than A. "
            "Designed for CI: gate deployment when a model regresses."
        ),
    ),  # noqa: E501
) -> None:
    """
    Compare two model runs with significance testing.

    Reads run result JSON files written by 'evalkit run --save-results'.
    Applies McNemar's test (binary outcomes) or Wilcoxon signed-rank test
    (continuous scores) and reports the test statistic, p-value, and effect size.

    Exit codes (with --fail-on-regression):
      0 = no significant difference, or result_b is significantly better
      2 = result_b is significantly WORSE than result_a (regression detected)

    Examples:
      evalkit compare baseline.json new_model.json
      evalkit compare baseline.json new_model.json --format json | jq .reject_null
      evalkit compare baseline.json new_model.json --fail-on-regression
    """
    from typing import cast

    from evalkit.metrics.comparison import McNemarTest, TestResult, WilcoxonTest

    def load_result_file(path: Path) -> dict[str, object]:
        """Load a saved result JSON (from evalkit run --save-results or result.save())."""
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)
        loaded: dict[str, object] = json.loads(path.read_text())
        return loaded

    a = load_result_file(result_a)
    b = load_result_file(result_b)

    if a.get("example_ids") != b.get("example_ids"):
        console.print(
            "[red]Warning: example IDs differ between result files. "
            "Paired tests require the same examples in the same order.[/red]"
        )
        raise typer.Exit(1)

    result: TestResult
    if test == "mcnemar":
        result = McNemarTest(alpha=alpha).test(
            model_a_correct=cast(list[int], a["correct"]),
            model_b_correct=cast(list[int], b["correct"]),
        )
    elif test == "wilcoxon":
        result = WilcoxonTest(alpha=alpha).test(
            model_a_scores=cast(list[float], a["scores"]),
            model_b_scores=cast(list[float], b["scores"]),
        )
    else:
        console.print(f"[red]Unknown test '{test}'. Use: mcnemar, wilcoxon[/red]")
        raise typer.Exit(1)

    # Directionality:
    # McNemar: effect_size is odds ratio (a_wins / b_wins). >1 means A won more
    #   discordant pairs, so B performed worse than A. Range: (0, ∞).
    # Wilcoxon: effect_size is rank-biserial correlation r = 1 - 2W/(n*(n+1)).
    #   diffs = a - b, so r > 0 means A scored higher than B overall.
    #   Range: [-1, 1]. Never exceeds 1, so the >1 check used for McNemar is wrong here.
    if test == "mcnemar":
        b_is_worse = result.reject_null and result.effect_size > 1
    else:
        # Wilcoxon: positive r means A > B, so B is worse when r > 0 and significant
        b_is_worse = result.reject_null and result.effect_size > 0

    if format == "json":
        print(
            json.dumps(
                {
                    "test": result.test_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "n_pairs": result.n_pairs,
                    "alpha": result.alpha,
                    "reject_null": result.reject_null,
                    "b_is_significantly_worse": b_is_worse,
                    "note": result.note or "",
                    "files": {"a": str(result_a), "b": str(result_b)},
                },
                indent=2,
            )
        )
    else:
        decision = (
            "[green]REJECT H₀[/green]"
            if result.reject_null
            else "[yellow]fail to reject H₀[/yellow]"
        )
        tab = Table(title=f"{result.test_name}: {result_a.stem} vs {result_b.stem}")
        tab.add_column("Field", style="dim")
        tab.add_column("Value", style="bold")
        tab.add_row("Test statistic", f"{result.statistic:.4f}")
        tab.add_row("p-value", f"{result.p_value:.4f}")
        tab.add_row("Effect size", f"{result.effect_size:.4f}")
        tab.add_row("n pairs", str(result.n_pairs))
        tab.add_row("α", str(result.alpha))
        tab.add_row("Decision", decision)
        if result.note:
            tab.add_row("Note", result.note)
        console.print(tab)

        if b_is_worse:
            console.print(
                Panel(
                    f"[red]{result_b.stem} is statistically significantly WORSE "
                    f"than {result_a.stem}.[/red]\n"
                    f"effect={result.effect_size:.3f}, p={result.p_value:.4f} < α={alpha}\n"
                    "Do not deploy the new model without investigation.",
                    title="[red]⚠ Regression Detected[/red]",
                    border_style="red",
                )
            )
        elif result.reject_null:
            console.print(
                Panel(
                    f"[green]{result_b.stem} is statistically significantly BETTER "
                    f"than {result_a.stem}.[/green]\n"
                    f"effect={result.effect_size:.3f}, p={result.p_value:.4f} < α={alpha}",
                    title="[green]✓ Improvement Confirmed[/green]",
                    border_style="green",
                )
            )

    if fail_on_regression and b_is_worse:
        raise typer.Exit(2)


@app.command()
def power(
    effect_size: float = typer.Argument(
        ..., help="Effect size to detect (accuracy difference, e.g. 0.05)."
    ),  # noqa: E501
    test: str = typer.Option(
        "proportion", "--test", "-t", help="Test type: proportion, mcnemar, ci, wilcoxon."
    ),  # noqa: E501
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level (Type I error rate)."),
    target_power: float = typer.Option(0.80, "--power", "-p", help="Target statistical power."),
    baseline_accuracy: float = typer.Option(
        0.70, "--baseline", "-b", help="Expected baseline model accuracy."
    ),  # noqa: E501
    observed_n: int | None = typer.Option(
        None, "--observed-n", "-n", help="If set, compute achieved power at this N."
    ),  # noqa: E501
) -> None:
    """
    Compute minimum sample size before running an evaluation.

    Run this BEFORE labelling data or spending API budget.

    Examples:
      evalkit power 0.05            # N needed to detect 5% accuracy gain
      evalkit power 0.05 --observed-n 150   # Power achieved at n=150
      evalkit power 0.03 --test ci   # N for ±3% CI half-width
    """
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=alpha, power=target_power)

    if test == "proportion":
        result = pa.for_proportion_difference(
            effect_size, p1=baseline_accuracy, observed_n=observed_n
        )
    elif test == "mcnemar":
        result = pa.for_mcnemar(effect_size, observed_n=observed_n)
    elif test == "ci":
        if not (0 < effect_size < 1):
            console.print(
                f"[red]For --test ci, effect_size is a CI half-width and must be in (0, 1). "
                f"Got {effect_size}. Example: evalkit power 0.05 --test ci[/red]"
            )
            raise typer.Exit(1)
        result = pa.for_ci_precision(
            effect_size, expected_accuracy=baseline_accuracy, observed_n=observed_n
        )
    elif test == "wilcoxon":
        result = pa.for_wilcoxon(effect_size, observed_n=observed_n)
    else:
        console.print(f"[red]Unknown test '{test}'. Use: proportion, mcnemar, ci, wilcoxon[/red]")
        raise typer.Exit(1)

    table = Table(title="Power Analysis")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="bold cyan")
    table.add_row("Test type", result.test_type)
    table.add_row("Effect size", str(result.effect_size))
    table.add_row("α (Type I error)", str(result.alpha))
    table.add_row("Target power", f"{result.desired_power:.0%}")
    table.add_row("Minimum N required", str(result.minimum_n))

    if result.achieved_power is not None:
        colour = "green" if result.is_adequate else "red"
        status = "✓ adequate" if result.is_adequate else "✗ underpowered"
        table.add_row(
            "Achieved power", f"[{colour}]{result.achieved_power:.3f} ({status})[/{colour}]"
        )

    console.print(table)

    if result.achieved_power is not None and not result.is_adequate:
        console.print(
            Panel(
                f"[yellow]Your n={observed_n} achieves {result.achieved_power:.0%} power, "
                f"below the {target_power:.0%} target.[/yellow]\n"
                f"A negative result here is [bold]inconclusive[/bold], not evidence of equivalence.\n"  # noqa: E501
                f"Collect at least n={result.minimum_n} examples before concluding models are equivalent.",  # noqa: E501
                title="[yellow]⚠ Underpowered Experiment[/yellow]",
                border_style="yellow",
            )
        )


@app.command()
def table(
    test: str = typer.Option(
        "proportion", "--test", "-t", help="Test type: proportion, mcnemar, ci, wilcoxon."
    ),  # noqa: E501
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level."),
    baseline: float = typer.Option(
        0.70, "--baseline", "-b", help="Baseline accuracy for proportion/CI tests."
    ),  # noqa: E501
) -> None:
    """
    Print a sample size planning table across effect sizes and power levels.

    The table shows how many examples you need to detect a given accuracy
    difference at a given statistical power. Screenshot this and paste it
    into your design doc before labelling any data.

    Example:
      evalkit table
      evalkit table --test ci --baseline 0.80
    """
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=alpha)
    pa.sample_size_table(test=test, baseline_accuracy=baseline)


@app.command()
def version(
    cite: bool = typer.Option(False, "--cite", help="Print BibTeX citation entry."),
) -> None:
    """Print evalkit version information."""
    from evalkit import __version__

    console.print(f"evalkit-research [bold cyan]{__version__}[/bold cyan]")

    if cite:
        import datetime

        current_year = datetime.datetime.now().year
        bibtex = (
            "@software{evalkit,\n"
            "  author  = {McConnell, Bonnie},\n"
            "  title   = {evalkit: Rigorous LLM Evaluation},\n"
            f"  year    = {{{current_year}}},\n"
            f"  version = {{{__version__}}},\n"
            "  url     = {https://github.com/bonnie-mcconnell/evalkit},\n"
            "  note    = {Bootstrap CIs, McNemar's test, BH-FDR correction, "
            "automated statistical audit}\n"
            "}"
        )
        console.print()
        console.print("[dim]BibTeX citation:[/dim]")
        console.print(bibtex)


def _print_metrics_table(result: ExperimentResult) -> None:
    """Render the metrics table using Rich."""
    table = Table(title=f"Results - {result.experiment_name}", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold cyan")
    table.add_column("95% CI", style="dim")
    table.add_column("±", style="dim")
    table.add_column("n", style="dim")

    for name, m in result.metrics.items():
        table.add_row(
            name,
            f"{m.value:.4f}",
            f"[{m.ci_lower:.4f}, {m.ci_upper:.4f}]",
            f"±{m.margin_of_error:.4f}",
            str(m.n),
        )

    console.print(table)


def _print_audit(audit: AuditReport) -> None:
    """Render the RigorChecker audit using Rich."""
    if not audit.findings:
        console.print(
            Panel(
                "[green]✅ No issues found. Experiment appears statistically sound.[/green]",
                title="RigorChecker",
                border_style="green",
            )
        )
        return

    severity_colour = {"error": "red", "warning": "yellow", "info": "cyan"}
    lines = []
    for f in sorted(audit.findings, key=_severity_sort_key):
        col = severity_colour[f.severity.value]
        lines.append(f"[{col}][{f.code}][/{col}] {f.message}")
        lines.append(f"  [dim]→ {f.action}[/dim]")
        lines.append("")

    border = "red" if not audit.passed else "yellow" if audit.warnings else "green"
    if not audit.passed:
        status = "FAIL"
        status_colour = "red"
    elif audit.warnings:
        n_warn = len(audit.warnings)
        status = f"PASS ({n_warn} warning{'s' if n_warn != 1 else ''})"
        status_colour = "yellow"
    else:  # pragma: no cover - only fires with INFO-only findings (e.g. --judge llm)
        status = "PASS"
        status_colour = "green"
    title = f"RigorChecker - [{status_colour}]{status}[/{status_colour}]"
    console.print(Panel("\n".join(lines).strip(), title=title, border_style=border))


if __name__ == "__main__":  # pragma: no cover
    app()
