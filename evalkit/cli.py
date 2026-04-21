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
console = Console()

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
        help="Judge type: exact, regex, or llm. llm requires a real --model (gpt-* or claude-*).",
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
    api_key: str | None = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
) -> None:
    """
    Run an evaluation on a JSONL dataset and produce a tearsheet with CIs.

    Examples:
      evalkit run data.jsonl --model mock --template "Q: {{ question }}"
      evalkit run data.jsonl --model gpt-4o-mini --judge llm --template "Q: {{ question }}"
      evalkit run data.jsonl --model gpt-4o-mini --format json | jq .metrics.Accuracy.value
    """
    from evalkit.analysis.report import ReportGenerator
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.experiment import Experiment
    from evalkit.core.judge import ExactMatchJudge, LLMJudge, RegexMatchJudge
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
    judge_obj: ExactMatchJudge | RegexMatchJudge | LLMJudge
    if judge == "exact":
        judge_obj = ExactMatchJudge()
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
        console.print(f"[red]Unknown judge type '{judge}'. Use: exact, regex, llm[/red]")
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

        provider = AnthropicProvider(model=model)
        runner = AsyncRunner(
            provider=provider,
            judge=judge_obj,
            template=prompt_template,
            concurrency=concurrency,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        console.print(
            f"[red]Unknown model '{model}'. Use: mock, gpt-4o, gpt-4o-mini, claude-*[/red]"
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
    experiment = Experiment(
        name=f"{ds.name}_{model}",
        dataset=ds,
        runner=runner,
        n_resamples=n_resamples,
    )

    with console.status("[cyan]Running evaluation..."):
        result = experiment.run()

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
            rpt = ReportGenerator()
            rpt.generate(result, output_path=output)
            console.print(f"\n[green]✓[/green] Report written to [bold]{output}[/bold]")
        else:
            suggested = dataset.stem + "_report.html"
            console.print(
                f"\n[dim]Tip: add --output {suggested} to generate an HTML tearsheet.[/dim]"
            )

    # Save results JSON for use with evalkit compare
    if save_results:
        run = result.run_result
        save_payload = {
            "model": run.model,
            "dataset": run.dataset_name,
            "n": run.n,
            "example_ids": run.example_ids,
            "correct": run.correct,
            "scores": run.scores,
        }
        save_results.write_text(json.dumps(save_payload, indent=2))
        console.print(f"[green]✓[/green] Run results saved to [bold]{save_results}[/bold]")


@app.command()
def compare(
    result_a: Path = typer.Argument(..., help="First run result JSON file."),
    result_b: Path = typer.Argument(..., help="Second run result JSON file."),
    test: str = typer.Option(
        "mcnemar", "--test", "-t", help="Statistical test: mcnemar or wilcoxon."
    ),  # noqa: E501
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level."),
) -> None:
    """
    Compare two model runs with significance testing.

    Reads run result JSON files written by 'evalkit run --save-results'.
    Applies McNemar's test (binary outcomes) or Wilcoxon signed-rank test
    (continuous scores) and reports the test statistic, p-value, and effect size.

    Example:
      evalkit compare gpt4o_results.json claude_results.json --test mcnemar
    """
    from typing import cast

    from evalkit.metrics.comparison import McNemarTest, TestResult, WilcoxonTest

    def load_results(path: Path) -> dict[str, object]:
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)
        result: dict[str, object] = json.loads(path.read_text())
        return result

    a = load_results(result_a)
    b = load_results(result_b)

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

    decision = (
        "[green]REJECT H₀[/green]" if result.reject_null else "[yellow]fail to reject H₀[/yellow]"
    )  # noqa: E501

    table = Table(title=f"{result.test_name} Test: {result_a.stem} vs {result_b.stem}")
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Test statistic", f"{result.statistic:.4f}")
    table.add_row("p-value", f"{result.p_value:.4f}")
    table.add_row("Effect size", f"{result.effect_size:.4f}")
    table.add_row("n pairs", str(result.n_pairs))
    table.add_row("α", str(result.alpha))
    table.add_row("Decision", decision)
    if result.note:
        table.add_row("Note", result.note)

    console.print(table)


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
def version() -> None:
    """Print evalkit version information."""
    from evalkit import __version__

    console.print(f"evalkit-research [bold cyan]{__version__}[/bold cyan]")


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

    border = "red" if not audit.passed else "yellow"
    title = f"RigorChecker - [{'red' if not audit.passed else 'green'}]{'FAIL' if not audit.passed else 'PASS'}[/{'red' if not audit.passed else 'green'}]"  # noqa: E501
    console.print(Panel("\n".join(lines).strip(), title=title, border_style=border))


if __name__ == "__main__":  # pragma: no cover
    app()