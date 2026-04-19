"""
EvalRunner: Execute a model over an EvalDataset and collect outputs.

Key design decisions:
1. Async-first: all real runners are async with rate limiting.
2. Checkpointing: partial results are written to disk after each batch.
   If a 500-example eval crashes at example 487, resume from checkpoint.
3. Cost tracking: every run accumulates token usage and estimated cost.
4. MockRunner: deterministic seeded outputs for CI and examples with no API keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalkit.core.dataset import EvalDataset, Example, PromptTemplate
from evalkit.core.judge import Judge, JudgmentResult
from evalkit.providers.base import ModelProvider

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExampleResult:
    """
    The complete result for a single evaluation example.

    Attributes
    ----------
    example_id:
        Links back to Example.id for alignment verification.
    prompt:
        The rendered prompt sent to the model.
    output:
        Raw model output.
    reference:
        Ground truth answer.
    judgment:
        JudgmentResult from the configured judge.
    latency_ms:
        Wall-clock time for this call in milliseconds.
    metadata:
        Provider-specific extras (token counts, cost, etc.).
    """

    example_id: str
    prompt: str
    output: str
    reference: Any
    judgment: JudgmentResult
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        return self.judgment.is_correct

    @property
    def score(self) -> float:
        return self.judgment.score


@dataclass
class RunResult:
    """
    Aggregated results from a complete evaluation run.

    Attributes
    ----------
    example_results:
        Per-example results, in dataset order.
    model:
        Model identifier.
    dataset_name:
        Name of the evaluated dataset.
    total_cost_usd:
        Estimated total API cost for this run.
    total_tokens:
        Total tokens consumed (input + output).
    wall_time_seconds:
        Total elapsed wall-clock time.
    """

    example_results: list[ExampleResult]
    model: str
    dataset_name: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.example_results)

    @property
    def correct(self) -> list[int]:
        """Binary correctness array. Aligned with example_results."""
        return [1 if r.is_correct else 0 for r in self.example_results]

    @property
    def scores(self) -> list[float]:
        """Continuous score array. Aligned with example_results."""
        return [r.score for r in self.example_results]

    @property
    def example_ids(self) -> list[str]:
        return [r.example_id for r in self.example_results]

    @property
    def references(self) -> list[Any]:
        return [r.reference for r in self.example_results]

    @property
    def outputs(self) -> list[str]:
        return [r.output for r in self.example_results]

    def cost_per_correct(self) -> float | None:
        """Cost-effectiveness: USD per correctly answered example."""
        n_correct = sum(self.correct)
        if n_correct == 0 or self.total_cost_usd == 0:
            return None
        return self.total_cost_usd / n_correct

    def summary(self) -> dict[str, Any]:
        raw_accuracy = sum(self.correct) / self.n if self.n else 0.0
        return {
            "model": self.model,
            "dataset": self.dataset_name,
            "n": self.n,
            "raw_accuracy": raw_accuracy,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "wall_time_s": round(self.wall_time_seconds, 2),
            "cost_per_correct": self.cost_per_correct(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return per-example results as a pandas DataFrame.

        Columns: example_id, prompt, output, reference, is_correct, score,
                 reasoning, latency_ms.

        This method is also available on ExperimentResult. Use RunResult.to_dataframe()
        when you want raw results without running the full Experiment pipeline.

        Requires pandas. Raises ImportError with install instructions if
        pandas is not available.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas") from None

        rows = [
            {
                "example_id": r.example_id,
                "prompt": r.prompt,
                "output": r.output,
                "reference": r.reference,
                "is_correct": r.is_correct,
                "score": r.score,
                "reasoning": r.judgment.reasoning,
                "latency_ms": r.latency_ms,
            }
            for r in self.example_results
        ]
        return pd.DataFrame(rows)


class AsyncRunner:
    """
    Evaluate a model over an EvalDataset with async concurrency and checkpointing.

    Parameters
    ----------
    provider:
        ModelProvider instance for making API calls.
    judge:
        Judge instance for evaluating outputs.
    template:
        PromptTemplate for converting examples to model inputs.
    concurrency:
        Maximum concurrent API calls. Higher values are faster but may
        hit rate limits. Default 5 is conservative.
    checkpoint_dir:
        Directory for saving partial results. If None, checkpointing
        is disabled. Strongly recommended for large evaluations.
    checkpoint_every:
        Save a checkpoint after every N examples. Default 50.
    system_prompt:
        Optional system prompt sent to the model on every call.
    max_tokens:
        Maximum tokens to generate per call.
    temperature:
        Sampling temperature. Use 0.0 for deterministic outputs.
    """

    def __init__(
        self,
        provider: ModelProvider,
        judge: Judge,
        template: PromptTemplate,
        concurrency: int = 5,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 50,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.judge = judge
        self.template = template
        self.concurrency = concurrency
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, dataset: EvalDataset) -> RunResult:
        """
        Synchronous entry point. Runs the async runner in an event loop.

        For use in scripts and CLIs. In async contexts, call `run_async` directly.
        """
        return asyncio.run(self.run_async(dataset))

    async def run_async(self, dataset: EvalDataset) -> RunResult:
        """
        Execute the evaluation asynchronously.

        Resumes from checkpoint if one exists for this dataset.
        """
        start_time = time.monotonic()
        completed: dict[str, ExampleResult] = {}

        # Resume from checkpoint if available
        checkpoint_path = self._checkpoint_path(dataset.name)
        if checkpoint_path and checkpoint_path.exists():
            completed = self._load_checkpoint(checkpoint_path)
            logger.info(
                "Resuming from checkpoint: %d/%d examples already done.",
                len(completed),
                len(dataset),
            )

        remaining = [ex for ex in dataset if ex.id not in completed]
        logger.info(
            "Running evaluation: %d examples, model=%s, concurrency=%d",
            len(remaining),
            self.provider.model,
            self.concurrency,
        )

        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [self._run_example(ex, semaphore) for ex in remaining]

        results_list: list[ExampleResult] = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results_list.append(result)
            completed[result.example_id] = result

            mid_run = (
                self.checkpoint_dir
                and checkpoint_path is not None
                and (i + 1) % self.checkpoint_every == 0
            )
            if mid_run:
                # checkpoint_path is always set when mid_run is True (mid_run checks
                # checkpoint_dir and checkpoint_path is not None), but mypy needs
                # the explicit guard - assert is wrong here because it's disabled
                # by Python -O and is inappropriate in library code.
                if checkpoint_path is None:  # pragma: no cover
                    raise RuntimeError("checkpoint_path is None despite checkpoint_dir being set")
                self._save_checkpoint(checkpoint_path, completed)
                logger.debug("Checkpoint saved: %d/%d done.", len(completed), len(dataset))

        # Final checkpoint
        if self.checkpoint_dir and checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, completed)

        # Reconstruct in original dataset order (as_completed gives arbitrary order)
        ordered = [completed[ex.id] for ex in dataset if ex.id in completed]

        cost_summary = self.provider.cost_summary()
        wall_time = time.monotonic() - start_time

        logger.info(
            "Evaluation complete. n=%d, cost=$%.4f, time=%.1fs",
            len(ordered),
            cost_summary["total_cost_usd"],
            wall_time,
        )

        return RunResult(
            example_results=ordered,
            model=self.provider.model,
            dataset_name=dataset.name,
            total_cost_usd=cost_summary["total_cost_usd"],
            total_tokens=cost_summary["total_tokens"],
            wall_time_seconds=wall_time,
        )

    async def _run_example(self, example: Example, semaphore: asyncio.Semaphore) -> ExampleResult:
        async with semaphore:
            prompt = example.render(self.template)
            t0 = time.monotonic()

            # Run the synchronous provider call in a thread pool to avoid
            # blocking the event loop.
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self.provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system=self.system_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )

            latency_ms = (time.monotonic() - t0) * 1000
            judgment = self.judge.judge(output, example.reference)

            return ExampleResult(
                example_id=example.id,
                prompt=prompt,
                output=output,
                reference=example.reference,
                judgment=judgment,
                latency_ms=latency_ms,
            )

    def _checkpoint_path(self, dataset_name: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        return self.checkpoint_dir / f"{safe_name}_{self.provider.model}_checkpoint.jsonl"

    def _save_checkpoint(self, path: Path, completed: dict[str, ExampleResult]) -> None:
        # Write to a temp file first, then rename atomically.
        # This prevents a corrupt checkpoint if the process is killed mid-write.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for result in completed.values():
                record = {
                    "example_id": result.example_id,
                    "prompt": result.prompt,
                    "output": result.output,
                    # reference is serialised as string; type is not preserved across
                    # checkpoint resume. Judges that call str() on the reference are safe.
                    "reference": str(result.reference),
                    "score": result.score,
                    "is_correct": result.is_correct,
                    "latency_ms": result.latency_ms,
                }
                f.write(json.dumps(record) + "\n")
        tmp.replace(path)

    def _load_checkpoint(self, path: Path) -> dict[str, ExampleResult]:
        completed = {}
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Corrupt checkpoint record at line %d: %s. Skipping.", lineno, e)
                    continue
                judgment = JudgmentResult(
                    score=record["score"],
                    is_correct=record["is_correct"],
                    raw_output=record["output"],
                )
                completed[record["example_id"]] = ExampleResult(
                    example_id=record["example_id"],
                    prompt=record["prompt"],
                    output=record["output"],
                    reference=record["reference"],
                    judgment=judgment,
                    latency_ms=record["latency_ms"],
                )
        return completed


class MockRunner:
    """
    Deterministic mock runner for tests and CI pipelines that require no API keys.

    Directly produces ExampleResult objects with seeded, controllable accuracy.
    The mock operates at the runner level (not provider level) because "correct"
    is only meaningful relative to the reference answer - which the provider
    layer has no access to.

    The mock output on a correct example is the reference answer itself, so
    any judge that accepts the reference as correct will score it as 1.0.
    On incorrect examples, it returns a deterministic but wrong value.

    Parameters
    ----------
    judge:
        Judge to use for scoring (same judge you'd use with a real runner).
    template:
        PromptTemplate for rendering examples (populates ExampleResult.prompt).
    accuracy:
        Fraction of examples to answer correctly. Default 0.82.
    seed:
        Base random seed. Same seed + same dataset → same results always.
    """

    def __init__(
        self,
        judge: Judge,
        template: PromptTemplate,
        accuracy: float = 0.82,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= accuracy <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
        self.judge = judge
        self.template = template
        self.accuracy = accuracy
        self.seed = seed
        self.provider = _MockProviderStub()

    def run(self, dataset: EvalDataset) -> RunResult:
        """Execute the mock evaluation synchronously."""
        start = time.monotonic()
        results = []

        for example in dataset:
            prompt = example.render(self.template)

            # Deterministic per-example seed derived from the base seed and example id.
            # Same seed + same dataset always produces the same results.
            digest = int(hashlib.md5(f"{self.seed}:{example.id}".encode()).hexdigest()[:8], 16)
            is_correct = (digest % 10000) < int(self.accuracy * 10000)

            output = str(example.reference) if is_correct else f"__wrong_{digest % 100}__"
            judgment = self.judge.judge(output, example.reference)

            results.append(
                ExampleResult(
                    example_id=example.id,
                    prompt=prompt,
                    output=output,
                    reference=example.reference,
                    judgment=judgment,
                    latency_ms=0.0,
                )
            )

        return RunResult(
            example_results=results,
            model="mock-model-v1",
            dataset_name=dataset.name,
            total_cost_usd=0.0,
            total_tokens=0,
            wall_time_seconds=time.monotonic() - start,
        )


@dataclass
class _MockProviderStub:
    """Minimal stub so Experiment can read runner.provider.model without crashing."""

    model: str = "mock-model-v1"
