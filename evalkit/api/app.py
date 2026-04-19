"""
evalkit REST API.

Endpoints:
  POST /runs          - start an evaluation run
  GET  /runs/{id}     - get run status and results
  POST /compare       - compare two runs with significance test
  POST /power         - compute required sample size
  GET  /health        - health check

The API is stateless across restarts (results stored in ./results/).
This is intentional - eval results are artifacts, not ephemeral state.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create the results directory on startup."""
    RESULTS_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="evalkit API",
    description="Rigorous LLM evaluation with bootstrap CIs and automated auditing.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response models ────────────────────────────────────────────────────


class RunRequest(BaseModel):
    dataset_records: list[dict[str, Any]] = Field(
        description="List of evaluation examples. Each must have a 'label' field."
    )
    model: str = Field(default="mock", description="Model to evaluate: mock, gpt-4o, etc.")
    template: str = Field(default="{{ question }}", description="Jinja2 prompt template.")
    reference_field: str = Field(default="label")
    mock_accuracy: float = Field(default=0.82, ge=0.0, le=1.0)
    n_resamples: int = Field(default=5000, ge=100)


class CompareRequest(BaseModel):
    run_id_a: str
    run_id_b: str
    test: str = Field(default="mcnemar", description="mcnemar or wilcoxon")
    alpha: float = Field(default=0.05)


class PowerRequest(BaseModel):
    effect_size: float
    test: str = Field(default="proportion")
    alpha: float = Field(default=0.05)
    target_power: float = Field(default=0.80)
    baseline_accuracy: float = Field(default=0.70)
    observed_n: int | None = None


# ── Background tasks ───────────────────────────────────────────────────────────


def _run_evaluation(run_id: str, request: RunRequest) -> None:
    """Run evaluation in the background and persist results."""
    from evalkit.analysis.report import ReportGenerator
    from evalkit.core.dataset import EvalDataset, PromptTemplate
    from evalkit.core.experiment import Experiment
    from evalkit.core.judge import ExactMatchJudge
    from evalkit.core.runner import MockRunner

    result_path = RESULTS_DIR / f"{run_id}.json"

    try:
        dataset = EvalDataset.from_list(
            request.dataset_records,
            reference_field=request.reference_field,
        )
        template = PromptTemplate(request.template)
        judge = ExactMatchJudge()

        if request.model == "mock":
            runner = MockRunner(judge=judge, template=template, accuracy=request.mock_accuracy)
        else:
            raise ValueError(f"Model '{request.model}' not yet supported via API. Use 'mock'.")

        experiment = Experiment(
            name=run_id,
            dataset=dataset,
            runner=runner,
        )
        result = experiment.run()

        # Persist results
        summary = result.run_result.summary()
        summary["metrics"] = {
            name: {
                "value": m.value,
                "ci_lower": m.ci_lower,
                "ci_upper": m.ci_upper,
                "n": m.n,
            }
            for name, m in result.metrics.items()
        }
        summary["audit_passed"] = result.posthoc_audit.passed
        summary["audit_findings"] = [
            {"code": f.code, "severity": f.severity.value, "message": f.message}
            for f in result.posthoc_audit.findings
        ]
        summary["correct"] = result.run_result.correct
        summary["scores"] = result.run_result.scores
        summary["example_ids"] = result.run_result.example_ids
        summary["status"] = "complete"

        result_path.write_text(json.dumps(summary, indent=2))

        # Also generate HTML report
        report_path = RESULTS_DIR / f"{run_id}.html"
        ReportGenerator().generate(result, output_path=report_path)

    except Exception as e:
        logger.error("Run %s failed: %s", run_id, e)
        result_path.write_text(json.dumps({"status": "failed", "error": str(e)}))


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/runs", status_code=202)
def start_run(request: RunRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Start an evaluation run asynchronously."""
    run_id = str(uuid.uuid4())
    result_path = RESULTS_DIR / f"{run_id}.json"
    result_path.write_text(json.dumps({"status": "running", "run_id": run_id}))

    background_tasks.add_task(_run_evaluation, run_id, request)
    return {"run_id": run_id, "status": "running"}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Get run status and results."""
    result_path = RESULTS_DIR / f"{run_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return json.loads(result_path.read_text())  # type: ignore[no-any-return]


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
def get_report(run_id: str) -> HTMLResponse:
    """Get the HTML tearsheet for a completed run."""
    report_path = RESULTS_DIR / f"{run_id}.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not yet available.")
    return HTMLResponse(content=report_path.read_text())


@app.post("/compare")
def compare_runs(request: CompareRequest) -> dict[str, Any]:
    """Compare two run results with statistical significance testing."""
    from evalkit.metrics.comparison import McNemarTest, WilcoxonTest

    def load(run_id: str) -> dict[str, Any]:
        p = RESULTS_DIR / f"{run_id}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
        data: dict[str, Any] = json.loads(p.read_text())
        if data.get("status") != "complete":
            raise HTTPException(status_code=400, detail=f"Run {run_id} is not complete.")
        return data

    a = load(request.run_id_a)
    b = load(request.run_id_b)

    if a["example_ids"] != b["example_ids"]:
        raise HTTPException(
            status_code=400,
            detail="Runs use different example sets. Paired tests require the same examples.",
        )

    if request.test == "mcnemar":
        result = McNemarTest(alpha=request.alpha).test(a["correct"], b["correct"])
    elif request.test == "wilcoxon":
        result = WilcoxonTest(alpha=request.alpha).test(a["scores"], b["scores"])
    else:
        raise HTTPException(status_code=400, detail="test must be 'mcnemar' or 'wilcoxon'")

    return {
        "test": result.test_name,
        "statistic": float(result.statistic),
        "p_value": float(result.p_value),
        "effect_size": float(result.effect_size),
        "reject_null": bool(result.reject_null),
        "alpha": float(result.alpha),
        "n_pairs": int(result.n_pairs),
        "note": result.note,
        "interpretation": (
            f"The difference between {request.run_id_a} and {request.run_id_b} is "
            + (
                "statistically significant."
                if result.reject_null
                else "NOT statistically significant."
            )  # noqa: E501
        ),
    }


@app.post("/power")
def power_analysis(request: PowerRequest) -> dict[str, Any]:
    """Compute required sample size for a given effect size and power."""
    from evalkit.analysis.power import PowerAnalysis

    pa = PowerAnalysis(alpha=request.alpha, power=request.target_power)

    if request.test == "proportion":
        result = pa.for_proportion_difference(
            request.effect_size, p1=request.baseline_accuracy, observed_n=request.observed_n
        )
    elif request.test == "mcnemar":
        result = pa.for_mcnemar(request.effect_size, observed_n=request.observed_n)
    elif request.test == "ci":
        result = pa.for_ci_precision(
            request.effect_size,
            expected_accuracy=request.baseline_accuracy,
            observed_n=request.observed_n,
        )
    elif request.test == "wilcoxon":
        result = pa.for_wilcoxon(request.effect_size, observed_n=request.observed_n)
    else:
        raise HTTPException(status_code=400, detail="Unknown test type.")

    return {
        "test_type": result.test_type,
        "effect_size": result.effect_size,
        "alpha": result.alpha,
        "target_power": result.desired_power,
        "minimum_n": result.minimum_n,
        "achieved_power": result.achieved_power,
        "is_adequate": result.is_adequate,
    }
