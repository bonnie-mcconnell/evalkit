"""
evalkit: Rigorous LLM evaluation with bootstrap confidence intervals,
significance testing, and automated statistical auditing.

The central thesis: most LLM evaluation results are statistically meaningless.
evalkit makes rigorous evaluation the default, not an afterthought.
"""

from evalkit.analysis.power import PowerAnalysis, PowerResult
from evalkit.analysis.report import ReportGenerator
from evalkit.analysis.rigour import AuditFinding, AuditReport, RigorChecker, Severity
from evalkit.core.dataset import EvalDataset, Example, PromptTemplate
from evalkit.core.experiment import ComparisonResult, Experiment, ExperimentResult, PreFlightError
from evalkit.core.judge import (
    ContainsJudge,
    DeterministicJudge,
    ExactMatchJudge,
    Judge,
    JudgmentResult,
    LLMJudge,
    RegexMatchJudge,
    SemanticSimilarityJudge,
    StochasticJudge,
)
from evalkit.core.runner import AsyncRunner, ExampleResult, MockRunner, RunResult
from evalkit.metrics.accuracy import (
    Accuracy,
    BalancedAccuracy,
    BLEUScore,
    ExpectedCalibrationError,
    F1Score,
    PrecisionScore,
    RecallScore,
    ROUGEScore,
)
from evalkit.metrics.agreement import AgreementResult, CohenKappa, KrippendorffAlpha
from evalkit.metrics.base import Metric, MetricResult
from evalkit.metrics.comparison import (
    BHCorrection,
    McNemarTest,
    MultipleComparisonResult,
    TestResult,
    WilcoxonTest,
)
from evalkit.providers.base import MockProvider, ModelProvider, ProviderResponse

# Optional providers - only importable when the relevant extra is installed.
# These are exposed at the root namespace so users can write:
#   from evalkit import OpenAIProvider
# and get a clear ImportError if they haven't run pip install evalkit-research[openai].
try:
    from evalkit.providers.base import OpenAIProvider  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # openai not installed; OpenAIProvider not available

try:
    from evalkit.providers.base import AnthropicProvider  # noqa: F401
except ImportError:  # pragma: no cover
    pass  # anthropic not installed; AnthropicProvider not available

try:
    from importlib.metadata import version as _version

    __version__ = _version("evalkit-research")
except Exception:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    # core
    "EvalDataset",
    "Example",
    "PromptTemplate",
    "Experiment",
    "ExperimentResult",
    "ComparisonResult",
    "PreFlightError",
    "Judge",
    "DeterministicJudge",
    "StochasticJudge",
    "JudgmentResult",
    "ExactMatchJudge",
    "RegexMatchJudge",
    "ContainsJudge",
    "LLMJudge",
    "SemanticSimilarityJudge",
    "AsyncRunner",
    "MockRunner",
    "ExampleResult",
    "RunResult",
    "ModelProvider",
    "MockProvider",
    "ProviderResponse",
    # optional providers (available when extras are installed)
    "OpenAIProvider",
    "AnthropicProvider",
    # metrics
    "Metric",
    "MetricResult",
    "Accuracy",
    "BalancedAccuracy",
    "F1Score",
    "PrecisionScore",
    "RecallScore",
    "BLEUScore",
    "ROUGEScore",
    "ExpectedCalibrationError",
    "AgreementResult",
    "CohenKappa",
    "KrippendorffAlpha",
    "McNemarTest",
    "WilcoxonTest",
    "BHCorrection",
    "TestResult",
    "MultipleComparisonResult",
    # analysis
    "PowerAnalysis",
    "PowerResult",
    "RigorChecker",
    "AuditReport",
    "AuditFinding",
    "Severity",
    "ReportGenerator",
]
