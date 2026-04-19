"""
Judge classes: evaluate whether a model output is correct.

The type hierarchy reflects a real architectural distinction:
- DeterministicJudge: free, instant, reproducible. ExactMatch, RegexMatch.
- StochasticJudge: expensive, slow, variable. LLMJudge, SemanticSimilarity.

This matters for batching strategy, retry logic, cost tracking, and
inter-rater agreement validation (only applicable to stochastic judges).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class _SupportsComplete(Protocol):
    """Minimal interface required by LLMJudge - any ModelProvider satisfies this."""

    def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    """
    The result of a single judgment.

    Attributes
    ----------
    score:
        Numeric score on [0, 1]. For binary judges: 0.0 or 1.0.
    is_correct:
        Boolean shortcut: score >= threshold.
    reasoning:
        Explanation from the judge (non-empty for LLMJudge).
    raw_output:
        The original model output being judged.
    metadata:
        Judge-specific extras (token usage, latency, etc.).
    """

    score: float
    is_correct: bool
    raw_output: str
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Judge(ABC):
    """
    Abstract base class for all evalkit judges.

    A Judge takes a model output and a reference answer and returns a
    JudgmentResult. Judges are stateless - call judge() multiple times
    without side effects.
    """

    @abstractmethod
    def judge(self, output: str, reference: Any) -> JudgmentResult:
        """
        Evaluate a single model output against the reference.

        Parameters
        ----------
        output:
            The model's raw text output.
        reference:
            The ground-truth answer or label.
        """
        ...

    def judge_batch(self, outputs: list[str], references: list[Any]) -> list[JudgmentResult]:
        """
        Judge a batch of outputs. Default: sequential calls to judge().

        Stochastic judges should override this to enable async batching.
        """
        if len(outputs) != len(references):
            raise ValueError("outputs and references must have the same length.")
        return [self.judge(out, ref) for out, ref in zip(outputs, references)]

    @property
    def is_stochastic(self) -> bool:
        """Whether this judge's outputs vary across calls for the same inputs."""
        return False


class DeterministicJudge(Judge, ABC):
    """
    Base for judges that produce the same result on every call.

    Deterministic judges are free to run and don't require agreement
    validation - they are by definition maximally consistent.
    """


class StochasticJudge(Judge, ABC):
    """
    Base for judges whose outputs may vary across calls.

    LLM judges and embedding-based judges fall here. Before trusting
    a StochasticJudge's scores, inter-rater agreement should be measured
    against either another judge or human raters.
    """

    @property
    def is_stochastic(self) -> bool:
        return True


class ExactMatchJudge(DeterministicJudge):
    """
    Judge that scores 1.0 iff output exactly equals reference after normalisation.

    Normalisation: strip leading/trailing whitespace, optionally lowercase,
    optionally strip punctuation. Overly strict normalisation is a common
    source of false negatives in evaluation.

    Parameters
    ----------
    case_sensitive:
        If False, both output and reference are lowercased before comparison.
    strip_punctuation:
        If True, remove non-alphanumeric characters before comparison.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_punctuation: bool = False,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.strip_punctuation = strip_punctuation

    def _normalize(self, text: str) -> str:
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        if self.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        return text

    def judge(self, output: str, reference: Any) -> JudgmentResult:
        norm_out = self._normalize(str(output))
        norm_ref = self._normalize(str(reference))
        correct = norm_out == norm_ref
        return JudgmentResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=output,
        )


class RegexMatchJudge(DeterministicJudge):
    """
    Judge that scores 1.0 iff output matches a regex pattern.

    Useful for structured outputs where the model should produce a specific
    format (e.g., "Answer: {letter}" for multiple-choice).

    Parameters
    ----------
    pattern:
        Regex pattern to search for in the output.
    extract_group:
        If set, extract this capture group and compare it to reference.
        If None, just check for pattern presence.
    flags:
        re module flags, e.g. re.IGNORECASE.
    """

    def __init__(
        self,
        pattern: str,
        extract_group: int | str | None = 1,
        flags: int = re.IGNORECASE,
    ) -> None:
        try:
            self._pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        self.extract_group = extract_group

    def judge(self, output: str, reference: Any) -> JudgmentResult:
        match = self._pattern.search(output)
        if match is None:
            return JudgmentResult(
                score=0.0,
                is_correct=False,
                raw_output=output,
                reasoning=f"Pattern '{self._pattern.pattern}' not found in output.",
            )

        if self.extract_group is None:
            # Just test presence
            return JudgmentResult(score=1.0, is_correct=True, raw_output=output)

        try:
            extracted = match.group(self.extract_group).strip()
        except IndexError:
            return JudgmentResult(
                score=0.0,
                is_correct=False,
                raw_output=output,
                reasoning=f"Capture group {self.extract_group} not found in match.",
            )

        correct = extracted.lower() == str(reference).lower()
        return JudgmentResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            raw_output=output,
            reasoning=f"Extracted: '{extracted}', Reference: '{reference}'",
        )


class SemanticSimilarityJudge(StochasticJudge):
    """
    Judge based on cosine similarity between sentence embeddings.

    Uses a local embedding model (sentence-transformers) by default, making
    this usable without API keys. The threshold parameter sets the minimum
    similarity for a "correct" judgment.

    Parameters
    ----------
    model_name:
        SentenceTransformers model identifier.
    threshold:
        Cosine similarity threshold above which a response is considered
        correct. 0.85 is a reasonable starting point.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = None  # Lazy load to avoid import cost at startup

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded sentence transformer: %s", self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SemanticSimilarityJudge. "
                    "pip install sentence-transformers"
                ) from None
        return self._model

    def judge(self, output: str, reference: Any) -> JudgmentResult:
        model = self._get_model()
        embeddings = model.encode([str(output), str(reference)], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        correct = similarity >= self.threshold
        return JudgmentResult(
            score=similarity,
            is_correct=correct,
            raw_output=output,
            reasoning=f"Cosine similarity: {similarity:.4f} (threshold: {self.threshold})",
            metadata={"similarity": similarity},
        )


class LLMJudge(StochasticJudge):
    """
    Use a language model to judge output quality.

    The LLMJudge returns a score on [0, 1] by prompting a model to evaluate
    the output against a rubric. Before trusting LLMJudge results, measure
    inter-rater agreement using CohenKappa or KrippendorffAlpha.

    Parameters
    ----------
    provider:
        A ModelProvider instance. The judge model should generally be a
        stronger/different model than the one being evaluated to avoid
        self-serving bias.
    system_prompt:
        The rubric/evaluation criteria. Be specific - vague rubrics are
        the primary cause of low judge agreement.
    score_field:
        The JSON key in the judge's response that contains the numeric score.
    """

    DEFAULT_SYSTEM_PROMPT = """You are an impartial evaluator assessing the quality of a model's response.  # noqa: E501

Evaluate the response against the reference answer using this rubric:
- 1.0: Correct and complete. The response matches the reference in substance.
- 0.5: Partially correct. The response captures some key information but is incomplete or has minor errors.  # noqa: E501
- 0.0: Incorrect. The response contradicts the reference or is entirely off-topic.

Respond ONLY with a JSON object: {"score": <float>, "reasoning": "<one sentence>"}"""

    def __init__(
        self,
        provider: _SupportsComplete,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        score_field: str = "score",
        reasoning_field: str = "reasoning",
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.score_field = score_field
        self.reasoning_field = reasoning_field

    def judge(self, output: str, reference: Any) -> JudgmentResult:
        user_prompt = (
            f"Model response:\n{output}\n\nReference answer:\n{reference}\n\nEvaluate the response."
        )

        raw_response = self.provider.complete(
            messages=[{"role": "user", "content": user_prompt}],
            system=self.system_prompt,
        )

        try:
            # Strip markdown code fences (```json ... ``` or ``` ... ```) if present.
            # str.strip(chars) removes a set of characters, not a substring -
            # using it here would be wrong. Use regex to remove the fence lines.
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip())
            parsed = json.loads(cleaned)
            score = float(parsed[self.score_field])
            reasoning = str(parsed.get(self.reasoning_field, ""))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("LLMJudge failed to parse response: %s. Defaulting to score=0.0.", e)
            score = 0.0
            reasoning = f"Parse error: {e}. Raw response: {raw_response[:200]}"

        score = max(0.0, min(1.0, score))
        return JudgmentResult(
            score=score,
            is_correct=score >= 0.5,
            raw_output=output,
            reasoning=reasoning,
        )
