"""
Model provider abstraction and built-in providers.

ModelProvider is the interface between evalkit's runner and the actual LLM APIs.
Keeping this separate from the runner lets you swap providers without changing
evaluation logic, and lets the mock provider produce reproducible results for
testing and examples that require no API keys.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

logger = logging.getLogger(__name__)


class CostSummary(TypedDict):
    """Typed return value for ModelProvider.cost_summary()."""

    total_cost_usd: float
    total_tokens: int
    call_count: int
    avg_cost_per_call: float


@dataclass
class ProviderResponse:
    """
    Normalised response from any model provider.

    Attributes
    ----------
    content:
        The model's text output.
    input_tokens:
        Number of tokens in the prompt.
    output_tokens:
        Number of tokens in the completion.
    model:
        Model identifier as returned by the API.
    cost_usd:
        Estimated cost in USD. None if the provider doesn't expose pricing.
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelProvider(ABC):
    """
    Abstract interface for LLM providers.

    Implement `_call` in subclasses. The base class handles retries,
    cost accumulation, and logging.
    """

    def __init__(self, model: str, max_retries: int = 3) -> None:
        self.model = model
        self.max_retries = max_retries
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._call_count: int = 0

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        """Make a single API call. No retry logic here - handled by complete()."""
        ...

    def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Make a completion request with retry logic.

        Returns the text content of the response. Cost and token tracking
        are accumulated internally and available via cost_summary().
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._call(messages, system, max_tokens, temperature)
                self._total_cost += response.cost_usd or 0.0
                self._total_tokens += response.total_tokens
                self._call_count += 1
                return response.content
            except Exception as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    "Provider call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    self.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Provider call failed after {self.max_retries} attempts."
        ) from last_error

    def cost_summary(self) -> CostSummary:
        """Total cost and token usage across all calls this session."""
        return CostSummary(
            total_cost_usd=round(self._total_cost, 6),
            total_tokens=self._total_tokens,
            call_count=self._call_count,
            avg_cost_per_call=(
                round(self._total_cost / self._call_count, 6) if self._call_count else 0.0
            ),
        )


class MockProvider(ModelProvider):
    """
    Deterministic mock provider for testing provider-level mechanics.

    Use this for testing retry logic, cost tracking, and token counting
    without making real API calls. It returns deterministic text output
    based on a hash of the input - the same prompt always gets the same
    response, making tests stable.

    Important: MockProvider output will not match any real reference answer,
    so it will always score 0% accuracy when used with an ExactMatchJudge.
    If you need controllable accuracy in tests, use MockRunner instead -
    it operates at the runner level where the reference is accessible.

    Parameters
    ----------
    seed:
        Base random seed for deterministic output generation.
    latency_ms:
        Simulated per-call latency in milliseconds. Useful for testing
        async concurrency and timeout behaviour.
    """

    def __init__(
        self,
        seed: int = 42,
        latency_ms: int = 0,
        model: str = "mock-provider-v1",
    ) -> None:
        super().__init__(model=model)
        self.base_seed = seed
        self.latency_ms = latency_ms

    def _call(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)

        # Deterministic output from input content hash
        content = str(messages) + str(system)
        digest = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
        response_text = f"mock_response_{(self.base_seed + digest) % 10000}"

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        input_tokens = max(1, len(last_user.split()) * 2)
        output_tokens = 4  # "mock_response_XXXX" is ~4 tokens

        return ProviderResponse(
            content=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=0.0,
        )


class OpenAIProvider(ModelProvider):
    """
    OpenAI API provider.

    Requires: pip install openai evalkit-research[openai]

    Parameters
    ----------
    model:
        OpenAI model identifier, e.g. "gpt-4o", "gpt-4o-mini".
    api_key:
        OpenAI API key. Defaults to OPENAI_API_KEY env var.
    """

    # Pricing per 1M tokens as of early 2025 (update as needed)
    _PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "o1": {"input": 15.00, "output": 60.00},
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries)
        try:
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:  # pragma: no cover
            raise ImportError("openai is required. pip install openai")

    def _call(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        from typing import cast

        from openai.types.chat import ChatCompletionMessageParam

        # Build the messages list with the SDK's expected TypedDict type.
        # cast() is necessary because mypy cannot narrow dict[str, str] to the
        # ChatCompletionMessageParam union - the runtime values are always valid.
        all_messages: list[ChatCompletionMessageParam] = []
        if system:
            all_messages.append(
                cast(ChatCompletionMessageParam, {"role": "system", "content": system})
            )
        for m in messages:
            all_messages.append(
                cast(ChatCompletionMessageParam, {"role": m["role"], "content": m["content"]})
            )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = response.usage
        if usage is None:  # pragma: no cover
            raise RuntimeError(
                "OpenAI returned no usage data. This should not happen with a "
                "standard chat completion call - check your API version."
            )
        pricing = self._PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            usage.prompt_tokens * pricing["input"] + usage.completion_tokens * pricing["output"]
        ) / 1_000_000

        return ProviderResponse(
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=self.model,
            cost_usd=cost,
        )


class AnthropicProvider(ModelProvider):
    """
    Anthropic API provider.

    Requires: pip install anthropic evalkit-research[anthropic]

    Parameters
    ----------
    model:
        Anthropic model identifier, e.g. "claude-3-5-sonnet-20241022".
    api_key:
        Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    """

    _PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries)
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:  # pragma: no cover
            raise ImportError("anthropic is required. pip install anthropic")

    def _call(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        # The Anthropic SDK's MessageParam is a TypedDict union that mypy cannot
        # narrow from dict[str, str] - the runtime values always satisfy the union.
        if system:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                system=system,
            )
        else:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )

        usage = response.usage
        pricing = self._PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            usage.input_tokens * pricing["input"] + usage.output_tokens * pricing["output"]
        ) / 1_000_000

        # response.content is a list of content blocks (TextBlock, ToolUseBlock, etc.).
        # Extract only text blocks, joining in order. hasattr check keeps this
        # compatible with test mocks that don't implement the full SDK union type.
        text_content = "".join(
            block.text
            for block in response.content
            if hasattr(block, "text") and isinstance(block.text, str)
        )

        return ProviderResponse(
            content=text_content,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=self.model,
            cost_usd=cost,
        )
