"""
Tests for Judge classes.
"""

from unittest.mock import MagicMock

import pytest

from evalkit.core.judge import ExactMatchJudge, LLMJudge, RegexMatchJudge


def test_exact_match_correct():
    j = ExactMatchJudge()
    r = j.judge("Paris", "Paris")
    assert r.is_correct
    assert r.score == 1.0


def test_exact_match_case_insensitive():
    j = ExactMatchJudge(case_sensitive=False)
    r = j.judge("PARIS", "paris")
    assert r.is_correct


def test_exact_match_case_sensitive_fails():
    j = ExactMatchJudge(case_sensitive=True)
    r = j.judge("PARIS", "paris")
    assert not r.is_correct


def test_exact_match_strips_whitespace():
    j = ExactMatchJudge()
    r = j.judge("  Paris  ", "paris")
    assert r.is_correct


def test_exact_match_wrong_answer():
    j = ExactMatchJudge()
    r = j.judge("London", "Paris")
    assert not r.is_correct
    assert r.score == 0.0


def test_regex_match_extracts_group():
    j = RegexMatchJudge(pattern=r"Answer:\s*([A-D])", extract_group=1)
    r = j.judge("The answer is: Answer: B", "B")
    assert r.is_correct


def test_regex_match_no_match():
    j = RegexMatchJudge(pattern=r"Answer:\s*([A-D])", extract_group=1)
    r = j.judge("I don't know", "B")
    assert not r.is_correct
    assert "not found" in r.reasoning


def test_regex_match_wrong_answer():
    j = RegexMatchJudge(pattern=r"Answer:\s*([A-D])", extract_group=1)
    r = j.judge("Answer: C", "B")
    assert not r.is_correct


def test_regex_invalid_pattern_raises():
    with pytest.raises(ValueError, match="Invalid regex"):
        RegexMatchJudge(pattern="[unclosed")


def test_judge_batch_aligned():
    j = ExactMatchJudge()
    outputs = ["yes", "no", "yes"]
    refs = ["yes", "yes", "yes"]
    results = j.judge_batch(outputs, refs)
    assert len(results) == 3
    assert results[0].is_correct
    assert not results[1].is_correct


def test_judge_batch_misaligned_raises():
    j = ExactMatchJudge()
    with pytest.raises(ValueError):
        j.judge_batch(["a", "b"], ["a"])


def test_exact_match_is_not_stochastic():
    j = ExactMatchJudge()
    assert not j.is_stochastic


def test_exact_match_strip_punctuation():
    j = ExactMatchJudge(strip_punctuation=True)
    r = j.judge("Paris!", "paris")
    assert r.is_correct


def test_exact_match_raw_output_preserved():
    j = ExactMatchJudge()
    r = j.judge("  Paris  ", "paris")
    assert r.raw_output == "  Paris  "  # raw_output is the original, unmodified string


# ── LLMJudge ──────────────────────────────────────────────────────────────────


def test_llm_judge_parses_valid_response():
    """LLMJudge correctly parses a well-formed JSON response from the provider."""
    mock_provider = MagicMock()
    mock_provider.complete.return_value = '{"score": 1.0, "reasoning": "Correct."}'
    j = LLMJudge(provider=mock_provider)
    r = j.judge("The capital of France is Paris.", "Paris")
    assert r.score == 1.0
    assert r.is_correct
    assert "Correct" in r.reasoning


def test_llm_judge_handles_markdown_fenced_json():
    """LLMJudge strips ```json ... ``` fences before parsing - a common LLM quirk."""
    mock_provider = MagicMock()
    mock_provider.complete.return_value = '```json\n{"score": 0.5, "reasoning": "Partial."}\n```'
    j = LLMJudge(provider=mock_provider)
    r = j.judge("Paris is the capital", "Paris")
    assert r.score == 0.5
    # score >= 0.5 is considered correct - partial credit counts
    assert r.is_correct


def test_llm_judge_defaults_to_zero_on_parse_failure():
    """
    When the provider returns malformed JSON, LLMJudge defaults to score=0.0
    rather than raising. This prevents a bad judge response from crashing an
    entire evaluation run - the parse error is logged as a warning.
    """
    mock_provider = MagicMock()
    mock_provider.complete.return_value = "I cannot evaluate this."
    j = LLMJudge(provider=mock_provider)
    r = j.judge("some output", "some reference")
    assert r.score == 0.0
    assert not r.is_correct
    assert "Parse error" in r.reasoning


def test_llm_judge_clamps_out_of_range_scores():
    """Scores outside [0, 1] from the provider are clamped, not passed through."""
    mock_provider = MagicMock()
    mock_provider.complete.return_value = '{"score": 1.5, "reasoning": "Very good."}'
    j = LLMJudge(provider=mock_provider)
    r = j.judge("output", "reference")
    assert r.score == 1.0


def test_llm_judge_is_stochastic():
    """LLMJudge must be identified as stochastic - required for RigorChecker agreement checks."""
    mock_provider = MagicMock()
    j = LLMJudge(provider=mock_provider)
    assert j.is_stochastic


# ── SemanticSimilarityJudge ───────────────────────────────────────────────────


def test_semantic_similarity_judge_raises_without_sentence_transformers(monkeypatch):
    """
    SemanticSimilarityJudge raises ImportError with a clear install message
    when sentence-transformers is not installed. This is the primary failure
    mode users encounter when they forget the optional dep.
    """
    import sys

    # Remove sentence_transformers from sys.modules so the lazy import fails
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore[arg-type]

    from evalkit.core.judge import SemanticSimilarityJudge

    j = SemanticSimilarityJudge()
    j._model = None  # reset cached model

    with pytest.raises(ImportError, match="sentence-transformers"):
        j.judge("output", "reference")


# ── RegexMatchJudge edge cases ─────────────────────────────────────────────────


def test_regex_presence_only_no_extract_group():
    """
    When extract_group=None, the judge checks presence only - any match scores 1.0
    regardless of what was captured. This is useful for format checking.
    """
    j = RegexMatchJudge(pattern=r"Answer:\s*[A-D]", extract_group=None)
    r = j.judge("The answer is: Answer: B", reference="anything")
    assert r.is_correct
    assert r.score == 1.0


def test_regex_presence_only_no_match_scores_zero():
    j = RegexMatchJudge(pattern=r"Answer:\s*[A-D]", extract_group=None)
    r = j.judge("I don't know", reference="anything")
    assert not r.is_correct
    assert r.score == 0.0


def test_regex_invalid_capture_group_number():
    """
    If a regex matches but the requested capture group doesn't exist,
    the judge should return score=0.0 with a clear reasoning message.
    """
    # Pattern has only 1 group (group 1), but we request group 2
    j = RegexMatchJudge(pattern=r"Answer:\s*([A-D])", extract_group=2)
    r = j.judge("Answer: B", reference="B")
    assert not r.is_correct
    assert r.score == 0.0
    assert "not found" in r.reasoning.lower() or "group" in r.reasoning.lower()
