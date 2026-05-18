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


def test_llm_judge_default_prompt_contains_no_noqa_comments():
    """
    Regression test: LLMJudge.DEFAULT_SYSTEM_PROMPT must not contain
    '# noqa' linter suppression comments.

    The prompt is passed verbatim to a language model as the evaluation
    rubric. Any embedded linter comments corrupt the rubric and produce
    meaningless judge scores.
    """
    assert "# noqa" not in LLMJudge.DEFAULT_SYSTEM_PROMPT, (
        "Python linter comment '# noqa' found in LLMJudge.DEFAULT_SYSTEM_PROMPT. "
        "It will be sent to the model as part of the evaluation rubric."
    )
    assert "# type:" not in LLMJudge.DEFAULT_SYSTEM_PROMPT, (
        "Python type comment '# type:' found in LLMJudge.DEFAULT_SYSTEM_PROMPT."
    )


# ── ContainsJudge ──────────────────────────────────────────────────────────────


def test_contains_judge_correct_when_present():
    """Score 1.0 when reference appears in output."""
    from evalkit.core.judge import ContainsJudge

    j = ContainsJudge()
    result = j.judge("The answer is Paris, France.", "Paris")
    assert result.is_correct
    assert result.score == 1.0


def test_contains_judge_incorrect_when_absent():
    """Score 0.0 when reference does not appear in output."""
    from evalkit.core.judge import ContainsJudge

    j = ContainsJudge()
    result = j.judge("I don't know.", "Paris")
    assert not result.is_correct
    assert result.score == 0.0


def test_contains_judge_case_insensitive_by_default():
    """Default is case-insensitive."""
    from evalkit.core.judge import ContainsJudge

    j = ContainsJudge()
    assert j.judge("The ANSWER is yes.", "yes").is_correct
    assert j.judge("The ANSWER is yes.", "YES").is_correct
    assert j.judge("The ANSWER is yes.", "Yes").is_correct


def test_contains_judge_case_sensitive_mode():
    """case_sensitive=True performs exact substring match."""
    from evalkit.core.judge import ContainsJudge

    j = ContainsJudge(case_sensitive=True)
    assert j.judge("The answer is Yes.", "Yes").is_correct
    assert not j.judge("The answer is Yes.", "yes").is_correct


def test_contains_judge_strips_whitespace_by_default():
    """Reference with leading/trailing whitespace still matches."""
    from evalkit.core.judge import ContainsJudge

    j = ContainsJudge()
    assert j.judge("  paris  ", "  paris  ").is_correct
    assert j.judge("paris", "  paris  ").is_correct


def test_contains_judge_exported():
    """ContainsJudge must be importable from the top-level evalkit package."""
    from evalkit import ContainsJudge  # noqa: F401


def test_contains_judge_cli_option(tmp_path):
    """evalkit run --judge contains must work end-to-end."""
    import json

    from typer.testing import CliRunner

    from evalkit.cli import app

    # n must be >= 30 (RigorChecker min_n) to avoid pre-flight halt in strict mode.
    # Alternate Paris/Berlin so the data is interesting for contains judge.
    records = [
        {"id": str(i), "text": "The capital is Paris indeed.", "label": "Paris"} for i in range(20)
    ] + [{"id": str(i + 20), "text": "I have no idea.", "label": "Paris"} for i in range(10)]
    data_file = tmp_path / "data.jsonl"
    data_file.write_text("\n".join(json.dumps(r) for r in records))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(data_file),
            "--model",
            "mock",
            "--judge",
            "contains",
            "--template",
            "{{ text }}",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
