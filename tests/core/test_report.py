"""
Tests for ReportGenerator.

We verify structure and safety, not pixel-perfect HTML, since the HTML
template is presentational and changes are expected.
"""

import pytest

from evalkit.analysis.report import ReportGenerator
from evalkit.core.dataset import EvalDataset, PromptTemplate
from evalkit.core.experiment import Experiment
from evalkit.core.judge import ExactMatchJudge
from evalkit.core.runner import MockRunner


@pytest.fixture
def experiment_result():
    records = [{"id": str(i), "question": f"q{i}", "label": f"ans{i % 5}"} for i in range(200)]
    ds = EvalDataset.from_list(records, name="test_ds")
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template, accuracy=0.80)
    return Experiment("report_test", ds, runner).run()


def test_generate_returns_html_string(experiment_result):
    html = ReportGenerator().generate(experiment_result)
    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")


def test_html_contains_experiment_name(experiment_result):
    html = ReportGenerator().generate(experiment_result)
    assert "report_test" in html


def test_html_contains_accuracy_value(experiment_result):
    html = ReportGenerator().generate(experiment_result)
    # Accuracy value should appear somewhere in the metrics table
    acc_value = f"{experiment_result.metrics['Accuracy'].value:.4f}"
    assert acc_value in html


def test_html_contains_ci_bounds(experiment_result):
    html = ReportGenerator().generate(experiment_result)
    acc = experiment_result.metrics["Accuracy"]
    assert f"{acc.ci_lower:.4f}" in html
    assert f"{acc.ci_upper:.4f}" in html


def test_html_contains_rigorchecker_section(experiment_result):
    html = ReportGenerator().generate(experiment_result)
    assert "RigorChecker" in html


def test_html_written_to_file(experiment_result, tmp_path):
    output = tmp_path / "report.html"
    ReportGenerator().generate(experiment_result, output_path=output)
    assert output.exists()
    content = output.read_text()
    assert "<!DOCTYPE html>" in content


def test_html_escapes_special_characters():
    """Experiment names with HTML special chars must not break the output."""
    records = [{"id": str(i), "question": "q", "label": "yes"} for i in range(50)]
    ds = EvalDataset.from_list(records, name="test & <dataset>")
    template = PromptTemplate("{{ question }}")
    runner = MockRunner(judge=ExactMatchJudge(), template=template)
    result = Experiment("test & <experiment>", ds, runner).run()
    html = ReportGenerator().generate(result)
    # The raw angle bracket must not appear unescaped
    assert "<experiment>" not in html
    assert "&lt;experiment&gt;" in html


def test_html_is_self_contained(experiment_result):
    """Report must not reference external stylesheets or scripts - only inline resources.
    Anchor links (e.g. to the GitHub repo in the footer) are fine."""
    html = ReportGenerator().generate(experiment_result)
    # No external CSS or JS resources - all styles are inlined
    assert '<link rel="stylesheet"' not in html
    assert 'src="http' not in html


def test_html_template_tokens_not_double_substituted():
    """
    If an experiment name contains a template token like '{{ model }}', the
    single-pass regex substitution must not expand it a second time.

    With sequential str.replace() the experiment name '{{ model }}' would be
    inserted for {{ experiment_name }}, then the next replace call for
    {{ model }} would expand it to the actual model name - corrupting the output.

    With single-pass re.sub(), both {{ experiment_name }} and {{ model }} are
    replaced in one scan of the original template, so the substituted value
    '{{ model }}' is never re-scanned.
    """
    records = [{"id": str(i), "question": "q", "label": "yes"} for i in range(50)]
    ds = EvalDataset.from_list(records, name="dataset")
    runner = MockRunner(judge=ExactMatchJudge(), template=PromptTemplate("{{ question }}"))
    result = Experiment("{{ model }}", ds, runner).run()
    html = ReportGenerator().generate(result)

    # The model name (mock-model-v1) should appear from the {{ model }} token in the template
    assert "mock-model-v1" in html
    # The experiment name "{{ model }}" should appear as literal text in the header,
    # not replaced by the model name a second time
    assert "evalkit" in html  # The title/header should still be present
    # Crucially: the output should contain the experiment name as text,
    # not have it silently replaced by a second substitution pass
    # We verify by checking the header title contains the literal {{ model }} text
    # (html.escape doesn't change {, }, or spaces)
    header_section = html[html.find("<header>") : html.find("</header>")]
    assert "{{ model }}" in header_section or "&#123;" in header_section


def test_html_contains_no_python_linter_comments(experiment_result):
    """Python linter suppression comments must never appear in generated HTML.

    The HTML template lives inside a triple-quoted Python string. Any
    # noqa: E501 comments appended to long lines inside that string get
    written verbatim into every report and render as visible text.

    The pyproject.toml per-file-ignores already suppresses E501 for
    report.py, so noqa comments are neither needed nor acceptable there.
    """
    html = ReportGenerator().generate(experiment_result)
    assert "# noqa" not in html, (
        "Python linter comment '# noqa' leaked into HTML output. "
        "Check report.py for noqa comments inside the template string."
    )
    assert "# type:" not in html, "Python type comment '# type:' leaked into HTML output."
