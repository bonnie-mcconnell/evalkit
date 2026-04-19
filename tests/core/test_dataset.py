"""
Tests for EvalDataset and PromptTemplate.
"""

import json

import pytest

from evalkit.core.dataset import EvalDataset, Example, PromptTemplate

# ── PromptTemplate ─────────────────────────────────────────────────────────────


def test_template_renders_correctly():
    tmpl = PromptTemplate("Q: {{ question }}\nA:")
    result = tmpl.render({"question": "What is 2+2?"})
    assert result == "Q: What is 2+2?\nA:"


def test_template_missing_field_raises():
    tmpl = PromptTemplate("{{ question }} {{ context }}")
    with pytest.raises(ValueError, match="rendering failed"):
        tmpl.render({"question": "only this"})


def test_template_invalid_syntax_raises():
    with pytest.raises(ValueError, match="Invalid Jinja2"):
        PromptTemplate("{{ unclosed")


def test_template_multiple_fields():
    tmpl = PromptTemplate("Context: {{ context }}\nQ: {{ question }}")
    result = tmpl.render({"context": "Paris is in France.", "question": "Where is Paris?"})
    assert "Paris is in France" in result
    assert "Where is Paris?" in result


# ── EvalDataset ────────────────────────────────────────────────────────────────


def test_dataset_from_list():
    records = [
        {"id": "1", "question": "What is 1+1?", "label": "2"},
        {"id": "2", "question": "What is 2+2?", "label": "4"},
    ]
    ds = EvalDataset.from_list(records)
    assert len(ds) == 2
    assert ds[0].id == "1"
    assert ds[0].reference == "2"
    assert "question" in ds[0].input_fields


def test_dataset_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        EvalDataset(examples=[], name="empty")


def test_dataset_duplicate_ids_raise():
    examples = [
        Example(id="1", input_fields={"q": "a"}, reference="yes"),
        Example(id="1", input_fields={"q": "b"}, reference="no"),
    ]
    with pytest.raises(ValueError, match="unique"):
        EvalDataset(examples=examples)


def test_dataset_from_jsonl(tmp_path):
    data = [
        {"id": "1", "question": "1+1?", "label": "2"},
        {"id": "2", "question": "2+2?", "label": "4"},
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in data))

    ds = EvalDataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds[0].reference == "2"
    assert ds.ids == ["1", "2"]


def test_dataset_from_jsonl_missing_ref_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "1", "question": "what?"}\n')
    with pytest.raises(ValueError, match="missing required field"):
        EvalDataset.from_jsonl(p)


def test_dataset_from_jsonl_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_jsonl("/nonexistent/path/data.jsonl")


def test_dataset_from_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"id": "1", "label": "yes"}\n\n{"id": "2", "label": "no"}\n')
    ds = EvalDataset.from_jsonl(p)
    assert len(ds) == 2


def test_dataset_from_csv_missing_ref_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,question\n1,What?\n")
    with pytest.raises(ValueError, match="missing required column"):
        EvalDataset.from_csv(p)


def test_dataset_from_list_missing_ref_raises():
    with pytest.raises(ValueError, match="missing"):
        EvalDataset.from_list([{"id": "1", "question": "q"}])


def test_dataset_from_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,question,label\n1,What?,yes\n2,How?,no\n")
    ds = EvalDataset.from_csv(p)
    assert len(ds) == 2
    assert ds[0].reference == "yes"


def test_dataset_label_distribution():
    records = [{"label": "yes"}] * 70 + [{"label": "no"}] * 30
    ds = EvalDataset.from_list(records)
    dist = ds.label_distribution()
    assert dist["yes"] == 70
    assert dist["no"] == 30


def test_dataset_references_property():
    records = [{"id": str(i), "label": str(i % 2)} for i in range(10)]
    ds = EvalDataset.from_list(records)
    assert len(ds.references) == 10


def test_dataset_iteration():
    records = [{"id": str(i), "label": "x"} for i in range(5)]
    ds = EvalDataset.from_list(records)
    ids = [ex.id for ex in ds]
    assert ids == ["0", "1", "2", "3", "4"]


def test_example_render():
    ex = Example(
        id="1",
        input_fields={"question": "Capital of France?"},
        reference="Paris",
    )
    tmpl = PromptTemplate("Q: {{ question }}\nA:")
    rendered = ex.render(tmpl)
    assert "Capital of France?" in rendered


def test_dataset_from_csv_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        EvalDataset.from_csv("/nonexistent/data.csv")


def test_dataset_from_huggingface_import_error():
    """from_huggingface should raise ImportError with install hint when datasets not installed."""
    import importlib.util

    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets is installed; import-error path cannot be tested")
    with pytest.raises(ImportError, match="datasets"):
        EvalDataset.from_huggingface("squad", split="validation")


def test_dataset_split_unstratified():
    """Non-stratified split should still produce non-overlapping subsets."""
    records = [{"id": str(i), "label": "x"} for i in range(50)]
    ds = EvalDataset.from_list(records)
    train, test = ds.split(test_size=0.2, stratify=False)
    train_ids = {e.id for e in train}
    test_ids = {e.id for e in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 50


def test_dataset_sample_preserves_name():
    records = [{"id": str(i), "label": "a"} for i in range(30)]
    ds = EvalDataset.from_list(records, name="mydata")
    sampled = ds.sample(10)
    assert "mydata" in sampled.name
    assert "sample" in sampled.name


def test_from_huggingface_raises_import_error_without_datasets(monkeypatch):
    """
    from_huggingface raises ImportError with helpful install message when
    the 'datasets' package is not installed (lines 324-348 in dataset.py).
    """
    import sys

    # Simulate datasets being absent
    monkeypatch.setitem(sys.modules, "datasets", None)  # type: ignore[arg-type]
    with pytest.raises((ImportError, TypeError)):
        EvalDataset.from_huggingface("squad", split="validation")


def test_from_huggingface_body_with_mock(monkeypatch):
    """
    Test the from_huggingface body using a mocked datasets module.
    Covers: happy path, max_examples selection, and missing reference_field error.
    """
    import sys
    from unittest.mock import MagicMock

    # Build a mock dataset with 10 items
    mock_records = [{"question": f"Q{i}", "answer": f"A{i}", "id": str(i)} for i in range(10)]
    mock_hf_dataset = MagicMock()
    mock_hf_dataset.__iter__ = lambda self: iter(mock_records)
    mock_hf_dataset.__len__ = lambda self: len(mock_records)

    # Make select() return a sliced version
    def mock_select(indices):
        sliced = mock_records[: max(indices) + 1]
        m = MagicMock()
        m.__iter__ = lambda self: iter(sliced)
        m.__len__ = lambda self: len(sliced)
        return m

    mock_hf_dataset.select = mock_select

    mock_datasets_module = MagicMock()
    mock_datasets_module.load_dataset.return_value = mock_hf_dataset
    monkeypatch.setitem(sys.modules, "datasets", mock_datasets_module)

    # Happy path with max_examples
    ds = EvalDataset.from_huggingface(
        "squad",
        split="validation",
        reference_field="answer",
        id_field="id",
        max_examples=5,
        name="squad_test",
    )
    assert len(ds) == 5
    assert ds.name == "squad_test"
    assert ds[0].reference == "A0"
    assert "answer" not in ds[0].input_fields
    assert "question" in ds[0].input_fields


def test_from_huggingface_missing_reference_field_raises(monkeypatch):
    """
    from_huggingface raises ValueError when a record is missing the reference field.
    Line 334 in dataset.py.
    """
    import sys
    from unittest.mock import MagicMock

    # Records without 'label' field
    bad_records = [{"question": "Q0", "id": "0"}]
    mock_hf_dataset = MagicMock()
    mock_hf_dataset.__iter__ = lambda self: iter(bad_records)
    mock_hf_dataset.__len__ = lambda self: 1

    mock_datasets_module = MagicMock()
    mock_datasets_module.load_dataset.return_value = mock_hf_dataset
    monkeypatch.setitem(sys.modules, "datasets", mock_datasets_module)

    with pytest.raises(ValueError, match="missing required field"):
        EvalDataset.from_huggingface("squad", split="validation", reference_field="label")
