"""
EvalDataset and PromptTemplate: load and format evaluation data.

EvalDataset is a thin wrapper around a list of examples. Its job is to:
1. Load data from JSONL, CSV, or HuggingFace datasets
2. Validate that required fields are present
3. Provide a PromptTemplate for converting raw examples to model inputs

The PromptTemplate uses Jinja2 for rendering. This is the right choice
because it handles edge cases (None values, whitespace, loops) better than
f-strings, and prompts in production are never simple f-strings.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """
    A single evaluation example.

    Attributes
    ----------
    id:
        Unique identifier. Used to verify alignment in paired tests.
        Auto-generated as a sequential integer if not provided in the data.
    input_fields:
        Raw fields from the source data (question, context, etc.).
    reference:
        The ground-truth answer or label. Required for metric computation.
    metadata:
        Optional extra fields (source, split, difficulty, etc.).
    """

    id: str
    input_fields: dict[str, Any]
    reference: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, template: PromptTemplate) -> str:
        """Render this example into a model-ready prompt string."""
        return template.render(self.input_fields)


class PromptTemplate:
    """
    Jinja2-based template for converting example fields to model prompts.

    Parameters
    ----------
    template_str:
        Jinja2 template string. Use ``{{ field_name }}`` for variable
        substitution. Example:
        ``"Answer the following question.\\n\\nQ: {{ question }}\\nA:"``

    Examples
    --------
    >>> tmpl = PromptTemplate("Q: {{ question }}\\nContext: {{ context }}")
    >>> tmpl.render({"question": "What is 2+2?", "context": "Basic arithmetic."})
    'Q: What is 2+2?\\nContext: Basic arithmetic.'
    """

    def __init__(self, template_str: str) -> None:
        self._env = Environment(undefined=StrictUndefined, autoescape=False)
        try:
            self._template = self._env.from_string(template_str)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e
        self.template_str = template_str

    def render(self, fields: dict[str, Any]) -> str:
        """Render the template with the given fields."""
        try:
            return self._template.render(**fields)
        except Exception as e:
            raise ValueError(
                f"Template rendering failed. Fields provided: {list(fields.keys())}. Error: {e}"
            ) from e

    def validate(self, dataset: EvalDataset) -> list[str]:
        """
        Check that all template variables exist in the dataset's examples.

        Call this before running an expensive evaluation to catch field-name
        mismatches early - before you've spent any API budget.

        Parameters
        ----------
        dataset:
            The EvalDataset to validate against.

        Returns
        -------
        List of error strings. Empty list means the template is compatible
        with the dataset.

        Raises
        ------
        ValueError
            If any template variable is missing from any example, with a
            message showing which variables are missing and which fields
            are actually available.

        Examples
        --------
        >>> template = PromptTemplate("Q: {{ question }}")
        >>> errors = template.validate(dataset)
        >>> if errors:
        ...     print("\\n".join(errors))
        """
        errors = []
        for i, example in enumerate(dataset):
            try:
                self.render(example.input_fields)
            except ValueError as e:
                errors.append(f"Example id={example.id!r}: {e}")
                if len(errors) >= 5:
                    remaining = len(dataset) - i - 1
                    if remaining:
                        errors.append(
                            f"... and potentially {remaining} more. "
                            "Fix the first errors and re-validate."
                        )
                    break
        return errors


class EvalDataset:
    """
    Container for evaluation examples with loading utilities.

    This is not a PyTorch Dataset or a HuggingFace Dataset - it's deliberately
    simpler. Evaluation sets are small enough (hundreds to thousands of examples)
    that lazy loading and batching are unnecessary complexity.

    Parameters
    ----------
    examples:
        List of Example objects.
    name:
        Human-readable dataset name, used in reports.
    """

    def __init__(self, examples: list[Example], name: str = "unnamed") -> None:
        if not examples:
            raise ValueError("EvalDataset cannot be empty.")
        self.examples = examples
        self.name = name
        self._validate_unique_ids()

    def _validate_unique_ids(self) -> None:
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in set(ids) if ids.count(id_) > 1]
            raise ValueError(f"Example IDs must be unique. Duplicates found: {duplicates[:5]}")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    @property
    def references(self) -> list[Any]:
        """All ground-truth labels, in dataset order."""
        return [e.reference for e in self.examples]

    @property
    def ids(self) -> list[str]:
        """All example IDs, in dataset order."""
        return [e.id for e in self.examples]

    def label_distribution(self) -> dict[str, int]:
        """
        Count of each unique reference label.

        Useful for detecting class imbalance before running evaluations.
        Only meaningful for classification tasks with discrete labels.
        """
        dist: dict[str, int] = {}
        for ref in self.references:
            key = str(ref)
            dist[key] = dist.get(key, 0) + 1
        return dist

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        id_field: str = "id",
        reference_field: str = "label",
        name: str | None = None,
    ) -> EvalDataset:
        """
        Load from a JSONL file where each line is a JSON object.

        Parameters
        ----------
        path:
            Path to the .jsonl file.
        id_field:
            JSON key to use as the example ID. If not present in a record,
            the row index is used.
        reference_field:
            JSON key containing the ground-truth label/answer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        examples = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i + 1}: {e}") from e

                if reference_field not in record:
                    raise ValueError(
                        f"Record on line {i + 1} is missing required field '{reference_field}'. "
                        f"Fields present: {list(record.keys())}"
                    )

                example_id = str(record.get(id_field, i))
                reference = record.pop(reference_field)
                record.pop(id_field, None)  # Don't duplicate id in input_fields

                examples.append(
                    Example(
                        id=example_id,
                        input_fields=record,
                        reference=reference,
                    )
                )

        dataset_name = name or path.stem
        logger.info("Loaded %d examples from %s", len(examples), path)
        return cls(examples, name=dataset_name)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        id_field: str = "id",
        reference_field: str = "label",
        name: str | None = None,
    ) -> EvalDataset:
        """Load from a CSV file. All values are treated as strings."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        examples = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if reference_field not in row:
                    raise ValueError(
                        f"CSV is missing required column '{reference_field}'. "
                        f"Columns present: {list(row.keys())}"
                    )
                example_id = row.get(id_field, str(i))
                reference = row.pop(reference_field)
                row.pop(id_field, None)

                examples.append(
                    Example(
                        id=str(example_id),
                        input_fields=dict(row),
                        reference=reference,
                    )
                )

        dataset_name = name or path.stem
        logger.info("Loaded %d examples from %s", len(examples), path)
        return cls(examples, name=dataset_name)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str = "test",
        reference_field: str = "label",
        id_field: str | None = None,
        max_examples: int | None = None,
        name: str | None = None,
    ) -> EvalDataset:
        """
        Load from HuggingFace Datasets hub.

        Parameters
        ----------
        dataset_name:
            HuggingFace dataset identifier, e.g. "glue/sst2".
        split:
            Dataset split to use. Always default to "test" to avoid
            accidentally evaluating on training data.
        max_examples:
            Cap the number of examples (useful for cheap prototyping,
            but note that the RigorChecker will flag small N).
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets is required. pip install datasets")

        logger.info("Loading %s/%s from HuggingFace...", dataset_name, split)
        hf_dataset = load_dataset(dataset_name, split=split)

        if max_examples:
            hf_dataset = hf_dataset.select(range(min(max_examples, len(hf_dataset))))

        examples = []
        for i, record in enumerate(hf_dataset):
            record = dict(record)
            if reference_field not in record:
                raise ValueError(
                    f"Record missing required field '{reference_field}'. "
                    f"Available fields: {list(record.keys())}"
                )
            example_id = str(record.pop(id_field, i)) if id_field else str(i)
            reference = record.pop(reference_field)
            examples.append(
                Example(
                    id=example_id,
                    input_fields=record,
                    reference=reference,
                )
            )

        dataset_name_clean = name or dataset_name.replace("/", "_")
        logger.info("Loaded %d examples from HuggingFace:%s", len(examples), dataset_name)
        return cls(examples, name=dataset_name_clean)

    @classmethod
    def from_list(
        cls,
        records: list[dict[str, Any]],
        reference_field: str = "label",
        id_field: str = "id",
        name: str = "inline",
    ) -> EvalDataset:
        """Create a dataset from a list of dicts. Useful for tests and examples."""
        examples = []
        for i, record in enumerate(records):
            record = dict(record)
            if reference_field not in record:
                raise ValueError(
                    f"Record {i} is missing '{reference_field}'. Fields: {list(record.keys())}"
                )
            example_id = str(record.pop(id_field, i))
            reference = record.pop(reference_field)
            examples.append(
                Example(
                    id=example_id,
                    input_fields=record,
                    reference=reference,
                )
            )
        return cls(examples, name=name)

    def split(
        self,
        test_size: float = 0.2,
        seed: int = 42,
        stratify: bool = True,
    ) -> tuple[EvalDataset, EvalDataset]:
        """
        Split the dataset into train and test subsets.

        Useful for creating a held-out evaluation set from a labelled pool.

        Parameters
        ----------
        test_size:
            Fraction of examples to reserve for the test set. Default 0.2.
        seed:
            Random seed for reproducibility.
        stratify:
            If True, preserve the label distribution in both splits.
            Strongly recommended - unstratified splits on imbalanced data
            can produce test sets with very different class distributions.

        Returns
        -------
        (train_dataset, test_dataset) tuple.

        Examples
        --------
        >>> train, test = dataset.split(test_size=0.2, stratify=True)
        >>> print(f"Train: {len(train)}, Test: {len(test)}")
        """
        import random as _random

        if not (0 < test_size < 1):
            raise ValueError(
                f"test_size must be in (0, 1), got {test_size}. "
                "Example: 0.2 for an 80/20 train/test split."
            )
        rng = _random.Random(seed)

        if stratify:
            # Group examples by label, split within each group, then merge.
            from collections import defaultdict

            groups: dict[str, list[Example]] = defaultdict(list)
            for ex in self.examples:
                groups[str(ex.reference)].append(ex)

            train_examples: list[Example] = []
            test_examples: list[Example] = []
            for group in groups.values():
                shuffled = group[:]
                rng.shuffle(shuffled)
                n_test = max(1, round(len(shuffled) * test_size))
                test_examples.extend(shuffled[:n_test])
                train_examples.extend(shuffled[n_test:])
        else:
            shuffled = self.examples[:]
            rng.shuffle(shuffled)
            n_test = max(1, round(len(shuffled) * test_size))
            test_examples = shuffled[:n_test]
            train_examples = shuffled[n_test:]

        if not train_examples or not test_examples:
            raise ValueError(
                f"Split produced an empty subset. "
                f"Dataset has {len(self)} examples; "
                f"test_size={test_size} requires at least 2 examples per class."
            )

        return (
            EvalDataset(train_examples, name=f"{self.name}_train"),
            EvalDataset(test_examples, name=f"{self.name}_test"),
        )

    def sample(self, n: int, seed: int = 42) -> EvalDataset:
        """
        Return a random sample of n examples without replacement.

        Useful for quick prototyping runs before committing to a full evaluation.
        The RigorChecker will flag the small N if it's underpowered.

        Parameters
        ----------
        n:
            Number of examples to sample.
        seed:
            Random seed.
        """
        import random as _random

        if n >= len(self):
            return self
        rng = _random.Random(seed)
        sampled = rng.sample(self.examples, n)
        return EvalDataset(sampled, name=f"{self.name}_sample{n}")
