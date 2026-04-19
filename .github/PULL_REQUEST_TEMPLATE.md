## What this PR does

<!-- One paragraph summary. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New metric or statistical test
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation only
- [ ] Refactor (no behaviour change)

## Checklist

- [ ] Tests added or updated - `pytest` passes with no new failures
- [ ] `ruff check evalkit/ tests/ examples/` passes
- [ ] `ruff format --check evalkit/ tests/ examples/` passes
- [ ] `mypy evalkit/` passes (strict mode)
- [ ] `pytest --cov=evalkit --cov-fail-under=85` passes
- [ ] Statistical methods have a reference in the docstring or PR description
- [ ] New public API is exported from `evalkit/__init__.py`
- [ ] CHANGELOG.md updated under `[Unreleased]`

## Statistical correctness (if applicable)

<!-- For new metrics/tests: describe how you verified correctness.
     Ideally include a numerical check against a known result or published value. -->

## Breaking changes (if applicable)

<!-- List any changes to existing public API and how users should migrate. -->
