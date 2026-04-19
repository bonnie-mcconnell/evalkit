.PHONY: install test fmt lint typecheck check demo clean help

## Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/## //'

## Install in editable mode with all dev dependencies
install:
	pip install -e ".[dev]"

## Run the full test suite (fast - no coverage)
test:
	pytest -q

## Format all code with ruff
fmt:
	ruff format evalkit/ tests/ examples/

## Lint all code with ruff (library + tests + examples)
lint:
	ruff check evalkit/ tests/ examples/
	ruff format --check evalkit/ tests/ examples/

## Type-check with mypy (strict)
typecheck:
	mypy evalkit/

## Run lint + typecheck + tests with coverage (what CI runs)
check: lint typecheck
	pytest --cov=evalkit --cov-fail-under=85 -q

## Run the zero-API-key demo end-to-end
demo:
	python examples/full_workflow.py

## Remove build artifacts and caches
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
