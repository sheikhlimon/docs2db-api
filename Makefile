.PHONY: test test-ci lint format typecheck clean help

help:
	@echo "Available commands:"
	@echo "  make test      - Run all tests with pytest"
	@echo "  make test-ci   - Run CI tests (excluding no_ci marked tests)"
	@echo "  make lint      - Run linters (ruff, pyright)"
	@echo "  make format    - Format code with ruff"
	@echo "  make typecheck - Run pyright type checker"
	@echo "  make clean     - Remove generated files"

test:
	uv run pytest

test-ci:
	uv run pytest -m "not no_ci"

lint:
	uv run ruff check --fix src/ tests/ demos/
	uv run pyright src/docs2db_api/

format:
	uv run ruff format src/ tests/ demos/

typecheck:
	uv run pyright src/docs2db_api/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov coverage.xml .coverage

