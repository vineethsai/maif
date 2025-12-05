# MAIF Makefile

.PHONY: help install install-dev build test lint format clean docs

# Default target
help:
	@echo "MAIF Commands:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make build        - Build the package"
	@echo "  make test         - Run all tests"
	@echo "  make test-quick   - Run quick tests only"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docs         - Build documentation"
	@echo "  make docs-serve   - Serve documentation locally"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-full:
	pip install -e ".[full]"

# Building
build: clean
	python -m build

# Testing
test:
	pytest tests/ -v --timeout=60

test-quick:
	pytest tests/test_core.py tests/test_fixes.py -v --timeout=30

test-cov:
	pytest tests/ -v --cov=maif --cov-report=term-missing --timeout=60

# Code quality
lint:
	ruff check maif/ --select=E9,F63,F7,F82
	@echo "Lint passed!"

format:
	black maif/
	isort maif/

security-check:
	bandit -r maif/ -ll

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Documentation
docs:
	cd docs && npm run docs:build

docs-serve:
	cd docs && npm run docs:dev

# Benchmarks
benchmark:
	python benchmarks/test_maif_benchmark_suite.py
