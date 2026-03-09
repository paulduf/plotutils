# just/common.just

set windows-shell := ["cmd.exe", "/C"]

# Run all tests
test *args:
    uv run pytest {{ args }}

# Run tests with coverage (generates .coverage file)
coverage *args:
    uv run coverage run -m pytest {{ args }}

# Print coverage report
coverage-report:
    uv run coverage report --show-missing

# Generate coverage HTML report
coverage-html:
    uv run coverage html

# Generate coverage and tests badges
badge:
    uv run coverage xml -o coverage.xml
    uv run genbadge coverage -i coverage.xml -o .cccicd/badges/coverage.svg
    uv run genbadge tests -i junit.xml -o .cccicd/badges/tests.svg

# Install dependencies
install:
    uv pip install -e .

# Run tests, coverage, and update badge (updates snapshots to ensure consistency)
deploy:
    uv run coverage run -m pytest --snapshot-update --junitxml=junit.xml
    just coverage-report
    just badge
    @echo Deploy complete

# Generate interactive HTML charts into docs/plots/
docs-img:
    uv run python docs/gen_charts.py

# Build documentation site
docs-build: docs-img
    uv run mkdocs build --strict

# Serve documentation locally with live reload
docs-serve: docs-img
    uv run mkdocs serve

# Serve docs without regenerating charts (faster iteration on .md / .py changes)
docs-serve-fast:
    uv run mkdocs serve

# Remove generated docs artifacts
docs-clean:
    if exist docs\img rmdir /s /q docs\img
    if exist _site rmdir /s /q _site

# List available commands
default:
    @just --list
