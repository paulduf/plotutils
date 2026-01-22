# just/common.just
set windows-shell := ["cmd.exe", "/C"]

# Run all tests
test *args:
    uv run pytest {{args}}

# Run tests with coverage (generates .coverage file)
coverage *args:
    uv run coverage run -m pytest {{args}}

# Print coverage report
coverage-report:
    uv run coverage report --show-missing

# Generate coverage HTML report
coverage-html:
    uv run coverage html

# Generate coverage badge
badge:
    uv run coverage-badge -f -o .cccicd/badges/coverage.svg

# Install dependencies
install:
    uv pip install -e .

# Run tests, coverage, and update badge (updates snapshots to ensure consistency)
deploy:
    uv run coverage run -m pytest --snapshot-update
    just coverage-report
    just badge
    @echo Deploy complete

# List available commands
default:
    @just --list
