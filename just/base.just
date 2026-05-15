# Tests, coverage, installation and release helpers.

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
