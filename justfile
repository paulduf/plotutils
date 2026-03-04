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

# Copy test snapshot SVGs into docs/img/ so mkdocs can reference them
docs-img:
    mkdir -p docs/img
    cp tests/__snapshots__/test_grouped_hist/test_plot_grouped_histogram_dict_input.svg docs/img/
    cp tests/__snapshots__/test_uncertainty/test_plot_confidence_scatter_stdev_extent.svg docs/img/
    cp tests/__snapshots__/test_uncertainty/test_plot_deviations_basic.svg docs/img/
    cp tests/__snapshots__/test_uncertainty/test_plot_deviations_with_levels.svg docs/img/
    cp tests/__snapshots__/test_uncertainty/test_plot_predictions_errors_basic.svg docs/img/
    cp tests/__snapshots__/test_uncertainty/test_plot_predictions_errors_color_and_shape.svg docs/img/
    cp tests/__snapshots__/test_vchart/test_hchart_with_row_facet.svg docs/img/
    cp tests/__snapshots__/test_vchart/test_vchart_with_column_facet.svg docs/img/
    cp tests/__snapshots__/test_vchart/test_hchart_predictions_errors_three_splits.svg docs/img/

# Build documentation site
docs-build: docs-img
    uv run mkdocs build --strict

# Serve documentation locally with live reload
docs-serve: docs-img
    uv run mkdocs serve

# Remove generated docs artifacts
docs-clean:
    rm -rf docs/img _site

# List available commands
default:
    @just --list
