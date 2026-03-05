# Plot utils

![Coverage](.cccicd/badges/coverage.svg)

A collection of high-level utility functions to facilitate plotting with [Altair](https://altair-viz.github.io/) and [Polars](https://pola.rs/).

## Documentation

Browse the full docs with interactive examples at **https://paulduf.github.io/plotutils/**.

## Installation

```bash
uv pip install -e .
```

## Development

```bash
# Install dependencies
uv sync --group dev
uv pip install -e .

# Run tests
just test

# Run tests with coverage
just coverage

# Build & serve docs locally
just docs-serve
```
