# Documentation site: chart generation, build and serve.
# Invoked via the `docs` module — e.g. `just docs serve`, `just docs build`.

# Generate interactive HTML charts into docs/plots/
img:
    uv run python docs/gen_charts.py

# Build documentation site
build: img
    uv run mkdocs build --strict

# Serve documentation locally with live reload
serve: img
    uv run mkdocs serve

# Serve docs without regenerating charts (faster iteration on .md / .py changes)
serve-fast:
    uv run mkdocs serve

# Remove generated docs artifacts
clean:
    rm -rf docs/img _site
