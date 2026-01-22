# Tooling Onboarding Guide

## Dependency Management
- Use `uv` for all dependency management and tool invocation.
- Always call tools with `uv run` for portability and reproducibility, e.g.:
  ```sh
  uv run pytest
  uv run coverage run -m pytest
  uv run coverage report
  ```
- Do not use `pip` or `python -m` directly for running tools/scripts.

## Testing
- The test suite uses:
  - **pytest** for running tests
  - **syrupy** for snapshot testing
  - **coverage.py** for code coverage
- Example commands:
  ```sh
  uv run pytest
  uv run coverage run -m pytest
  uv run coverage report
  ```
- For snapshot tests, see the `tests/__snapshots__/` directory.

## Summary
- Always use `uv run` for all Python tooling.
- Tests: pytest + syrupy + coverage.py
- This ensures consistent, portable, and reproducible development workflows.

---
This file is intended for onboarding AI agents and developers working with project tooling in this codebase.
