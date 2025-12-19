#!/usr/bin/env python3
"""
run_coverage.py
Runs tests with coverage, generates a report and badge, and can be used in CI or locally.
"""

import subprocess
import sys
import shutil


def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode


def main():
    # Run tests with coverage
    run(["uv", "run", "coverage", "run", "-m", "pytest"])

    # Print coverage report to terminal
    run(["uv", "run", "coverage", "report"])

    # Generate HTML report (optional)
    run(["uv", "run", "coverage", "html"])

    # Generate coverage badge if coverage-badge is installed
    if shutil.which("coverage-badge") or shutil.which("uv"):
        try:
            run(
                ["uv", "run", "coverage-badge", "-o", ".cccicd/badges/coverage.svg"],
                check=False,
            )
            print("Coverage badge generated as .cccicd/badges/coverage.svg.")
        except Exception as e:
            print(f"Could not generate badge: {e}")
    else:
        print("coverage-badge not installed. Skipping badge generation.")


if __name__ == "__main__":
    sys.exit(main())
