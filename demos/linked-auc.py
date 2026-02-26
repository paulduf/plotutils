"""Linked AUC demo — 2 × 2 interactive grid.

Layout
------
┌─────────────────────┬──────────────────────┐
│  AUC – AUC scatter  │  ROC curve (X axis)  │
│  (click a variable) │                      │
├─────────────────────┼──────────────────────┤
│  Score distribution │  ROC curve (Y axis)  │
│  (var × outcome X   │                      │
│   × outcome Y)      │                      │
└─────────────────────┴──────────────────────┘

Usage
-----
    python demos/linked-auc.py                             # synthetic data, boxplot
    python demos/linked-auc.py --kind strip                # synthetic data, strip plot
    python demos/linked-auc.py --dataset diabetes          # sklearn diabetes dataset
    python demos/linked-auc.py --dataset diabetes --kind strip
"""

import argparse

import numpy as np
import polars as pl

from plotutils.auc import AUCReport

OUTPUT = "demos/_output/linked-auc.html"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Linked AUC explorer demo.")
parser.add_argument(
    "--kind",
    choices=["box", "strip"],
    default="box",
    help="Distribution chart style: 'box' (grouped boxplot) or 'strip' (jittered strip). "
    "Default: box.",
)
parser.add_argument(
    "--dataset",
    choices=["synthetic", "diabetes"],
    default="synthetic",
    help="Dataset: 'synthetic' (toy, 5 vars / 3 outcomes) or 'diabetes' (sklearn, "
    "10 features / 3 progression quartile outcomes). Default: synthetic.",
)
args = parser.parse_args()

# ── 1. Dataset ────────────────────────────────────────────────────────────────
if args.dataset == "synthetic":
    rng = np.random.default_rng(42)
    n = 400

    variables = [f"var_{i}" for i in range(5)]
    outcomes = [f"outcome_{i}" for i in range(3)]

    # Independent binary labels for each outcome
    labels = {out: rng.integers(0, 2, size=n) for out in outcomes}

    # One score per variable — a weighted sum of outcome labels plus noise.
    # Each variable has a different affinity for each outcome (realistic biomarker
    # setup: one measurement, multiple predictions).  Using a single score column
    # per variable means every chart (ROC, distribution) shows the same values,
    # so a patient found at a given threshold on the ROC can be located at the
    # same y-position in the distribution chart.
    strengths = rng.uniform(0.3, 1.5, size=(len(variables), len(outcomes)))

    data: dict = {
        "patient_id": [f"P{i + 1:03d}" for i in range(n)],
        **{out: lbl.tolist() for out, lbl in labels.items()},
    }
    for i, var in enumerate(variables):
        score = sum(strengths[i, j] * labels[out] for j, out in enumerate(outcomes))
        score += rng.normal(0, 1.0, size=n)
        data[var] = score.tolist()

    df = pl.DataFrame(data)

else:  # diabetes
    from sklearn.datasets import load_diabetes  # type: ignore[import-untyped]

    X, y = load_diabetes(return_X_y=True)
    # Feature names are fixed for this dataset (age, sex, bmi, bp, s1–s6).
    variables = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    n = len(y)

    # Binarise the continuous progression target at three quartile thresholds.
    # Higher threshold → fewer positives (more stringent "high progression" label).
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    outcomes = ["prog_q75", "prog_q50", "prog_q25"]

    data = {
        "patient_id": [f"P{i + 1:03d}" for i in range(n)],
        "prog_q75": (y > q75).astype(int).tolist(),
        "prog_q50": (y > q50).astype(int).tolist(),
        "prog_q25": (y > q25).astype(int).tolist(),
    }
    for j, var in enumerate(variables):
        data[var] = X[:, j].tolist()

    df = pl.DataFrame(data)

# ── 2. Generate and save report ───────────────────────────────────────────────
AUCReport(
    df,
    variables=variables,
    outcomes=outcomes,
    id_col="patient_id",
    kind=args.kind,
).to_html(path=OUTPUT)

print(f"Saved to {OUTPUT}  (dataset={args.dataset!r}, kind={args.kind!r})")
