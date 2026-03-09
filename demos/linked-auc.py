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

from plotutils.auc import AUCReport
from plotutils.datasets import load_binary_diabetes, load_synthetic

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
    choices=["synthetic", "synthetic-missing", "synthetic-reversed", "diabetes"],
    default="synthetic",
    help=(
        "Dataset: 'synthetic' (toy, 5 vars / 3 outcomes), "
        "'synthetic-missing' (same with introduced null patterns), "
        "'synthetic-reversed' (adds an anti-correlated variable to demo auto_reverse), or "
        "'diabetes' (sklearn, 10 features / 3 progression quartile outcomes). "
        "Default: synthetic."
    ),
)
args = parser.parse_args()

# ── 1. Dataset ────────────────────────────────────────────────────────────────
if args.dataset in ("synthetic", "synthetic-missing"):
    ds = load_synthetic(missing=args.dataset == "synthetic-missing")
elif args.dataset == "synthetic-reversed":
    ds = load_synthetic(anti_correlated=True)
else:  # diabetes
    ds = load_binary_diabetes()

df, variables, outcomes = ds.df, ds.variables, ds.outcomes

# ── 2. Generate and save report ───────────────────────────────────────────────
AUCReport(
    df,
    variables=variables,
    outcomes=outcomes,
    id_col="patient_id",
    kind=args.kind,
).to_html(path=OUTPUT)

print(f"Saved to {OUTPUT}  (dataset={args.dataset!r}, kind={args.kind!r})")
