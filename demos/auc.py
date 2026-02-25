"""Quick demo for plot_roc_curve."""

import numpy as np
import polars as pl

from plotutils.auc import plot_roc_curve

OUTPUT = "demos/_output/auc.html"

rng = np.random.default_rng(0)
n = 400
labels = rng.integers(0, 2, size=n)
# Scores correlated with labels so the curve is non-trivial
scores = rng.normal(loc=labels * 1.5, scale=1.0)

df = pl.DataFrame({"score": scores.tolist(), "label": labels.tolist()})

chart = plot_roc_curve(
    df,
    score_col="score",
    label_col="label",
    specificity_levels=[0.90, 0.75, 0.50],
)

chart.save(OUTPUT)
print(f"Saved to {OUTPUT}")
