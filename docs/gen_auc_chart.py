"""Generate the interactive AUC chart embedded in docs/plots/auc.md."""

import numpy as np
import polars as pl

from plotutils.auc import plot_roc_curve

rng = np.random.default_rng(42)

n = 300
labels = rng.integers(0, 2, size=n)
# Positives score higher on average, with enough overlap to give AUC ~ 0.82
scores = rng.normal(loc=labels * 1.2, scale=1.0)
# Clip to [0, 1] so scores feel like probabilities
scores = np.clip((scores - scores.min()) / (scores.max() - scores.min()), 0, 1)

df = pl.DataFrame({"score": scores.tolist(), "label": labels.tolist()})

chart = plot_roc_curve(
    df,
    score_col="score",
    label_col="label",
    specificity_levels=[0.95, 0.90, 0.80],
    width=480,
    height=440,
)

out = "docs/plots/auc_chart.html"
chart.save(out)
print(f"Saved → {out}")
