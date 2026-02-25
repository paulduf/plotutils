# ROC Curve & AUC

Plot a Receiver Operating Characteristic (ROC) curve from classifier scores and binary labels.
The x-axis is **sensitivity** (true positive rate) and the y-axis is **specificity**
(true negative rate). The AUC is computed via the trapezoidal rule and equals the
standard ROC AUC.

## Interactive example

Hover the **intersection points** to see the target specificity, the actual specificity
achieved, the corresponding sensitivity, and the **score cutoff** in the data.

<iframe
  src="../auc_chart.html"
  width="100%"
  height="520"
  style="border:none; overflow:hidden;"
  scrolling="no">
</iframe>

```python
import polars as pl
from plotutils.auc import plot_roc_curve

df = pl.DataFrame({
    "score": [...],   # classifier probability / score (higher → more likely positive)
    "label": [...],   # ground-truth binary label (0 = negative, 1 = positive)
})

chart = plot_roc_curve(
    df,
    score_col="score",
    label_col="label",
    specificity_levels=[0.95, 0.90, 0.80],
)

chart.save("roc.html")   # interactive; or .show() in a notebook
```

## Specificity levels

`specificity_levels` annotates the curve at one or more **target specificity values**.
For each target a dashed cross-hair is drawn: a horizontal line from the y-axis to the
curve, then a vertical line down to the x-axis.

The actual specificity shown is always **≥ the requested level** (conservative selection):
the closest curve point at or above the target is chosen.  When multiple points tie,
the one with the highest sensitivity is preferred.

```python
chart = plot_roc_curve(df, specificity_levels=[0.95, 0.90, 0.80])
```

Hovering an intersection point shows:

| Tooltip field      | Meaning                                   |
|--------------------|-------------------------------------------|
| Target specificity | The value you passed in                   |
| Actual specificity | The specificity of the closest curve point |
| Sensitivity        | Corresponding true positive rate          |
| Cutoff             | Score threshold in your data              |

## Computing the curve and AUC directly

The two compute helpers are available separately if you need the raw numbers:

```python
from plotutils.auc import _compute_roc, _compute_auc

roc_df = _compute_roc(df, score_col="score", label_col="label")
# ┌───────────┬─────────────┬─────────────┐
# │ threshold ┆ sensitivity ┆ specificity │
# │ f64       ┆ f64         ┆ f64         │
# ╞═══════════╪═════════════╪═════════════╡
# │ null      ┆ 0.0         ┆ 1.0         │  ← start boundary
# │ 0.97      ┆ 0.04        ┆ 1.0         │
# │ …         ┆ …           ┆ …           │
# │ null      ┆ 1.0         ┆ 0.0         │  ← end boundary
# └───────────┴─────────────┴─────────────┘

auc = _compute_auc(roc_df)   # e.g. 0.823
```

`_compute_roc` is fully vectorised in Polars: scores are grouped by unique value
(ties handled correctly), sorted descending, and cumulative TP / FP counts are derived
with `cum_sum` — no Python loop over thresholds.

## Reference

::: plotutils.auc.plot_roc_curve

::: plotutils.auc._compute_roc

::: plotutils.auc._compute_auc
