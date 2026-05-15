# Forest Plot

A forest plot shows per-item point estimates and confidence intervals as
horizontal bars, one row per item (subgroup, variable, study, …).  The
x-axis uses a **log scale** with natural-scale tick labels, making the chart
suited for hazard ratios, odds ratios, and risk ratios.

A solid vertical rule marks the null effect (`null_value`, default `1.0`).

## Example

```python
import polars as pl
from plotutils.forest import plot_forest

df = pl.DataFrame({
    "subgroup": ["Overall", "Age < 65", "Age ≥ 65", "Male", "Female"],
    "hr":  [0.78, 0.72, 0.85, 0.80, 0.75],
    "low": [0.62, 0.55, 0.65, 0.61, 0.57],
    "high":[0.98, 0.94, 1.11, 1.05, 0.99],
})

chart = plot_forest(
    df,
    center_col="hr",
    low_col="low",
    high_col="high",
    label_col="subgroup",
    x_title="Hazard Ratio",
)
```

--8<-- "plots/forest_basic.html"

## With minimum-effect boundaries and colour grouping

Pass `min_effect` to draw symmetric dashed rules at
`null_value / min_effect` and `null_value * min_effect` — the boundary of a
minimum clinically meaningful effect.  `color_col` maps a categorical column
to colour.

```python
chart = plot_forest(
    df,
    center_col="hr",
    low_col="low",
    high_col="high",
    label_col="subgroup",
    min_effect=1.25,      # draws lines at 0.80 and 1.25
    color_col="population",
    title="Treatment effect by subgroup",
)
```

--8<-- "plots/forest_effect.html"

## Grouped subgroups

Pass `group_col` to display multiple rows per label alongside one another
(for example, two treatment arms compared at every subgroup).  An
`alt.YOffset` encoding dodges the CI bars and center points vertically
within each y-band so the groups do not overlap, and — when `color_col`
is not set — marks are auto-coloured by `group_col` to keep the groups
visually distinguishable.

```python
df = pl.DataFrame({
    "subgroup": ["Overall", "Overall", "Age < 65", "Age < 65",
                 "Age ≥ 65", "Age ≥ 65", "Male", "Male",
                 "Female", "Female"],
    "treatment": ["A", "B"] * 5,
    "hr":  [0.78, 0.92, 0.72, 0.84, 0.85, 1.02, 0.80, 0.95, 0.75, 0.88],
    "low": [0.62, 0.74, 0.55, 0.65, 0.65, 0.80, 0.61, 0.74, 0.57, 0.69],
    "high":[0.98, 1.16, 0.94, 1.10, 1.11, 1.30, 1.05, 1.22, 0.99, 1.13],
})

chart = plot_forest(
    df,
    center_col="hr",
    low_col="low",
    high_col="high",
    label_col="subgroup",
    group_col="treatment",
    min_effect=1.25,
    x_title="Hazard Ratio",
    title="Treatment A vs B by subgroup",
)
```

--8<-- "plots/forest_grouped.html"

## Reference

::: plotutils.forest.plot_forest
