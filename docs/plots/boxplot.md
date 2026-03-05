# Boxplots & Strip Plots

Doubly-grouped distribution plots for one continuous variable against two binary outcomes.
Two chart variants are provided:

- **Boxplot** (`plot_bivariate_boxes`) — side-by-side boxplots for each (label_x, label_y) combination.
- **Strip plot** (`plot_bivariate_strip`) — jittered strip plot showing every individual sample, coloured by label_y.

## Example data

```python
import polars as pl

df = pl.DataFrame({
    "score": [0.8, 0.6, 0.9, 0.7, 0.5, 0.3, 0.4, 0.2, 0.85, 0.65, 0.35, 0.25],
    "outcome_a": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    "outcome_b": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "patient_id": [f"P{i:03d}" for i in range(12)],
})
```

## Boxplot

```python
from plotutils.boxplot import plot_bivariate_boxes

chart = plot_bivariate_boxes(
    df,
    score_col="score",
    label_x_col="outcome_a",
    label_y_col="outcome_b",
)
```

--8<-- "plots/boxplot_boxes.html"

## Strip plot

```python
from plotutils.boxplot import plot_bivariate_strip

chart = plot_bivariate_strip(
    df,
    score_col="score",
    label_x_col="outcome_a",
    label_y_col="outcome_b",
    id_col="patient_id",
)
```

--8<-- "plots/boxplot_strip.html"

## Missing scores

When some patients have missing scores, pass `missing_score_df` to show them
as cross marks below the main chart area:

```python
missing_df = pl.DataFrame({
    "outcome_a": [1, 0, 1],
    "outcome_b": [0, 1, 0],
    "patient_id": ["M001", "M002", "M003"],
})

chart = plot_bivariate_boxes(
    df,
    score_col="score",
    label_x_col="outcome_a",
    label_y_col="outcome_b",
    id_col="patient_id",
    missing_score_df=missing_df,
)
```

## Reference

::: plotutils.boxplot.plot_bivariate_boxes

::: plotutils.boxplot.plot_bivariate_strip
