# Deviations

Visualize how individual measurements deviate from their per-group mean. This removes the "level" effect and focuses purely on scatter and variability, making it easy to compare consistency across conditions that have very different absolute values.

## Basic usage

```python
import polars as pl
from plotutils.uncertainty import plot_deviations

df = pl.DataFrame({
    "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
    "value": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
              2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
              4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
})

chart = plot_deviations(df, x_col="category", y_col="value")
```

<iframe
  src="../deviations_basic.html"
  width="100%"
  height="480"
  style="border:none; overflow:hidden;"
  scrolling="no">
</iframe>

Each point shows `y - mean(y)` for its group. The horizontal line at zero is the group mean reference.

## Relative deviations

Use `relative=True` to express deviations as a fraction of the group mean —
`(y - mean) / mean`. This is useful when comparing groups with very different
magnitudes, since the y-axis becomes dimensionless:

```python
chart = plot_deviations(df, x_col="category", y_col="value", relative=True)
```

## Tolerance bands

Add symmetric reference lines with `add_levels` to mark acceptable deviation
thresholds. For example, `add_levels=[0.1, 0.2]` draws lines at ±0.1 and ±0.2:

```python
df = pl.DataFrame({
    "x": ["A"] * 10 + ["B"] * 10,
    "y": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.05, 0.95,
          2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.1, 1.9, 2.05, 1.95],
})

chart = plot_deviations(
    df,
    x_col="x",
    y_col="y",
    add_levels=[0.1, 0.2],
)
```

<iframe
  src="../deviations_levels.html"
  width="100%"
  height="480"
  style="border:none; overflow:hidden;"
  scrolling="no">
</iframe>

## Reference

::: plotutils.uncertainty.plot_deviations
