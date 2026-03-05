# Error bars

Visualize repeated measurements across categories with automatic aggregation and configurable error bars.

Each x category can contain multiple y values. The chart shows the mean as a point and the spread as error bars, computed by Altair's native `mark_errorbar`.

## Example

```python
import polars as pl
from plotutils.uncertainty import plot_confidence_scatter

df = pl.DataFrame({
    "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
    "value": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
              2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
              4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
})

chart = plot_confidence_scatter(
    df,
    x_col="category",
    y_col="value",
    extent="stdev",
)
```

--8<-- "plots/confidence_scatter.html"

The `extent` parameter controls the error bar type:

| `extent` | Description |
|----------|-------------|
| `"ci"` (default) | Bootstrap 95% confidence interval |
| `"stdev"` | ±1 standard deviation |
| `"stderr"` | Standard error of the mean |
| `"iqr"` | Interquartile range (25th–75th percentile) |

!!! note
    The default `extent="ci"` uses bootstrap resampling, which is non-deterministic.
    Use `extent="stdev"` or `extent="stderr"` for reproducible output.

## Numeric x-axis with custom labels

When x values are numeric (e.g., model capacity, regularization strength),
pass `x_labels` to display readable labels while keeping a quantitative axis
with proper spacing:

```python
df = pl.DataFrame({
    "x": [1.0] * 10 + [2.0] * 10 + [3.0] * 10,
    "y": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
          2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
          4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
})

chart = plot_confidence_scatter(
    df,
    x_labels={1.0: "Low", 2.0: "Medium", 3.0: "High"},
    extent="stdev",
    scale_type="log",  # optional log scale on x
)
```

## Reference

::: plotutils.uncertainty.plot_confidence_scatter
