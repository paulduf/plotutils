# Confidence scatter plot with error bars

Create scatter plots with error bars. Aggregates raw data automatically using Altair's native `mark_errorbar`.

## Example

```python
import polars as pl
from plotutils.errorbar import plot_confidence_scatter

df = pl.DataFrame({
    "x": [1.0] * 10 + [2.0] * 10 + [3.0] * 10,
    "y": [1.0, 0.8, 1.2, ...],  # multiple samples per x
})

chart = plot_confidence_scatter(
    df,
    x_labels={1.0: "Low", 2.0: "Medium", 3.0: "High"},
)
```

## Reference

::: plotutils.errorbar.plot_confidence_scatter
