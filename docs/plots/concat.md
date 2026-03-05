# Concatenation (facet-like with independent axes)

Altair's built-in facet shares all axes across every sub-chart. This is
often undesirable when comparing across splits or conditions that have
very different value ranges. `vchart` and `hchart` produce concatenated
layouts where **each sub-chart computes its own axis domain
independently**, while still appearing as a faceted grid.

Both functions accept two dimensions:

- **Concat dimension** (`column` for `hchart`, `row` for `vchart`) —
  independent axes per panel.
- **Facet dimension** (`row` for `hchart`, `column` for `vchart`) —
  Altair's native facet within each panel, sharing axes.

This gives a grid where one direction has independent axes and the other
has shared axes: exactly what you need when, say, comparing models
across train/test splits that live in different value ranges.

## Example: models across splits and groups

```python
import polars as pl
from plotutils.concat import hchart, vchart
from plotutils.uncertainty import plot_predictions_errors

df = pl.DataFrame({
    "true":  [1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
              50.0, 60.0, 70.0, 75.0, 80.0, 90.0, 50.0, 60.0, 70.0, 75.0, 80.0, 90.0],
    "pred":  [1.1, 2.3, 2.8,  4.2, 5.1, 5.8,   1.4, 1.7, 3.5,  3.6, 5.3, 5.5,
              52.0, 58.0, 72.0, 73.0, 82.0, 91.0, 55.0, 55.0, 75.0, 70.0, 84.0, 93.0],
    "split": ["train"] * 12 + ["test"] * 12,
    "group": (["A"] * 6 + ["B"] * 6) * 2,
    "model": (["linear"] * 3 + ["tree"] * 3) * 4,
})
```

### Horizontal concat + vertical facet

Columns = splits (independent axes), rows = groups (shared axes within
each split), color/shape = models:

```python
chart = hchart(
    column="split",
    row="group",
    df=df,
    func=plot_predictions_errors,
    color_col="model",
    shape_col="model",
)
```

--8<-- "plots/hchart_row_facet.html"

### Vertical concat + horizontal facet

The inverse layout — rows = splits (independent axes), columns = groups
(shared axes):

```python
chart = vchart(
    row="split",
    column="group",
    df=df,
    func=plot_predictions_errors,
    color_col="model",
    shape_col="model",
)
```

--8<-- "plots/vchart_column_facet.html"

### Three splits side by side

Without a facet dimension, each panel simply gets its own axes:

```python
df_split = pl.DataFrame({
    "true":  [1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0,  50.0, 60.0, 70.0, 80.0],
    "pred":  [1.2, 1.8, 3.2, 3.9,  11.0, 19.0, 31.0, 39.0,  52.0, 58.0, 72.0, 78.0],
    "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
})

chart = hchart(
    column="split",
    df=df_split,
    func=plot_predictions_errors,
    width=300,
    height=300,
)
```

--8<-- "plots/hchart_three_splits.html"

## Reference

::: plotutils.concat
