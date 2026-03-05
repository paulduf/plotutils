# Parallel Coordinates

Visualize multivariate data by drawing one line per observation across
parallel vertical axes.  Hover a line to highlight it and see all values
in the tooltip.

## Basic usage

```python
import polars as pl
from plotutils.parallel import plot_parallel_coordinates

df = pl.DataFrame({
    "sepal_length": [5.1, 4.9, 7.0, 6.3, 6.5, 5.8],
    "sepal_width":  [3.5, 3.0, 3.2, 3.3, 2.8, 2.7],
    "petal_length": [1.4, 1.4, 4.7, 4.4, 4.6, 5.1],
    "petal_width":  [0.2, 0.2, 1.4, 1.3, 1.5, 1.9],
    "species": ["setosa", "setosa", "versicolor",
                "versicolor", "virginica", "virginica"],
})

chart = plot_parallel_coordinates(
    df,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    color_col="species",
)
```

--8<-- "plots/parallel_basic.html"

## Normalized

When columns have very different scales, use `normalize=True` to apply
min-max normalization so all axes share the 0–1 range:

```python
chart = plot_parallel_coordinates(
    df,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    color_col="species",
    normalize=True,
)
```

--8<-- "plots/parallel_normalized.html"

## Log transforms

When normalizing, you can apply a per-column transform *before*
normalization.  Use `transforms=["log", "linear", ...]` to log-transform
specific columns (useful for skewed distributions):

```python
chart = plot_parallel_coordinates(
    df,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    color_col="species",
    normalize=True,
    transforms=["linear", "linear", "log", "log"],
)
```

## Sample identifiers

Pass `id_col` to include a sample identifier in the tooltip:

```python
df_with_id = df.with_columns(pl.Series("id", ["s1", "s2", "s3", "s4", "s5", "s6"]))

chart = plot_parallel_coordinates(
    df_with_id,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    color_col="species",
    id_col="id",
)
```

## Reference

::: plotutils.parallel.plot_parallel_coordinates
