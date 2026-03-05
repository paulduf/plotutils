# Prediction errors

Visualize predicted vs. true values for regression models. An identity line (y = x)
is always shown — points on the line are perfect predictions, points above overestimate,
and points below underestimate.

Both axes share the same scale domain (with 2% padding), so the visual distance from
the identity line is geometrically accurate.

## Basic usage

```python
import polars as pl
from plotutils.uncertainty import plot_predictions_errors

df = pl.DataFrame({
    "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8, 9.1, 10.3],
})

chart = plot_predictions_errors(df)
```

<iframe
  src="../predictions_errors_basic.html"
  width="100%"
  height="480"
  style="border:none; overflow:hidden;"
  scrolling="no">
</iframe>

## Color and shape encoding

Use `color_col` and `shape_col` to distinguish groups — for example, different models
or train/test splits:

```python
df = pl.DataFrame({
    "true":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "pred":  [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8],
    "model": ["A", "A", "B", "B", "A", "A", "B", "B"],
    "split": ["train", "test", "train", "test", "train", "test", "train", "test"],
})

chart = plot_predictions_errors(df, color_col="model", shape_col="split")
```

<iframe
  src="../predictions_errors_color.html"
  width="100%"
  height="480"
  style="border:none; overflow:hidden;"
  scrolling="no">
</iframe>

## Multi-panel comparison

Combine with [`hchart` / `vchart`](concat.md) to compare across conditions with
independent axes per panel. This is useful when different splits or datasets
have very different value ranges:

```python
from plotutils.concat import hchart

chart = hchart(
    column="split",
    row="group",
    df=df,
    func=plot_predictions_errors,
    color_col="model",
    shape_col="model",
)
```

See the [Concatenation](concat.md) page for full examples.

## Reference

::: plotutils.uncertainty.plot_predictions_errors
