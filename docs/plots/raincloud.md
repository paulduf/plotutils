# Raincloud Plot

A raincloud plot combines three complementary views of distributional data:
a **half-violin** (KDE density), a **boxplot**, and a **jittered strip** of
individual data points.

## Example

```python
import polars as pl
from plotutils.raincloud import plot_raincloud

df = pl.DataFrame({
    "group": ["A"] * 20 + ["B"] * 20,
    "value": [1.0 + i * 0.05 for i in range(20)]
           + [2.0 + i * 0.05 for i in range(20)],
})

chart = plot_raincloud(df, x_col="group", y_col="value")
```

--8<-- "plots/raincloud.html"

## Reference

::: plotutils.raincloud.plot_raincloud
