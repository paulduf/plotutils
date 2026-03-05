# Altair Plot Utils

A collection of ready-made, interactive chart functions built on
[Altair](https://altair-viz.github.io/) and
[Polars](https://pola.rs/).

## Philosophy

Most plotting libraries force you to choose between **quick-and-dirty**
(matplotlib one-liners that look bad) and **publication-ready** (hours of
tweaking). This package sits in between:

- **One function call** produces a complete, styled chart with sensible
  defaults (grid, axis labels, tooltips).
- **Polars-native** — pass a `pl.DataFrame` directly, no pandas
  conversion needed.
- **Interactive by default** — every chart supports hover tooltips,
  highlighting, or linked selection out of the box.
- **Composable** — combine any chart with `hchart` / `vchart` to get
  faceted layouts with independent axes.

## Available plots

| Module | Functions | Description |
|--------|-----------|-------------|
| `plotutils.hist` | `plot_grouped_histogram` | Side-by-side grouped histograms |
| `plotutils.uncertainty` | `plot_confidence_scatter`, `plot_deviations`, `plot_predictions_errors` | Error bars, deviation plots, predicted-vs-true scatter |
| `plotutils.boxplot` | `plot_bivariate_boxes`, `plot_bivariate_strip` | Doubly-grouped boxplots and jittered strip plots |
| `plotutils.parallel` | `plot_parallel_coordinates` | Parallel coordinates with normalization and highlighting |
| `plotutils.auc` | `plot_roc_curve`, `AUCReport` | ROC curves with specificity annotations; linked multi-variable AUC explorer |
| `plotutils.concat` | `hchart`, `vchart` | Facet-like layouts with independent axes per panel |

## Quick start

```python
import polars as pl
from plotutils.uncertainty import plot_confidence_scatter

df = pl.DataFrame({
    "category": ["A"] * 20 + ["B"] * 20,
    "value": [1.0 + i * 0.1 for i in range(20)]
           + [2.0 + i * 0.1 for i in range(20)],
})

chart = plot_confidence_scatter(df, x_col="category", y_col="value")
chart.save("chart.html")   # interactive HTML
chart.show()               # Jupyter / notebook
```

## Installation

```bash
uv pip install -e .
```

## Building the docs

```bash
just docs-img     # generate interactive chart HTML files
just docs-build   # build the site (runs docs-img first)
just docs-serve   # live-reload dev server
```
