# UpSet Plot

UpSet plots visualize **set intersections** as a scalable alternative to Venn
diagrams.  They work well for **3–30 sets** by replacing overlapping shapes
with a matrix layout.

An UpSet plot has three components:

| Component | Position | Shows |
|---|---|---|
| **Intersection bars** | top | size (cardinality) of each intersection |
| **Intersection matrix** | bottom | which sets participate (dots + connecting lines) |
| **Set size bars** | left | total membership of each individual set |

> **Tip** — always include the set-size bars (the default) so readers can
> judge intersection sizes relative to their parent sets.

## Basic example

```python
import polars as pl
from plotutils.upset import plot_upset

df = pl.DataFrame(
    {
        "Drama": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        "Comedy": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        "Action": [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "Sci-Fi": [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    }
)
chart = plot_upset(df, title="Movie genre overlaps")
```

--8<-- "plots/upset_basic.html"

## Filtering and sorting

You can filter intersections by degree (number of participating sets) and
limit the total number shown.  Sorting can be by `"frequency"` (default) or
`"degree"`, and you can pass a list for multi-level sorting.

```python
chart = plot_upset(
    df,
    sort_by=["degree", "frequency"],
    min_degree=1,
    n_intersections=10,
    title="Degree ≥ 1, sorted by degree then frequency",
)
```

--8<-- "plots/upset_filtered.html"

## Without set sizes

Set `show_set_sizes=False` to hide the left-side bar chart and show only the
intersection bars and matrix.

```python
chart = plot_upset(df, show_set_sizes=False, title="No set-size bars")
```

--8<-- "plots/upset_no_sizes.html"

## When to use UpSet plots

Beyond the classic "which sets overlap?" question, UpSet plots are useful
in many practical scenarios:

- **Missing-data patterns** — Encode each column as a set ("has missing
  value in column X").  The intersection bars immediately reveal which
  combinations of missing columns are most common, helping decide
  imputation strategies.
- **Constraint / rule satisfaction** — Each set represents a rule or
  constraint that a record satisfies.  The plot shows how many records
  satisfy which combinations, making it easy to spot rules that rarely
  co-occur or always fire together.
- **Feature co-occurrence** — In NLP or product analytics, each set is a
  feature (tag, keyword, flag) present on an item.  The plot highlights
  which feature bundles dominate.
- **Multi-label classification** — Each set is a predicted or true label.
  The plot shows how labels overlap across samples, exposing common
  multi-label patterns.
- **Survey / questionnaire analysis** — Each set is a response option
  selected by respondents (e.g. "Which tools do you use?").  The plot
  shows which tool combinations are most popular.
- **Genomics / pathway membership** — Genes can belong to multiple
  biological pathways.  The plot reveals which pathway overlaps contain
  the most genes.
- **Error / alert co-occurrence** — In monitoring systems, each set is an
  alert type.  The plot shows which alert combinations fire together,
  helping diagnose correlated failures.

## Reference

::: plotutils.upset.plot_upset
