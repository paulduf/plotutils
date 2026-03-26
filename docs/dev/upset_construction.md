# UpSet Plot Construction

This page walks through how `plot_upset` transforms a binary membership
matrix into the final three-panel chart.  It is intended for developers
who want to understand (or modify) the internals.

## Input format

A Polars DataFrame where each column is a set and each row is an element.
Values are `0` / `1` (or `bool`).

```
┌───────┬────────┬────────┬────────┐
│ Drama ┆ Comedy ┆ Action ┆ Sci-Fi │
│  i64  ┆   i64  ┆   i64  ┆   i64  │
╞═══════╪════════╪════════╪════════╡
│   1   ┆    0   ┆    0   ┆    0   │
│   1   ┆    1   ┆    0   ┆    0   │
│   1   ┆    0   ┆    1   ┆    0   │
│   0   ┆    1   ┆    1   ┆    0   │
│   0   ┆    0   ┆    0   ┆    1   │
│   1   ┆    1   ┆    1   ┆    0   │
│   1   ┆    0   ┆    0   ┆    1   │
│   0   ┆    0   ┆    1   ┆    1   │
└───────┴────────┴────────┴────────┘
```

## Step 1 — `_preprocess_upset()` overview

The internal helper `_preprocess_upset()` takes the raw DataFrame and
produces four derived tables.  The steps are described below, with
example intermediate results based on the data above.

### 1a. Intersection counts (`group_by` + `len`)

Group rows by their full combination of set columns, count occurrences,
and compute the **degree** (number of active sets) for each intersection.

```python
intersection_df = (
    df.group_by(set_cols, maintain_order=True)
      .agg(pl.len().alias("cardinality"))
      .with_columns(pl.sum_horizontal(*set_cols).alias("degree"))
)
```

Result:

```
┌───────┬────────┬────────┬────────┬─────────────┬────────┐
│ Drama ┆ Comedy ┆ Action ┆ Sci-Fi ┆ cardinality ┆ degree │
╞═══════╪════════╪════════╪════════╪═════════════╪════════╡
│   1   ┆    0   ┆    0   ┆    0   │      1      │    1   │
│   1   ┆    1   ┆    0   ┆    0   │      1      │    2   │
│   1   ┆    0   ┆    1   ┆    0   │      1      │    2   │
│   0   ┆    1   ┆    1   ┆    0   │      1      │    2   │
│   0   ┆    0   ┆    0   ┆    1   │      1      │    1   │
│   1   ┆    1   ┆    1   ┆    0   │      1      │    3   │
│   1   ┆    0   ┆    0   ┆    1   │      1      │    2   │
│   0   ┆    0   ┆    1   ┆    1   │      1      │    2   │
└───────┴────────┴────────┴────────┴─────────────┴────────┘
```

### 1b. Filter

Optional filters are applied in this order:

1. `min_degree` — remove intersections with fewer participating sets
2. `max_degree` — remove intersections with more participating sets

Example with `min_degree=2`:

```
┌───────┬────────┬────────┬────────┬─────────────┬────────┐
│ Drama ┆ Comedy ┆ Action ┆ Sci-Fi ┆ cardinality ┆ degree │
╞═══════╪════════╪════════╪════════╪═════════════╪════════╡
│   1   ┆    1   ┆    0   ┆    0   │      1      │    2   │
│   1   ┆    0   ┆    1   ┆    0   │      1      │    2   │
│   0   ┆    1   ┆    1   ┆    0   │      1      │    2   │
│   1   ┆    1   ┆    1   ┆    0   │      1      │    3   │
│   1   ┆    0   ┆    0   ┆    1   │      1      │    2   │
│   0   ┆    0   ┆    1   ┆    1   │      1      │    2   │
└───────┴────────┴────────┴────────┴─────────────┴────────┘
```

### 1c. Sort

Sort by `cardinality` (frequency) or `degree`, with set column values as
a deterministic tiebreaker.  Default: frequency descending.

### 1d. Assign stable identifiers

After sorting, two columns are added:

- `_intersection_id` — concatenation of set values (e.g. `"1-1-0-0"`)
- `_order` — row index (0, 1, 2, ...) encoding the visual x-axis position

```
┌─────────────────┬────────┬─────────────┬────────┐
│ _intersection_id ┆ _order ┆ cardinality ┆ degree │
╞═════════════════╪════════╪═════════════╪════════╡
│ 1-1-0-0         ┆      0 ┆           1 ┆      2 │
│ 1-0-1-0         ┆      1 ┆           1 ┆      2 │
│ ...             ┆    ... ┆         ... ┆    ... │
└─────────────────┴────────┴─────────────┴────────┘
```

The x-axis in every chart component uses
`sort=EncodingSortField(field="_order", order="ascending")`,
ensuring all panels share the same column ordering.

### 1e. Set sizes and ordering

Set sizes are computed from the original DataFrame (`df[col].sum()` per set
column) and sorted **descending** so the largest set appears at the top of
the matrix y-axis.

```
┌──────────┬──────────┬────────┐
│ set_name ┆ set_size ┆ _y_pos │
╞══════════╪══════════╪════════╡
│ Drama    ┆        5 ┆      0 │
│ Action   ┆        4 ┆      1 │
│ Comedy   ┆        3 ┆      2 │
│ Sci-Fi   ┆        3 ┆      3 │
└──────────┴──────────┴────────┘
```

The `_y_pos` column (integer 0..N-1) is the quantitative y-axis position
used by all matrix and set-size chart layers.

### 1f. Matrix (long form via `unpivot`)

The intersection table is unpivoted so each (intersection, set) pair
becomes a row.  A `_member` column indicates whether the set is active
in that intersection.

```python
matrix_df = intersection_df.unpivot(
    on=set_cols,
    index=["_intersection_id", "_order", "cardinality", "degree"],
    variable_name="_set_name",
    value_name="_member",
).with_columns(
    pl.col("_set_name").replace_strict(set_to_pos).alias("_y_pos")
)
```

Excerpt:

```
┌─────────────────┬────────┬───────────┬─────────┬────────┐
│ _intersection_id ┆ _order ┆ _set_name ┆ _member ┆ _y_pos │
╞═════════════════╪════════╪═══════════╪═════════╪════════╡
│ 1-1-0-0         ┆      0 ┆ Drama     ┆       1 ┆      0 │
│ 1-1-0-0         ┆      0 ┆ Comedy    ┆       1 ┆      2 │
│ 1-1-0-0         ┆      0 ┆ Action    ┆       0 ┆      1 │
│ 1-1-0-0         ┆      0 ┆ Sci-Fi    ┆       0 ┆      3 │
│ ...             ┆    ... ┆ ...       ┆     ... ┆    ... │
└─────────────────┴────────┴───────────┴─────────┴────────┘
```

### 1g. Connecting lines

For each intersection with degree >= 2, compute the min and max `_y_pos`
of active dots.  These become the endpoints of vertical `mark_rule` lines.

```
┌─────────────────┬────────┬────────┬────────┐
│ _intersection_id ┆ _order ┆ _y_min ┆ _y_max │
╞═════════════════╪════════╪════════╪════════╡
│ 1-1-0-0         ┆      0 ┆      0 ┆      2 │
│ 1-0-1-0         ┆      1 ┆      0 ┆      1 │
│ ...             ┆    ... ┆    ... ┆    ... │
└─────────────────┴────────┴────────┴────────┘
```

---

## Step 2 — Chart assembly

The four DataFrames feed into three Altair sub-charts that are composed
via `vconcat` (vertical) and `hconcat` (horizontal).

### 2a. Cardinality bars (top)

`mark_bar` with:

- `x = _intersection_id:N` sorted by `_order`
- `y = cardinality:Q`
- `axis=None` on x (the matrix below serves as the categorical axis)
- Hover highlight via a shared `selection_point`

### 2b. Intersection matrix (bottom)

Three layers sharing the same quantitative y-axis (`_y_pos:Q`) with a
custom `labelExpr` that maps integers to set names:

| Layer | Mark | Data | Purpose |
|---|---|---|---|
| Background dots | `mark_circle(color="#e0e0e0")` | Full `matrix_df` | Shows the grid of all positions |
| Connecting lines | `mark_rule(strokeWidth=2)` | `lines_df` | Vertical lines between active dots |
| Active dots | `mark_circle` | `matrix_df` where `_member == 1` | Filled dots showing set membership |

**Why quantitative y instead of nominal?**
Vega-Lite's `mark_rule` requires `y:Q` / `y2:Q` to draw between two
endpoints.  Nominal y-axes do not support `y2`.  The `labelExpr` trick
maps integer positions back to set names on the axis.

### 2c. Set-size bars (left, optional)

`mark_bar(orient="horizontal")` with:

- `x = set_size:Q` (reversed scale so bars grow leftward)
- `y = _y_pos:Q` (same scale/axis as the matrix)
- Fixed bar thickness via `size` parameter

### 2d. Final composition

```
┌─────────────┬──────────────────────────────────────┐
│  (spacer)   │       Cardinality bars               │
├─────────────┼──────────────────────────────────────┤
│  Set size   │       Intersection matrix             │
│    bars     │  (bg dots + lines + active dots)      │
└─────────────┴──────────────────────────────────────┘
```

```python
main = alt.vconcat(bar_chart, matrix_chart, spacing=0)
       .resolve_scale(x="shared")

left = alt.vconcat(spacer, set_size_chart, spacing=0)

chart = alt.hconcat(left, main, spacing=5)
```

The `resolve_scale(x="shared")` on the main column ensures the bar chart
columns align perfectly with the matrix columns below.

The spacer is an invisible chart that occupies the space above the
set-size bars, keeping them vertically aligned with the matrix.

---

## Key implementation choices

| Decision | Rationale |
|---|---|
| Preprocessing in Polars, not Vega-Lite transforms | Explicit, testable, debuggable; avoids chaining brittle Vega transforms |
| Quantitative y with `labelExpr` | Only way to get `mark_rule` with `y`/`y2` endpoints for connecting lines |
| Sets sorted by size descending | Convention from the UpSet paper: puts the most important sets first |
| `_order` row index for x-sorting | Decouples visual position from data content; stable across datasets |
| `maintain_order=True` on `group_by` | Ensures deterministic output across Polars process runs |
| Single `selection_point` added once per sub-chart | Avoids Altair's "deduplicated selection parameter" warning |
