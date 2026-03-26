"""UpSet plots for visualizing set intersections.

UpSet plots are a scalable alternative to Venn diagrams, effective for
3–30 sets.  They combine a dot-matrix showing which sets participate in
each intersection with bar charts for intersection sizes and (optionally)
individual set sizes.

Reference
---------
Lex et al., "UpSet: Visualization of Intersecting Sets",
IEEE Trans. Vis. Comput. Graph., 20(12), 1983–1992, 2014.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import altair as alt
import polars as pl


class _UpsetData(NamedTuple):
    intersection_df: pl.DataFrame
    matrix_df: pl.DataFrame
    lines_df: pl.DataFrame
    set_sizes_df: pl.DataFrame


_SORT_KEY = Literal["frequency", "degree"]
_SORT_DIR = Literal["ascending", "descending"]

# Default sort direction per key when sort_order is not given per key.
_DEFAULT_DIR: dict[str, _SORT_DIR] = {
    "frequency": "descending",
    "degree": "ascending",
}


def _preprocess_upset(
    df: pl.DataFrame,
    set_cols: list[str],
    *,
    sort_by: _SORT_KEY | list[_SORT_KEY] = "frequency",
    sort_order: _SORT_DIR | list[_SORT_DIR] | None = None,
    min_degree: int = 0,
    max_degree: int | None = None,
    n_intersections: int | None = None,
) -> _UpsetData:
    """Compute intersection counts, matrix, connecting-line endpoints and set sizes.

    Parameters
    ----------
    df : pl.DataFrame
        Binary membership matrix (one column per set, values 0/1).
    set_cols : list[str]
        Column names to treat as sets.
    sort_by, sort_order, min_degree, max_degree, n_intersections
        Filtering / sorting options forwarded from :func:`plot_upset`.

    Returns
    -------
    _UpsetData
        Named tuple of ``(intersection_df, matrix_df, lines_df, set_sizes_df)``.
    """
    for col in set_cols:
        if df[col].dtype not in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Boolean,
        ):
            msg = f"Column '{col}' must be integer or boolean, got {df[col].dtype}"
            raise TypeError(msg)

    # Cast booleans to int so sum_horizontal works uniformly.
    df_sets = df.select(
        pl.col(c).cast(pl.Int64) if df[c].dtype == pl.Boolean else pl.col(c)
        for c in set_cols
    )

    # --- intersection counts ---
    intersection_df = (
        df_sets
        .group_by(set_cols, maintain_order=True)
        .agg(pl.len().alias("cardinality"))
        .with_columns(pl.sum_horizontal(*set_cols).alias("degree"))
    )

    # --- filter ---
    intersection_df = intersection_df.filter(pl.col("degree") >= min_degree)
    if max_degree is not None:
        intersection_df = intersection_df.filter(pl.col("degree") <= max_degree)

    # --- sort (set_cols as tiebreaker for deterministic output) ---
    keys = [sort_by] if isinstance(sort_by, str) else list(sort_by)

    # Resolve per-key sort directions.
    if sort_order is None:
        dirs = [_DEFAULT_DIR[k] for k in keys]
    elif isinstance(sort_order, str):
        dirs = [sort_order] * len(keys)
    else:
        dirs = list(sort_order)

    # Map "frequency" → the actual column name "cardinality".
    col_names = ["cardinality" if k == "frequency" else k for k in keys]

    # Deduplicate while preserving order, then append set_cols as tiebreaker.
    seen: set[str] = set()
    sort_cols: list[str] = []
    desc_flags: list[bool] = []
    for c, d in zip(col_names, dirs):
        if c not in seen:
            seen.add(c)
            sort_cols.append(c)
            desc_flags.append(d == "descending")

    intersection_df = intersection_df.sort(
        [*sort_cols, *set_cols],
        descending=[*desc_flags, *([False] * len(set_cols))],
    )

    # --- limit ---
    if n_intersections is not None:
        intersection_df = intersection_df.head(n_intersections)

    # --- stable identifiers ---
    intersection_df = intersection_df.with_columns(
        pl.concat_str(set_cols, separator="-").alias("_intersection_id"),
    ).with_row_index("_order")

    # --- set sizes (sorted descending so largest set is at the top) ---
    set_sizes = [(col, int(df_sets[col].sum())) for col in set_cols]
    set_sizes.sort(key=lambda t: t[1], reverse=True)
    ordered_set_names = [name for name, _ in set_sizes]
    set_to_pos = {name: i for i, name in enumerate(ordered_set_names)}

    set_sizes_df = pl.DataFrame({
        "set_name": ordered_set_names,
        "set_size": [size for _, size in set_sizes],
        "_y_pos": list(range(len(set_cols))),
    })

    # --- intersection set labels (for tooltips) ---
    set_size_map = dict(zip(ordered_set_names, [s for _, s in set_sizes]))
    # Build a human-readable label like "Drama & Comedy" for each intersection.
    _labels: list[str] = []
    _pct_labels: list[str] = []
    for row in intersection_df.iter_rows(named=True):
        active = [c for c in set_cols if row[c] == 1]
        _labels.append(" & ".join(active) if active else "(none)")
        parts = [
            f"{row['cardinality'] / set_size_map[c] * 100:.0f}% of {c}"
            for c in active
        ]
        _pct_labels.append(", ".join(parts) if parts else "")
    intersection_df = intersection_df.with_columns(
        pl.Series("_sets_label", _labels),
        pl.Series("_pct_label", _pct_labels),
    )

    # --- matrix (long form) ---
    matrix_df = (
        intersection_df
        .unpivot(
            on=set_cols,
            index=[
                "_intersection_id", "_order", "cardinality", "degree",
                "_sets_label", "_pct_label",
            ],
            variable_name="_set_name",
            value_name="_member",
        )
        .with_columns(
            pl.col("_set_name")
            .replace_strict(set_to_pos)
            .cast(pl.Int64)
            .alias("_y_pos"),
        )
    )

    # --- connecting lines (y_min / y_max of active dots per intersection) ---
    lines_df = (
        matrix_df
        .filter(pl.col("_member") == 1)
        .group_by(
            "_intersection_id", "_order", "cardinality", "degree",
            "_sets_label", "_pct_label",
        )
        .agg(
            pl.col("_y_pos").min().alias("_y_min"),
            pl.col("_y_pos").max().alias("_y_max"),
        )
        .filter(pl.col("_y_min") != pl.col("_y_max"))
    )

    return _UpsetData(intersection_df, matrix_df, lines_df, set_sizes_df)


def plot_upset(
    df: pl.DataFrame,
    set_cols: list[str] | None = None,
    *,
    sort_by: _SORT_KEY | list[_SORT_KEY] = "frequency",
    sort_order: _SORT_DIR | list[_SORT_DIR] | None = None,
    min_degree: int = 0,
    max_degree: int | None = None,
    n_intersections: int | None = None,
    show_set_sizes: bool = True,
    title: str = "",
    width: int = 600,
    height: int = 300,
    bar_color: str = "#4c78a8",
    dot_color: str = "#333333",
    line_color: str = "#333333",
) -> alt.VConcatChart | alt.HConcatChart:
    """Create an UpSet plot for visualizing set intersections.

    Parameters
    ----------
    df : pl.DataFrame
        Binary membership matrix.  Each row is an element; each column
        listed in *set_cols* contains ``0`` / ``1`` (or ``bool``) indicating
        membership.
    set_cols : list[str] | None
        Columns to treat as sets.  When ``None`` every column in *df* is
        used.
    sort_by : str or list[str]
        Sort key(s).  Accepted values are ``"frequency"`` (intersection
        size) and ``"degree"`` (number of participating sets).  A list
        applies multi-level sorting, e.g. ``["degree", "frequency"]``
        sorts primarily by degree, then by size within each degree.
    sort_order : str, list[str], or None
        Sort direction(s).  A single string applies to all keys; a list
        sets the direction per key.  When ``None`` (default), each key
        uses its natural default: ``"descending"`` for frequency (biggest
        first) and ``"ascending"`` for degree (simplest first).
    min_degree : int
        Hide intersections whose degree is below this threshold.
    max_degree : int | None
        Hide intersections whose degree is above this threshold.
    n_intersections : int | None
        Show only the first *n* intersections (after sorting).
    show_set_sizes : bool
        If ``True`` a horizontal bar chart of individual set sizes is
        displayed to the left of the matrix.
    title : str
        Chart title.
    width : int
        Width of the cardinality bar chart (and matrix) in pixels.
    height : int
        Height of the cardinality bar chart in pixels.
    bar_color : str
        Fill colour for the cardinality and set-size bars.
    dot_color : str
        Fill colour for the active dots in the intersection matrix.
    line_color : str
        Stroke colour for the connecting lines in the matrix.

    Returns
    -------
    alt.VConcatChart | alt.HConcatChart
        Composited Altair chart.
    """
    alt.data_transformers.disable_max_rows()

    if set_cols is None:
        set_cols = df.columns

    data = _preprocess_upset(
        df, set_cols,
        sort_by=sort_by, sort_order=sort_order,
        min_degree=min_degree, max_degree=max_degree,
        n_intersections=n_intersections,
    )

    # -- shared hover selection --
    highlight = alt.selection_point(
        name="upset_highlight",
        on="pointerover",
        fields=["_intersection_id"],
        empty=False,
    )

    # -- shared encodings --
    # Explicit domain list guarantees sort order across vconcat sub-charts
    # (EncodingSortField can be unreliable when resolve_scale merges scales).
    x_domain = data.intersection_df.sort("_order")["_intersection_id"].to_list()
    n_sets = len(set_cols)
    ordered_set_names = data.set_sizes_df["set_name"].to_list()
    y_label_expr = (
        "{"
        + ", ".join(f"{i}: '{name}'" for i, name in enumerate(ordered_set_names))
        + "}[datum.value]"
    )
    matrix_height = max(n_sets * 30, 100)

    # ── cardinality bar chart (top) ───────────────────────────────────
    bar_chart = (
        alt.Chart(data.intersection_df)
        .mark_bar()
        .encode(
            x=alt.X("_intersection_id:N", sort=x_domain, axis=None),
            y=alt.Y("cardinality:Q", title="Intersection Size"),
            color=alt.condition(
                highlight, alt.value(bar_color), alt.value("#ddd"),
            ),
            tooltip=[
                alt.Tooltip("_sets_label:N", title="Sets"),
                alt.Tooltip("cardinality:Q", title="Size"),
                alt.Tooltip("degree:Q", title="Degree"),
                alt.Tooltip("_pct_label:N", title="% of set"),
            ],
        )
        .properties(width=width, height=height)
    )

    # ── intersection matrix (bottom) ──────────────────────────────────
    y_axis = alt.Axis(
        values=list(range(n_sets)),
        tickCount=n_sets,
        labelExpr=y_label_expr,
        title=None,
        grid=False,
    )
    y_scale = alt.Scale(domain=[-0.5, n_sets - 0.5])
    y_enc = alt.Y("_y_pos:Q", scale=y_scale, axis=y_axis)

    bg_dots = (
        alt.Chart(data.matrix_df)
        .mark_circle(size=80, color="#e0e0e0")
        .encode(
            x=alt.X("_intersection_id:N", sort=x_domain, axis=None),
            y=y_enc,
        )
    )

    active_dots = (
        alt.Chart(data.matrix_df.filter(pl.col("_member") == 1))
        .mark_circle(size=80)
        .encode(
            x=alt.X("_intersection_id:N", sort=x_domain, axis=None),
            y=y_enc,
            color=alt.condition(
                highlight, alt.value(dot_color), alt.value("#888"),
            ),
            tooltip=[
                alt.Tooltip("_sets_label:N", title="Sets"),
                alt.Tooltip("cardinality:Q", title="Size"),
                alt.Tooltip("degree:Q", title="Degree"),
                alt.Tooltip("_pct_label:N", title="% of set"),
            ],
        )
    )

    matrix_layers: list[alt.Chart] = [bg_dots, active_dots]

    if len(data.lines_df) > 0:
        connecting_lines = (
            alt.Chart(data.lines_df)
            .mark_rule(strokeWidth=2)
            .encode(
                x=alt.X("_intersection_id:N", sort=x_domain, axis=None),
                y=alt.Y(
                    "_y_min:Q",
                    scale=alt.Scale(domain=[-0.5, n_sets - 0.5]),
                ),
                y2=alt.Y2("_y_max:Q"),
                color=alt.condition(
                    highlight, alt.value(line_color), alt.value("#bbb"),
                ),
            )
        )
        # lines between background and active dots
        matrix_layers = [bg_dots, connecting_lines, active_dots]

    matrix_chart = (
        alt.layer(*matrix_layers)
        .add_params(highlight)
        .properties(width=width, height=matrix_height)
    )

    # ── assemble main column ──────────────────────────────────────────
    main = alt.vconcat(bar_chart, matrix_chart, spacing=0).resolve_scale(
        x="shared",
    )

    # ── optional set-size bars (left) ─────────────────────────────────
    if show_set_sizes:
        set_bar_width = 120
        bar_thickness = max(8, matrix_height // (n_sets + 1))
        # Use mark_bar oriented horizontally: x encodes the set size,
        # y is the categorical position (quantitative with custom labels).
        # The `size` param sets the bar thickness in pixels and `orient`
        # forces horizontal so the bar extends from x=0 to x=set_size.
        set_size_chart = (
            alt.Chart(data.set_sizes_df)
            .mark_bar(size=bar_thickness, orient="horizontal")
            .encode(
                x=alt.X(
                    "set_size:Q",
                    title="Set Size",
                    scale=alt.Scale(reverse=True),
                ),
                y=alt.Y("_y_pos:Q", scale=y_scale, axis=y_axis),
                color=alt.value(bar_color),
                tooltip=[
                    alt.Tooltip("set_name:N", title="Set"),
                    alt.Tooltip("set_size:Q", title="Size"),
                ],
            )
            .properties(width=set_bar_width, height=matrix_height)
        )
        # empty spacer above the set-size bars to align with the bar chart
        spacer = (
            alt.Chart(pl.DataFrame({"x": [0]}))
            .mark_point(opacity=0, size=0)
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("x:Q", axis=None))
            .properties(width=set_bar_width, height=height)
        )
        left = alt.vconcat(spacer, set_size_chart, spacing=0)
        chart = alt.hconcat(left, main, spacing=5)
    else:
        chart = main

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5,
    ).configure_view(strokeWidth=0)
