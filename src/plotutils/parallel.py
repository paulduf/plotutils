"""Parallel coordinates plots for multivariate data exploration.

Draws one line per observation across a shared set of vertical axes,
one per selected column.  Supports optional min-max normalization
(with per-column log transform) and interactive line highlighting
on mouseover.
"""

from __future__ import annotations

import math
from typing import Literal

import altair as alt
import polars as pl


def plot_parallel_coordinates(
    df: pl.DataFrame,
    columns: list[str],
    color_col: str | None = None,
    id_col: str | None = None,
    normalize: bool = False,
    transforms: list[Literal["linear", "log"]] | None = None,
    title: str = "",
    width: int = 600,
    height: int = 400,
    opacity: float = 0.5,
    highlight_opacity: float = 1.0,
) -> alt.LayerChart:
    """Parallel coordinates plot for multivariate data.

    Each row of ``df`` is drawn as a poly-line that passes through one
    vertical axis per column in ``columns``.  Hovering a line highlights
    it and shows a tooltip with all column values.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    columns : list[str]
        Column names to display as parallel axes (left to right).
    color_col : str or None
        Optional column for color encoding (categorical).
    id_col : str or None
        Optional column with sample identifiers, shown in tooltip.
    normalize : bool
        If True, apply min-max normalization to each column so all axes
        share the 0–1 range.
    transforms : list of {"linear", "log"} or None
        Per-column transform applied *before* normalization.  Must have
        the same length as ``columns``.  Only used when
        ``normalize=True``.  ``"log"`` applies a natural-log transform;
        ``"linear"`` leaves the column unchanged.
    title : str
        Chart title.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.
    opacity : float
        Default line opacity for unselected lines.
    highlight_opacity : float
        Line opacity when highlighted on mouseover.

    Returns
    -------
    alt.LayerChart
    """
    alt.data_transformers.disable_max_rows()

    if transforms is not None and len(transforms) != len(columns):
        raise ValueError(
            f"transforms length ({len(transforms)}) must match "
            f"columns length ({len(columns)})"
        )

    # --- data preparation (Polars) ----------------------------------------
    df_work = df.with_row_index("_index")

    # Keep only needed columns
    keep = ["_index"] + list(columns)
    if color_col is not None:
        keep.append(color_col)
    if id_col is not None:
        keep.append(id_col)
    df_work = df_work.select([c for c in keep if c in df_work.columns])

    if normalize:
        for i, col in enumerate(columns):
            series = df_work[col].cast(pl.Float64)

            # Apply log transform if requested
            if transforms is not None and transforms[i] == "log":
                series = series.log()

            col_min = series.min()
            col_max = series.max()
            if col_min is not None and col_max is not None and col_max != col_min:
                series = (series - col_min) / (col_max - col_min)
            else:
                series = pl.Series(col, [0.5] * len(series))

            df_work = df_work.with_columns(series.alias(col))

    # --- tooltip layer (wide format) --------------------------------------
    # Build tooltip from original (wide) data so all column values appear.
    tooltip_fields: list[alt.Tooltip] = []
    if id_col is not None:
        tooltip_fields.append(alt.Tooltip(f"{id_col}:N", title="ID"))
    if color_col is not None:
        tooltip_fields.append(alt.Tooltip(f"{color_col}:N"))
    for col in columns:
        tooltip_fields.append(alt.Tooltip(f"{col}:Q", format=".3f"))

    # --- selection for highlight ------------------------------------------
    highlight = alt.selection_point(
        on="pointerover",
        fields=["_index"],
        nearest=True,
    )

    # --- fold data for line chart -----------------------------------------
    # Altair's transform_fold works on the Vega-Lite side, but we need the
    # wide-format DataFrame for tooltips.  We fold in Polars instead so
    # both layers share the same base DataFrame.
    id_vars = ["_index"]
    if color_col is not None:
        id_vars.append(color_col)
    if id_col is not None:
        id_vars.append(id_col)

    df_long = df_work.unpivot(
        on=columns,
        index=id_vars,
        variable_name="_axis",
        value_name="_value",
    )

    # Preserve column order via a categorical with the original order
    df_long = df_long.with_columns(
        pl.col("_axis").cast(pl.Categorical),
    )

    # Sort for deterministic rendering
    df_long = df_long.sort("_index", "_axis")

    # Re-attach wide columns for tooltip (join back)
    df_long = df_long.join(
        df_work.select(["_index"] + list(columns)),
        on="_index",
        how="left",
        suffix="_orig",
    )

    # --- line layer -------------------------------------------------------
    line_encode = {
        "x": alt.X("_axis:N", title=None, sort=columns,
                    axis=alt.Axis(labelAngle=0)),
        "y": alt.Y("_value:Q", title="Normalized value" if normalize else "Value"),
        "detail": alt.Detail("_index:N"),
        "opacity": alt.condition(
            highlight,
            alt.value(highlight_opacity),
            alt.value(opacity * 0.3),
        ),
        "tooltip": tooltip_fields,
    }
    if color_col is not None:
        line_encode["color"] = alt.Color(f"{color_col}:N")
        line_encode["strokeWidth"] = alt.condition(
            highlight,
            alt.value(3),
            alt.value(1),
        )
    else:
        line_encode["strokeWidth"] = alt.condition(
            highlight,
            alt.value(3),
            alt.value(1),
        )

    lines = (
        alt.Chart(df_long)
        .mark_line()
        .encode(**line_encode)
        .add_params(highlight)
    )

    # --- invisible point layer for reliable hover -------------------------
    point_encode = {
        "x": alt.X("_axis:N", sort=columns),
        "y": alt.Y("_value:Q"),
        "detail": alt.Detail("_index:N"),
        "tooltip": tooltip_fields,
    }
    if color_col is not None:
        point_encode["color"] = alt.Color(f"{color_col}:N")

    points = (
        alt.Chart(df_long)
        .mark_point(opacity=0, size=100)
        .encode(**point_encode)
        .add_params(highlight)
    )

    chart = alt.layer(lines, points).properties(width=width, height=height)

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)
