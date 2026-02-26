"""Doubly-grouped distribution plots for one variable against two binary outcomes.

Two chart variants are provided:

* :func:`plot_bivariate_boxes` — side-by-side boxplots for each
  (label_x, label_y) combination.
* :func:`plot_bivariate_strip` — jittered strip plot showing every individual
  sample, coloured by label_y and spread horizontally within each label_x band.
"""

from __future__ import annotations

import random

import altair as alt
import polars as pl


def plot_bivariate_boxes(
    df: pl.DataFrame,
    score_col: str,
    label_x_col: str,
    label_y_col: str,
    title: str = "",
    width: int = 340,
    height: int = 300,
    y_title: str | None = None,
    x_title: str | None = None,
    color_title: str | None = None,
    id_col: str | None = None,
) -> alt.Chart | alt.LayerChart:
    """Doubly-grouped boxplot: score vs two binary outcomes.

    The outer grouping (x-axis) is ``label_x_col``; the inner grouping
    (side-by-side boxes within each outer group) is ``label_y_col``.
    This produces four boxes — one per (label_x, label_y) combination —
    making it easy to read off the marginal and joint effects of both
    outcomes on the variable's distribution.

    Parameters
    ----------
    df : pl.DataFrame
        Raw data with score and binary label columns.
    score_col : str
        Column with the continuous variable.
    label_x_col : str
        First binary outcome column (outer x-axis grouping).
    label_y_col : str
        Second binary outcome column (inner grouping via color / xOffset).
    title : str
        Chart title.
    width, height : int
        Chart dimensions in pixels.
    y_title : str or None
        Y-axis title. Defaults to ``score_col``.
    x_title : str or None
        X-axis title. Defaults to ``label_x_col``.
    color_title : str or None
        Legend title for the color encoding. Defaults to ``label_y_col``.
    id_col : str or None
        Optional column name containing patient / sample identifiers.  When
        provided, a transparent point layer is added on top of the boxes so
        that hovering over an individual data point reveals its ID.

    Returns
    -------
    alt.Chart or alt.LayerChart
    """
    alt.data_transformers.disable_max_rows()

    df_plot = df.with_columns(
        pl.col(label_x_col).cast(pl.Utf8),
        pl.col(label_y_col).cast(pl.Utf8),
    )

    chart: alt.Chart | alt.LayerChart = (
        alt.Chart(df_plot)
        .mark_boxplot()
        .encode(
            x=alt.X(
                f"{label_x_col}:N",
                title=x_title or label_x_col,
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset(f"{label_y_col}:N"),
            y=alt.Y(f"{score_col}:Q", title=y_title or score_col),
            color=alt.Color(f"{label_y_col}:N", title=color_title or label_y_col),
        )
        .properties(width=width, height=height)
    )

    if id_col is not None:
        id_layer = (
            alt.Chart(df_plot)
            .mark_point(opacity=0, size=300, filled=True)
            .encode(
                x=alt.X(f"{label_x_col}:N"),
                xOffset=alt.XOffset(f"{label_y_col}:N"),
                y=alt.Y(f"{score_col}:Q"),
                tooltip=[
                    alt.Tooltip(f"{id_col}:N", title="ID"),
                    alt.Tooltip(f"{label_x_col}:N"),
                    alt.Tooltip(f"{label_y_col}:N"),
                    alt.Tooltip(f"{score_col}:Q", format=".3f"),
                ],
            )
        )
        chart = alt.layer(chart, id_layer)

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)


def plot_bivariate_strip(
    df: pl.DataFrame,
    score_col: str,
    label_x_col: str,
    label_y_col: str,
    title: str = "",
    width: int = 340,
    height: int = 300,
    y_title: str | None = None,
    x_title: str | None = None,
    color_title: str | None = None,
    jitter_seed: int = 0,
    id_col: str | None = None,
) -> alt.Chart:
    """Jittered strip plot: score vs two binary outcomes.

    Each sample is drawn as a semi-transparent circle. Within each
    ``label_x_col`` band the points are spread horizontally: the two
    ``label_y_col`` sub-groups are offset to opposite sides of the band
    centre (±15 px by default), and an additional small random jitter
    (±7 px) reduces overplotting within each sub-group.

    Parameters
    ----------
    df : pl.DataFrame
        Raw data with score and binary label columns.
    score_col : str
        Column with the continuous variable.
    label_x_col : str
        First binary outcome column (outer x-axis grouping).
    label_y_col : str
        Second binary outcome column (inner color grouping).
    title : str
        Chart title.
    width, height : int
        Chart dimensions in pixels.
    y_title : str or None
        Y-axis title. Defaults to ``score_col``.
    x_title : str or None
        X-axis title. Defaults to ``label_x_col``.
    color_title : str or None
        Legend title for the color encoding. Defaults to ``label_y_col``.
    jitter_seed : int
        Seed for the random horizontal jitter (reproducible renders).
    id_col : str or None
        Optional column name containing patient / sample identifiers.  When
        provided, the ID appears in the tooltip on mouseover.

    Returns
    -------
    alt.Chart
    """
    alt.data_transformers.disable_max_rows()

    # Map each unique label_y value to a symmetric pixel offset so that the
    # two sub-groups are spread to opposite sides of the x-band centre.
    unique_ly = sorted(df[label_y_col].unique().to_list())
    n_ly = len(unique_ly)
    ly_to_idx = {v: i for i, v in enumerate(unique_ly)}

    half_span_px = 15  # half-distance between sub-group centres (pixels)
    jitter_px = 7      # max random jitter per point (pixels)

    rng = random.Random(jitter_seed)
    ly_list = df[label_y_col].to_list()

    if n_ly <= 1:
        offsets = [rng.uniform(-jitter_px, jitter_px) for _ in ly_list]
    else:
        offsets = [
            (2 * ly_to_idx[ly] / (n_ly - 1) - 1) * half_span_px
            + rng.uniform(-jitter_px, jitter_px)
            for ly in ly_list
        ]

    df_plot = df.with_columns(
        pl.col(label_x_col).cast(pl.Utf8),
        pl.col(label_y_col).cast(pl.Utf8),
        pl.Series("_x_offset", offsets),
    )

    tooltip = [
        alt.Tooltip(f"{label_x_col}:N"),
        alt.Tooltip(f"{label_y_col}:N"),
        alt.Tooltip(f"{score_col}:Q", format=".3f"),
    ]
    if id_col is not None:
        tooltip.insert(0, alt.Tooltip(f"{id_col}:N", title="ID"))

    chart = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.45, size=18)
        .encode(
            x=alt.X(
                f"{label_x_col}:N",
                title=x_title or label_x_col,
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("_x_offset:Q"),
            y=alt.Y(f"{score_col}:Q", title=y_title or score_col),
            color=alt.Color(f"{label_y_col}:N", title=color_title or label_y_col),
            tooltip=tooltip,
        )
        .properties(width=width, height=height)
    )

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)
