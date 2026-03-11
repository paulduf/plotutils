"""Forest plot: per-item effect estimates with confidence intervals on a log scale.

A forest plot places one row per item (subgroup, variable, study, …) and shows
a **point estimate** and a **horizontal confidence-interval bar** for each.
The x-axis uses a logarithmic scale with natural-scale tick labels, making the
chart suitable for hazard ratios, odds ratios, and risk ratios.

A solid vertical rule marks the null effect (default ``null_value=1.0``).
Optional dashed symmetric rules mark a minimum clinically meaningful effect.
"""

from __future__ import annotations

import altair as alt
import polars as pl


def plot_forest(
    df: pl.DataFrame,
    center_col: str,
    low_col: str,
    high_col: str,
    label_col: str,
    *,
    null_value: float = 1.0,
    min_effect: float | None = None,
    title: str = "",
    width: int = 400,
    height: int | None = None,
    color_col: str | None = None,
    sort_col: str | None = None,
    ascending: bool = True,
    x_title: str | None = None,
    y_title: str | None = None,
) -> alt.LayerChart:
    """Forest plot with log-scale x-axis.

    Parameters
    ----------
    df : pl.DataFrame
        Data with one row per item.  Must contain ``center_col``, ``low_col``,
        ``high_col``, and ``label_col``.
    center_col : str
        Column with the point estimate (e.g. hazard ratio).
    low_col : str
        Column with the lower bound of the confidence interval.
    high_col : str
        Column with the upper bound of the confidence interval.
    label_col : str
        Column with the item label shown on the y-axis.  Row order in *df* is
        preserved (``sort=None``).
    null_value : float
        x-position of the "no effect" vertical rule.  Use ``1.0`` for ratios
        (HR, OR, RR) and ``0.0`` is not supported because ``log(0)`` is
        undefined.  Default ``1.0``.
    min_effect : float or None
        When provided, two dashed rules are drawn at
        ``null_value * min_effect`` and ``null_value / min_effect``,
        representing the boundary of a minimum meaningful effect.  For
        example, ``min_effect=1.25`` with ``null_value=1.0`` draws lines at
        0.8 and 1.25.
    title : str
        Chart title.
    width : int
        Chart width in pixels.
    height : int or None
        Chart height in pixels.  When ``None`` (default) the height is
        computed automatically as ``max(80, n_rows * 22 + 40)``.
    color_col : str or None
        Optional column used to colour CI bars and center points.  When
        provided an ``alt.Color(:N)`` legend is added.
    sort_col : str or None
        Column to sort rows by.  When ``None`` (default) the DataFrame row
        order is preserved.  Any numeric or string column in *df* is accepted
        (e.g. ``center_col`` to rank by effect size).
    ascending : bool
        Sort direction when ``sort_col`` is set.  ``True`` = smallest value at
        the top; ``False`` = largest value at the top.  Default ``True``.
    x_title : str or None
        X-axis title.  Defaults to ``center_col``.
    y_title : str or None
        Y-axis title.  Defaults to ``label_col``.

    Returns
    -------
    alt.LayerChart
    """
    alt.data_transformers.disable_max_rows()

    n_rows = len(df)
    _height = height if height is not None else max(80, n_rows * 22 + 40)

    log_scale = alt.Scale(type="log")

    y_sort: alt.EncodingSortField | None
    if sort_col is not None:
        y_sort = alt.EncodingSortField(
            field=sort_col, order="ascending" if ascending else "descending"
        )
    else:
        y_sort = None

    color_enc: alt.Color | alt.value
    if color_col is not None:
        color_enc = alt.Color(f"{color_col}:N")
    else:
        color_enc = alt.value("steelblue")

    # ── Reference lines ──────────────────────────────────────────────────────

    null_rule = (
        alt.Chart()
        .mark_rule(color="#333", strokeWidth=1.2)
        .encode(x=alt.X(datum=null_value, scale=log_scale))
    )

    layers: list = [null_rule]

    if min_effect is not None:
        me_hi = null_value * min_effect
        me_lo = null_value / min_effect
        me_rule_hi = (
            alt.Chart()
            .mark_rule(color="#888", strokeDash=[4, 4], strokeWidth=1.0)
            .encode(x=alt.X(datum=me_hi, scale=log_scale))
        )
        me_rule_lo = (
            alt.Chart()
            .mark_rule(color="#888", strokeDash=[4, 4], strokeWidth=1.0)
            .encode(x=alt.X(datum=me_lo, scale=log_scale))
        )
        layers += [me_rule_lo, me_rule_hi]

    # ── CI bars ──────────────────────────────────────────────────────────────

    ci_bars = (
        alt.Chart(df)
        .mark_rule(strokeWidth=1.5)
        .encode(
            x=alt.X(
                f"{low_col}:Q",
                scale=log_scale,
                title=x_title or center_col,
            ),
            x2=alt.X2(f"{high_col}:Q"),
            y=alt.Y(
                f"{label_col}:N",
                sort=y_sort,
                title=y_title or label_col,
                axis=alt.Axis(labelLimit=200),
            ),
            color=color_enc,
        )
    )
    layers.append(ci_bars)

    # ── Center points ────────────────────────────────────────────────────────

    tooltip = [
        alt.Tooltip(f"{label_col}:N"),
        alt.Tooltip(f"{center_col}:Q", format=".3f", title="Estimate"),
        alt.Tooltip(f"{low_col}:Q", format=".3f", title="Low"),
        alt.Tooltip(f"{high_col}:Q", format=".3f", title="High"),
    ]
    if color_col is not None:
        tooltip.insert(1, alt.Tooltip(f"{color_col}:N"))

    center_pts = (
        alt.Chart(df)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(f"{center_col}:Q", scale=log_scale),
            y=alt.Y(f"{label_col}:N", sort=y_sort),
            color=color_enc,
            tooltip=tooltip,
        )
    )
    layers.append(center_pts)

    # ── Assemble ─────────────────────────────────────────────────────────────

    chart: alt.LayerChart = alt.layer(*layers).properties(
        width=width, height=_height
    )

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)
