from typing import Literal, NamedTuple

import polars as pl
import altair as alt

ErrorBarExtent = Literal["ci", "stdev", "stderr", "iqr"]


class _XEncodings(NamedTuple):
    """Pair of x-axis encodings returned by :func:`_build_x_encodings`.

    *full* carries the axis title and custom label configuration.
    *simple* has only the type and scale (used when Altair already inherits
    the axis from another layer).
    """

    full: alt.X
    simple: alt.X


def _build_x_encodings(
    x_col: str,
    x_title: str,
    x_labels: dict[float, str] | None,
    scale: alt.Scale,
) -> _XEncodings:
    """Build the full and simple x-axis encodings.

    When *x_labels* is provided the axis is quantitative with custom tick
    labels; otherwise it is nominal (categorical).
    """
    if x_labels is not None:
        x_values = list(x_labels.keys())
        label_expr = (
            " : ".join(
                f"datum.value === {pos} ? '{lbl}'" for pos, lbl in x_labels.items()
            )
            + " : datum.value"
        )
        x_axis = alt.Axis(values=x_values, labelExpr=label_expr)
        return _XEncodings(
            full=alt.X(f"{x_col}:Q", title=x_title, scale=scale, axis=x_axis),
            simple=alt.X(f"{x_col}:Q", scale=scale),
        )
    return _XEncodings(
        full=alt.X(f"{x_col}:N", title=x_title),
        simple=alt.X(f"{x_col}:N"),
    )


def plot_confidence_scatter(
    df: pl.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    title: str = "",
    width: int = 600,
    height: int = 400,
    x_title: str | None = None,
    y_title: str | None = None,
    point_color: str = "steelblue",
    extent: ErrorBarExtent = "ci",
    identity_line: bool = False,
    identity_line_color: str = "gray",
    zero: bool = False,
    x_labels: dict[float, str] | None = None,
    scale_type: Literal["linear", "log"] = "linear",
) -> alt.LayerChart:
    """Create a scatter plot with error bars using Altair.

    The function aggregates multiple y values per x category, computing
    mean and confidence intervals automatically.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with raw data (multiple y values per x category).
    x_col : str
        Column for x-axis (categorical or numeric).
    y_col : str
        Column for y values (will be aggregated per x category).
    title : str
        Plot title.
    width, height : int
        Chart dimensions.
    x_title, y_title : str or None
        Axis titles (defaults to column names).
    point_color : str
        Color for points and error bars.
    extent : str
        Error bar extent: "ci" (95% CI), "stdev", "stderr", or "iqr".
    identity_line : bool
        If True, adds y = x identity line.
    identity_line_color : str
        Color of the identity line.
    zero : bool
        If True, y-axis scale includes zero.
    x_labels : dict[float, str] or None
        Mapping of numeric x values to custom labels (enables quantitative
        x-axis with labelled ticks).
    scale_type : str
        Scale type for both axes: "linear" or "log".

    Returns
    -------
    alt.LayerChart
        Altair layered chart with points and error bars.
    """
    alt.data_transformers.disable_max_rows()

    x_title = x_title or x_col
    y_title = y_title or y_col

    # Sort for deterministic SVG rendering (vl-convert renders data in input order)
    df = df.sort(x_col, y_col)

    _scale = alt.Scale(zero=zero, type=scale_type)
    x_enc = _build_x_encodings(x_col, x_title, x_labels, _scale)

    # Error bars with aggregation
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(extent=extent, color=point_color)
        .encode(
            x=x_enc.full,
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=_scale),
        )
    )

    # Points showing mean
    points = (
        alt.Chart(df)
        .mark_point(filled=True, color=point_color)
        .encode(
            x=x_enc.simple,
            y=alt.Y(f"mean({y_col}):Q", scale=_scale),
        )
    )

    layers = [error_bars, points]

    # Identity line
    if identity_line:
        y_min = float(df[y_col].min())  # type: ignore[arg-type]
        y_max = float(df[y_col].max())  # type: ignore[arg-type]

        identity_layer = (
            alt.Chart(pl.DataFrame({"x": [y_min, y_max], "y": [y_min, y_max]}))
            .mark_line(color=identity_line_color, strokeDash=[5, 5])
            .encode(x="x:Q", y="y:Q")
        )
        layers.insert(0, identity_layer)

    chart = alt.layer(*layers).properties(width=width, height=height)

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)


def plot_deviations(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    relative: bool = False,
    add_levels: list[float] | None = None,
    x_labels: dict[float, str] | None = None,
    scale_type: Literal["linear", "log"] = "linear",
) -> alt.LayerChart:
    """Create a plot showing deviations of y values from their per-group mean.

    Computes ``y - mean(y)`` per x group. When *relative* is True, computes
    ``(y - mean(y)) / mean(y)`` instead. A horizontal line at 0 is always
    drawn. Additional symmetric level lines (e.g. tolerance bands) can be
    added via *add_levels*.

    Parameters
    ----------
    df : pl.DataFrame
        Raw data with multiple y values per x category.
    x_col : str
        Column for x-axis (categorical or numeric).
    y_col : str
        Column for y values.
    title : str
        Plot title.
    relative : bool
        If True, deviations are divided by the group mean.
    add_levels : list[float] or None
        Extra horizontal levels drawn symmetrically at +level and -level.
    x_labels : dict[float, str] or None
        Mapping of numeric x values to custom labels (enables quantitative
        x-axis with labelled ticks).
    scale_type : str
        Scale type for the x-axis: "linear" or "log".
    """
    alt.data_transformers.disable_max_rows()

    # Compute deviations from per-group mean
    dev_col = "deviation"
    df_dev = df.with_columns(
        (
            (pl.col(y_col) - pl.col(y_col).mean().over(x_col))
            / (pl.col(y_col).mean().over(x_col) if relative else 1)
        ).alias(dev_col)
    )

    # Sort for deterministic SVG rendering (vl-convert renders data in input order)
    df_dev = df_dev.sort(x_col, dev_col)

    y_title = dev_col if not relative else "relative deviation"

    _scale = alt.Scale(type=scale_type)
    x_enc = _build_x_encodings(x_col, x_col, x_labels, _scale)

    points = (
        alt.Chart(df_dev)
        .mark_point(filled=True, color="steelblue")
        .encode(
            x=x_enc.full,
            y=alt.Y(f"{dev_col}:Q", title=y_title),
        )
    )

    # Horizontal zero line
    zero_line = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(y="y:Q")
    )

    layers: list[alt.Chart | alt.LayerChart] = [points, zero_line]

    # Optional symmetric level lines
    if add_levels:
        level_values = [v for lv in add_levels for v in (lv, -lv)]
        level_df = pl.DataFrame({"y": level_values})
        level_lines = (
            alt.Chart(level_df)
            .mark_rule(color="black", strokeDash=[3, 3])
            .encode(y="y:Q")
        )
        layers.append(level_lines)

    chart = alt.layer(*layers).properties(width=600, height=400)

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)
