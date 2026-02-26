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


def _build_identity_line(
    val_min: float,
    val_max: float,
    color: str = "gray",
    stroke_dash: list[int] | None = None,
    scale: alt.Scale | None = None,
) -> alt.Chart:
    """Build a dashed y = x identity line between *val_min* and *val_max*.

    Parameters
    ----------
    val_min, val_max : float
        Endpoints of the identity line on both axes.
    color : str
        Stroke color of the line.
    stroke_dash : list[int] or None
        Dash pattern. Defaults to ``[5, 5]``.
    scale : alt.Scale or None
        Scale applied to both axes. When *None* the Altair default is used.

    Returns
    -------
    alt.Chart
        A single-layer line chart representing y = x.
    """
    if stroke_dash is None:
        stroke_dash = [5, 5]
    x_enc = alt.X("x:Q", scale=scale) if scale else alt.X("x:Q")
    y_enc = alt.Y("y:Q", scale=scale) if scale else alt.Y("y:Q")
    return (
        alt.Chart(pl.DataFrame({"x": [val_min, val_max], "y": [val_min, val_max]}))
        .mark_line(color=color, strokeDash=stroke_dash)
        .encode(x=x_enc, y=y_enc)
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
        layers.insert(
            0, _build_identity_line(y_min, y_max, color=identity_line_color, scale=_scale)
        )

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


def plot_predictions_errors(
    df: pl.DataFrame,
    true_col: str = "true",
    pred_col: str = "pred",
    title: str = "",
    width: int = 600,
    height: int = 600,
    x_title: str | None = None,
    y_title: str | None = None,
    point_color: str = "steelblue",
    color_col: str | None = None,
    shape_col: str | None = None,
    identity_line_color: str = "gray",
    point_size: int = 60,
    point_opacity: float = 0.7,
) -> alt.LayerChart:
    """Create a prediction-vs-truth scatter plot with a y = x identity line.

    Each point represents one observation. The x-axis shows the true
    (ground-truth) value and the y-axis shows the predicted value. Both
    axes share the same scale domain so the identity line is a true
    diagonal. Points on the identity line represent perfect predictions.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with at least *true_col* and *pred_col* columns.
    true_col : str
        Column for true (ground-truth) values (x-axis).
    pred_col : str
        Column for predicted values (y-axis).
    title : str
        Plot title.
    width, height : int
        Chart dimensions. Default 600x600 (square) to reinforce equal
        axis scaling.
    x_title, y_title : str or None
        Axis titles (defaults to column names).
    point_color : str
        Fixed color for all points. Ignored when *color_col* is set.
    color_col : str or None
        Column mapped to point color (nominal). When set, *point_color*
        is ignored and Altair picks a categorical palette automatically.
    shape_col : str or None
        Column mapped to point shape (nominal). When set, each unique
        value gets a distinct marker shape.
    identity_line_color : str
        Color of the y = x dashed identity line.
    point_size : int
        Size (area) of the scatter points.
    point_opacity : float
        Opacity of the scatter points (0.0 -- 1.0).

    Returns
    -------
    alt.LayerChart
        Altair layered chart with scatter points and identity line.
    """
    alt.data_transformers.disable_max_rows()

    x_title = x_title or true_col
    y_title = y_title or pred_col

    # Sort for deterministic SVG rendering
    sort_cols = [true_col, pred_col]
    if color_col is not None:
        sort_cols.append(color_col)
    if shape_col is not None and shape_col not in sort_cols:
        sort_cols.append(shape_col)
    df = df.sort(*sort_cols)

    # Shared scale: same domain for both axes, with 2% padding for visibility
    raw_min = float(
        min(df[true_col].min(), df[pred_col].min())  # type: ignore[arg-type]
    )
    raw_max = float(
        max(df[true_col].max(), df[pred_col].max())  # type: ignore[arg-type]
    )
    pad = (raw_max - raw_min) * 0.02
    global_min = raw_min - pad
    global_max = raw_max + pad
    shared_scale = alt.Scale(domain=[global_min, global_max])

    # Encodings
    x_enc = alt.X(f"{true_col}:Q", title=x_title, scale=shared_scale)
    y_enc = alt.Y(f"{pred_col}:Q", title=y_title, scale=shared_scale)

    encode_kwargs: dict = {"x": x_enc, "y": y_enc}

    if color_col is not None:
        encode_kwargs["color"] = alt.Color(f"{color_col}:N")

    if shape_col is not None:
        encode_kwargs["shape"] = alt.Shape(f"{shape_col}:N")

    tooltip_cols = [f"{true_col}:Q", f"{pred_col}:Q"]
    if color_col is not None:
        tooltip_cols.append(f"{color_col}:N")
    if shape_col is not None and shape_col != color_col:
        tooltip_cols.append(f"{shape_col}:N")
    encode_kwargs["tooltip"] = tooltip_cols

    # Build scatter layer
    mark_kwargs: dict = {
        "filled": True,
        "size": point_size,
        "opacity": point_opacity,
    }
    if color_col is None:
        mark_kwargs["color"] = point_color

    points = alt.Chart(df).mark_point(**mark_kwargs).encode(**encode_kwargs)

    # Identity line (always shown, behind points)
    identity_layer = _build_identity_line(
        global_min, global_max, color=identity_line_color, scale=shared_scale
    )

    layers: list[alt.Chart | alt.LayerChart] = [identity_layer, points]

    chart = alt.layer(*layers).properties(width=width, height=height)

    if title:
        chart = chart.properties(title=title)

    return chart.configure_axis(
        gridColor="gray", gridDash=[3, 3], gridOpacity=0.5
    ).configure_view(strokeWidth=0)
