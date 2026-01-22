from typing import Literal

import polars as pl
import altair as alt

ErrorBarExtent = Literal["ci", "stdev", "stderr", "iqr"]


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
) -> alt.LayerChart:
    """
    Create a scatter plot with error bars using Altair.

    The function aggregates multiple y values per x category, computing
    mean and confidence intervals automatically.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with raw data (multiple y values per x category)
    x_col : str
        Column for x-axis (categorical or numeric)
    y_col : str
        Column for y values (will be aggregated per x category)
    title : str
        Plot title
    width, height : int
        Chart dimensions
    x_title, y_title : str or None
        Axis titles (defaults to column names)
    point_color : str
        Color for points and error bars
    extent : str
        Error bar extent: "ci" (95% CI), "stdev", "stderr", or "iqr"
    identity_line : bool
        If True, adds y = x identity line
    identity_line_color : str
        Color of the identity line
    zero : bool
        If True, y-axis scale includes zero (x-axis always excludes zero when numeric)
    x_labels : dict[float, str] or None
        Mapping of numeric x values to custom labels (enables quantitative x-axis)

    Returns
    -------
    alt.LayerChart
        Altair layered chart with points and error bars
    """
    alt.data_transformers.disable_max_rows()

    x_title = x_title or x_col
    y_title = y_title or y_col

    # Sort data for deterministic rendering
    df = df.sort(x_col, y_col)

    y_scale = alt.Scale(zero=zero)

    # Determine if x is quantitative (numeric with labels) or nominal (categorical)
    if x_labels is not None:
        # Add padding to avoid points on edges
        x_values = list(x_labels.keys())
        x_min, x_max = min(x_values), max(x_values)
        x_padding = (x_max - x_min) * 0.1  # 10% padding on each side
        x_scale = alt.Scale(zero=False, domain=[x_min - x_padding, x_max + x_padding])
        # Build labelExpr for custom tick labels
        label_expr = " : ".join(
            f"datum.value === {pos} ? '{lbl}'" for pos, lbl in x_labels.items()
        ) + " : datum.value"
        x_axis = alt.Axis(values=x_values, labelExpr=label_expr)
        x_encoding = alt.X(f"{x_col}:Q", title=x_title, scale=x_scale, axis=x_axis)
        x_encoding_simple = alt.X(f"{x_col}:Q", scale=x_scale)
    else:
        x_encoding = alt.X(f"{x_col}:N", title=x_title)
        x_encoding_simple = alt.X(f"{x_col}:N")

    # Error bars with aggregation
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(extent=extent, color=point_color)
        .encode(
            x=x_encoding,
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=y_scale),
        )
    )

    # Points showing mean
    points = (
        alt.Chart(df)
        .mark_point(filled=True, color=point_color)
        .encode(
            x=x_encoding_simple,
            y=alt.Y(f"mean({y_col}):Q", scale=y_scale),
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

    return (
        chart
        .configure_axis(gridColor="gray", gridDash=[3, 3], gridOpacity=0.5)
        .configure_view(strokeWidth=0)
    )
