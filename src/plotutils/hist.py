import math
import polars as pl
import altair as alt


def plot_grouped_histogram(
    data: dict[str, list[float]] | pl.DataFrame,
    n_bins: int = 30,
    width: int = 600,
    height: int = 400,
    x_title: str = "Value",
    y_title: str = "Counts",
    opacity: float = 0.7,
    value_column: str = "value",
    group_column: str = "group",
) -> alt.Chart:
    """
    Create a grouped (side-by-side) histogram using Altair.

    Parameters
    ----------
    - `data` : `dict[str, list[float]]` or pl.DataFrame
        Either a dictionary mapping group names to data arrays,
        or a polars DataFrame with value and group columns
    - `n_bins` : `int`
        Number of bins for histogram
    - `width`, `height` : `int`
        Chart dimensions
    - `x_title`, `y_title` : `str`
        Axis titles
    - `opacity` : `float`
        Bar opacity
    - `value_column` : `str`
        Name of the value column (only used when data is a DataFrame)
    - `group_column` : `str`
        Name of the group column (only used when data is a DataFrame)

    Returns
    -------
    `alt.Chart`
        Altair chart object
    """
    # Convert input to DataFrame if needed
    if isinstance(data, dict):
        df_list = []
        for group_name, values in data.items():
            df_list.append(
                pl.DataFrame(
                    {
                        "value": [float(v) for v in values],
                        "group": [group_name] * len(values),
                    }
                )
            )
        df_combined = pl.concat(df_list)
    else:
        df_combined = data.select(
            [pl.col(value_column).alias("value"), pl.col(group_column).alias("group")]
        )

    # Compute nice bin edges
    v_min_raw = df_combined["value"].min()
    v_max_raw = df_combined["value"].max()

    if v_min_raw is None or v_max_raw is None:
        raise ValueError("Cannot compute histogram: data contains no valid values")

    v_min = float(v_min_raw)  # type: ignore[arg-type]
    v_max = float(v_max_raw)  # type: ignore[arg-type]

    raw_bin_width = (v_max - v_min) / n_bins
    magnitude = 10 ** math.floor(math.log10(raw_bin_width))
    nice_widths = [1, 2, 5, 10]
    bin_width = magnitude * min(
        nice_widths, key=lambda x: abs(x * magnitude - raw_bin_width)
    )

    v_min_rounded = math.floor(v_min / bin_width) * bin_width

    # Bin the data
    df_binned = (
        df_combined.with_columns(
            [
                (
                    v_min_rounded
                    + ((pl.col("value") - v_min_rounded) / bin_width).floor()
                    * bin_width
                ).alias("bin_start")
            ]
        )
        .group_by(["bin_start", "group"])
        .agg(pl.len().alias("count"))
        # Sort both columns for deterministic SVG rendering
        # (polars group_by uses hash-based grouping with non-deterministic output order)
        .sort("bin_start", "group")
        .with_columns([pl.col("bin_start").cast(pl.Int32).alias("bin_label")])
    )

    # Disable vegafusion
    alt.data_transformers.disable_max_rows()

    # Create chart
    chart = (
        alt.Chart(df_binned)
        .mark_bar(opacity=opacity)
        .encode(
            alt.X("bin_label:O", title=x_title, axis=alt.Axis(labelAngle=-45)),
            alt.Y("count:Q", title=y_title),
            alt.XOffset("group:N"),
            alt.Color("group:N", legend=alt.Legend(title=None)),
            tooltip=["group:N", "bin_start:Q", "count:Q"],
        )
        .properties(width=width, height=height)
        .configure_axis(gridColor="gray", gridDash=[3, 3], gridOpacity=0.5)
        .configure_view(strokeWidth=0)
    )

    return chart
