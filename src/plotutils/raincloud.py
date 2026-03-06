"""Raincloud plots: half-violin + boxplot + jittered strip in a faceted layout.

A raincloud plot combines three complementary views of distributional data:

* **Half-violin** (KDE density) — shows the full probability density.
* **Boxplot** — shows summary statistics (quartiles, median, whiskers).
* **Jittered strip** — shows every individual data point.

Each category is rendered as an independent facet so that the violin density
and scatter jitter use their own x-scales while the y-scale is shared for
cross-group comparison.
"""

from __future__ import annotations

import random
from typing import Literal

import altair as alt
import polars as pl


def plot_raincloud(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    jitter: Literal["uniform", "gauss"] = "gauss",
    title: str = "",
    width: int | None = None,
    height: int = 300,
    y_title: str | None = None,
    jitter_seed: int = 0,
) -> alt.FacetChart:
    """Raincloud plot: half-violin, boxplot and jittered scatter per category.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing at least ``x_col`` (categorical) and ``y_col``
        (continuous).
    x_col : str
        Column used for grouping (one facet per unique value).
    y_col : str
        Column with the continuous variable to visualize.
    jitter : {"uniform", "gauss"}
        Type of horizontal jitter applied to the scatter points.
        ``"uniform"`` spreads points evenly; ``"gauss"`` uses a normal
        distribution centred at zero.
    title : str
        Chart title.
    width : int or None
        Per-facet width in pixels.  When *None* the width is computed
        automatically as ``max(min(600 / n_groups, 150), 100)``.
    height : int
        Chart height in pixels.
    y_title : str or None
        Y-axis title.  Defaults to ``y_col``.
    jitter_seed : int
        Seed for the random jitter (ensures reproducible renders).

    Returns
    -------
    alt.FacetChart
    """
    alt.data_transformers.disable_max_rows()

    rng = random.Random(jitter_seed)

    jitter_values: list[float] = []
    for group in df[x_col].unique(maintain_order=True).to_list():
        n = df.filter(pl.col(x_col) == group).height
        if jitter == "uniform":
            jitter_values.extend(
                rng.uniform(-100, 100) for _ in range(n)
            )
        elif jitter == "gauss":
            jitter_values.extend(
                rng.gauss(0, 0.05) for _ in range(n)
            )
        else:
            raise ValueError(f"Unknown jitter type: {jitter}")

    df_plot = df.with_columns(pl.Series("_jitter", jitter_values))

    # --- Half-violin (KDE density) ---
    violin = (
        alt.Chart()
        .transform_density(
            density=y_col,
            groupby=[x_col],
            as_=[y_col, "density"],
        )
        .mark_area(orient="horizontal", opacity=0.5)
        .encode(
            x=alt.X("density:Q", axis=None),
            y=alt.Y(f"{y_col}:Q", axis=alt.Axis(title=y_title or y_col)),
        )
    )

    # --- Boxplot ---
    box = (
        alt.Chart()
        .mark_boxplot(size=20, ticks=True, median={"color": "black"})
        .encode(
            x=alt.X(f"{x_col}:N", axis=None),
            y=alt.Y(f"{y_col}:Q"),
        )
    )

    # --- Jittered scatter ---
    scatter = (
        alt.Chart()
        .mark_circle(size=25, opacity=0.5)
        .encode(
            x=alt.X("_jitter:Q", axis=None),
            y=alt.Y(f"{y_col}:Q"),
        )
    )

    n_groups = df[x_col].n_unique()
    facet_width = width or max(min(600 // n_groups, 150), 100)

    chart = (
        (box + violin + scatter)
        .properties(width=facet_width, height=height)
        .facet(
            column=alt.Column(f"{x_col}:N"),
            data=df_plot,
        )
        .resolve_scale(x="independent", y="shared")
    )

    if title:
        chart = chart.properties(title=title)

    return chart
