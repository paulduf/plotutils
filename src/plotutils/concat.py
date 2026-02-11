"""Facet-like concatenation with independent axes.

Altair's built-in facet shares *all* axes across every sub-chart.
:func:`vchart` and :func:`hchart` instead produce concatenated layouts
where each sub-chart computes its own axis domain independently, while
still appearing as a faceted grid.  Use ``vchart`` when sub-charts
should share the y-axis direction (stacked vertically, one per row) and
``hchart`` when they should share the x-axis direction (laid out
horizontally, one per column).

Both functions accept an optional second dimension (``column`` for
``vchart``, ``row`` for ``hchart``) which is handled by Altair's native
facet — sharing axes within each concat panel.  This gives a grid where
one direction has independent axes and the other has shared axes.
"""

import altair as alt
import polars as pl
from typing import Protocol


class ChartFunc(Protocol):
    def __call__(self, df: pl.DataFrame, title: str, *args, **kwargs) -> alt.Chart: ...


def _strip_config(charts: list) -> tuple[list, alt.Undefined.__class__]:
    """Remove config from sub-charts and return it separately.

    Altair forbids ``configure_*`` on charts nested inside concat
    operations.  This helper strips the config from the first chart
    that carries one (all sub-charts come from the same function, so
    the config is identical) and returns the cleaned list together with
    the extracted config so the caller can apply it at the top level.
    """
    config = alt.Undefined
    stripped = []
    for c in charts:
        if c.config is not alt.Undefined:
            config = c.config
            c = c.copy()
            c.config = alt.Undefined
        stripped.append(c)
    return stripped, config


def vchart(
    *args,
    row: str,
    column: str | None = None,
    df: pl.DataFrame,
    func: ChartFunc,
    **kwargs,
) -> alt.VConcatChart:
    """Stack charts vertically with independent axes per row.

    Each row gets its own axis domain.  When *column* is provided, each
    row is additionally faceted horizontally using Altair's native facet,
    so columns within the same row share axes.
    """
    groups = list(df.group_by(row, maintain_order=True))
    sub_charts = [func(_df, *args, title=_name, **kwargs) for _name, _df in groups]
    stripped, config = _strip_config(sub_charts)
    if column is not None:
        stripped = [
            c.facet(column=f"{column}:N", data=_df)
            for c, (_name, _df) in zip(stripped, groups)
        ]
    result = alt.vconcat(*stripped)
    if config is not alt.Undefined:
        result.config = config
    return result


def hchart(
    *args,
    column: str,
    row: str | None = None,
    df: pl.DataFrame,
    func: ChartFunc,
    **kwargs,
) -> alt.HConcatChart:
    """Lay out charts horizontally with independent axes per column.

    Each column gets its own axis domain.  When *row* is provided, each
    column is additionally faceted vertically using Altair's native facet,
    so rows within the same column share axes.
    """
    groups = list(df.group_by(column, maintain_order=True))
    sub_charts = [func(_df, *args, title=_name, **kwargs) for _name, _df in groups]
    stripped, config = _strip_config(sub_charts)
    if row is not None:
        stripped = [
            c.facet(row=f"{row}:N", data=_df)
            for c, (_name, _df) in zip(stripped, groups)
        ]
    result = alt.hconcat(*stripped)
    if config is not alt.Undefined:
        result.config = config
    return result
