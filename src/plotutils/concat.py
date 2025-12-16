import altair as alt
import polars as pl
from typing import Protocol


class ChartFunc(Protocol):
    def __call__(self, df: pl.DataFrame, title: str, *args, **kwargs) -> alt.Chart: ...


def vchart(
    *args,
    row: str,
    df: pl.DataFrame,
    func: ChartFunc,
    **kwargs,
) -> alt.VConcatChart:
    """Stack vertical facet charts.

    Each chart is facetted over column dimension, with shared y-axis.
    Allows to have row-wise shared y-axes.
    """

    return alt.vconcat(
        *(func(_df, *args, title=_name, **kwargs) for _name, _df in df.group_by(row))
    )


def hchart(
    *args,
    column: str,
    df: pl.DataFrame,
    func: ChartFunc,
    **kwargs,
) -> alt.HConcatChart:
    """Stack horizontal facet charts.

    Each chart is facetted over row dimension, with shared x-axis.
    Allows to have column-wise shared x-axes.
    """

    return alt.hconcat(
        *(func(_df, *args, title=_name, **kwargs) for _name, _df in df.group_by(column))
    )
