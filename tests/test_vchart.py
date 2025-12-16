import pytest
import polars as pl
import altair as alt
from plotutils.concat import vchart


# A valid ChartFunc implementation
def valid_func(df: pl.DataFrame, title: str, *args, **kwargs) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_point()
        .encode(x=df.columns[0], y=df.columns[1])
        .properties(title=title)
    )


# An invalid function: missing 'title' argument
def invalid_func_missing_title(df: pl.DataFrame) -> alt.Chart:
    return alt.Chart(df).mark_point()


# An invalid function: wrong return type
def invalid_func_wrong_return(df: pl.DataFrame, title: str, *args, **kwargs):
    return "not a chart"


def test_vchart_with_valid_func():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "row": ["x", "y"]})
    chart = vchart(row="row", df=df, func=valid_func)
    assert isinstance(chart, alt.VConcatChart)


def test_vchart_with_invalid_func_missing_title():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "row": ["x", "y"]})
    with pytest.raises(TypeError):
        vchart(row="row", df=df, func=invalid_func_missing_title)


def test_vchart_with_invalid_func_wrong_return():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "row": ["x", "y"]})
    with pytest.raises(
        Exception
    ):  # Could be AssertionError or TypeError depending on runtime checks
        vchart(row="row", df=df, func=invalid_func_wrong_return)
