import pytest
import polars as pl
import altair as alt
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.concat import hchart, vchart
from plotutils.uncertainty import plot_predictions_errors


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


# --- shared test data ---
# Two models compared across train/test splits and two groups (A, B).
# Train range ~1-6, test range ~50-95: independent axes per split are needed.
# Groups and models provide the facet and color dimensions.

_PRED_DF = pl.DataFrame({
    "true":  [1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
              50.0, 60.0, 70.0, 75.0, 80.0, 90.0, 50.0, 60.0, 70.0, 75.0, 80.0, 90.0],
    "pred":  [1.1, 2.3, 2.8,  4.2, 5.1, 5.8,   1.4, 1.7, 3.5,  3.6, 5.3, 5.5,
              52.0, 58.0, 72.0, 73.0, 82.0, 91.0, 55.0, 55.0, 75.0, 70.0, 84.0, 93.0],
    "split": ["train"] * 12 + ["test"] * 12,
    "group": (["A"] * 6 + ["B"] * 6) * 2,
    "model": (["linear"] * 3 + ["tree"] * 3) * 4,
})


# --- vchart basic tests ---


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


# --- hchart basic tests ---


def test_hchart_with_valid_func():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "col": ["x", "y"]})
    chart = hchart(column="col", df=df, func=valid_func)
    assert isinstance(chart, alt.HConcatChart)


def test_hchart_with_invalid_func_missing_title():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "col": ["x", "y"]})
    with pytest.raises(TypeError):
        hchart(column="col", df=df, func=invalid_func_missing_title)


# --- single-dimension concat ---


def test_vchart_predictions_errors_basic(snapshot, snapshot_svg):
    """Vertical concat of prediction error plots, one per split."""
    chart = vchart(row="split", df=_PRED_DF, func=plot_predictions_errors)
    chart_dict = chart.to_dict()

    assert "vconcat" in chart_dict
    assert len(chart_dict["vconcat"]) == 2
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_hchart_predictions_errors_basic(snapshot, snapshot_svg):
    """Horizontal concat of prediction error plots, one per split."""
    chart = hchart(column="split", df=_PRED_DF, func=plot_predictions_errors)
    chart_dict = chart.to_dict()

    assert "hconcat" in chart_dict
    assert len(chart_dict["hconcat"]) == 2
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_hchart_predictions_errors_three_splits(report_theme, snapshot, snapshot_svg):
    """Horizontal concat with train/val/test splits."""
    df = pl.DataFrame({
        "true":  [1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0,  50.0, 60.0, 70.0, 80.0],
        "pred":  [1.2, 1.8, 3.2, 3.9,  11.0, 19.0, 31.0, 39.0,  52.0, 58.0, 72.0, 78.0],
        "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
    })
    chart = hchart(
        column="split", df=df, func=plot_predictions_errors, width=300, height=300,
    )
    chart_dict = chart.to_dict()

    assert "hconcat" in chart_dict
    assert len(chart_dict["hconcat"]) == 3
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


# --- two-dimension: concat + facet ---


def test_hchart_with_row_facet(report_theme, snapshot, snapshot_svg):
    """hchart: columns=split (independent axes), rows=group (shared axes via facet)."""
    chart = hchart(
        column="split",
        row="group",
        df=_PRED_DF,
        func=plot_predictions_errors,
        color_col="model",
        shape_col="model",
    )
    chart_dict = chart.to_dict()

    assert "hconcat" in chart_dict
    assert len(chart_dict["hconcat"]) == 2
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_vchart_with_column_facet(report_theme, snapshot, snapshot_svg):
    """vchart: rows=split (independent axes), columns=group (shared axes via facet)."""
    chart = vchart(
        row="split",
        column="group",
        df=_PRED_DF,
        func=plot_predictions_errors,
        color_col="model",
        shape_col="model",
    )
    chart_dict = chart.to_dict()

    assert "vconcat" in chart_dict
    assert len(chart_dict["vconcat"]) == 2
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_hchart_with_row_facet_color_only(snapshot, snapshot_svg):
    """hchart with row facet and color dimension only."""
    chart = hchart(
        column="split",
        row="group",
        df=_PRED_DF,
        func=plot_predictions_errors,
        color_col="model",
    )
    chart_dict = chart.to_dict()

    assert "hconcat" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg
