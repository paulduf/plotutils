import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.forest import plot_forest


@pytest.fixture
def hr_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subgroup": ["Overall", "Age < 65", "Age ≥ 65", "Male", "Female"],
            "hr": [0.78, 0.72, 0.85, 0.80, 0.75],
            "low": [0.62, 0.55, 0.65, 0.61, 0.57],
            "high": [0.98, 0.94, 1.11, 1.05, 0.99],
            "category": ["All", "Young", "Old", "Male", "Female"],
        }
    )


def test_plot_forest_basic(report_theme, snapshot, snapshot_svg, hr_df):
    """Basic forest plot: null rule + CI bars + center points = 3 layers."""
    chart = plot_forest(
        hr_df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        x_title="Hazard Ratio",
    )
    chart_dict = chart.to_dict()
    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) == 3  # null rule, CI bars, center points
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_forest_with_min_effect(report_theme, snapshot, snapshot_svg, hr_df):
    """min_effect adds two dashed rules → 5 layers total."""
    chart = plot_forest(
        hr_df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        min_effect=1.25,
    )
    chart_dict = chart.to_dict()
    assert len(chart_dict["layer"]) == 5  # null + 2 me rules + CI bars + points
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_forest_with_color(report_theme, snapshot, snapshot_svg, hr_df):
    """color_col adds a color encoding on bars and points."""
    chart = plot_forest(
        hr_df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        color_col="category",
    )
    chart_dict = chart.to_dict()
    # Verify color encoding is present on the last layer (center points)
    point_layer = chart_dict["layer"][-1]
    assert "color" in point_layer["encoding"]
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_forest_custom_dims(snapshot, hr_df):
    """Explicit width and height are reflected in chart properties."""
    chart = plot_forest(
        hr_df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        width=600,
        height=200,
        title="Custom dims",
    )
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 200
    assert normalize_chart_dict(chart_dict) == snapshot


def test_plot_forest_sort_col(report_theme, snapshot, snapshot_svg, hr_df):
    """sort_col produces an EncodingSortField on the y encoding."""
    chart = plot_forest(
        hr_df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        sort_col="hr",
        ascending=False,
    )
    chart_dict = chart.to_dict()
    ci_layer = chart_dict["layer"][-2]
    y_sort = ci_layer["encoding"]["y"]["sort"]
    assert y_sort["field"] == "hr"
    assert y_sort["order"] == "descending"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_forest_preserves_order(report_theme, snapshot, snapshot_svg):
    """Row order in the DataFrame is preserved (sort=None on y-axis)."""
    df = pl.DataFrame(
        {
            "label": ["Z", "A", "M", "B", "Y"],
            "est": [1.2, 0.9, 1.05, 0.8, 1.3],
            "lo": [0.9, 0.7, 0.85, 0.6, 1.0],
            "hi": [1.6, 1.1, 1.3, 1.0, 1.7],
        }
    )
    chart = plot_forest(df, center_col="est", low_col="lo", high_col="hi", label_col="label")
    chart_dict = chart.to_dict()
    # y encoding on CI bars layer (index -2) should have sort=None
    ci_layer = chart_dict["layer"][-2]
    assert ci_layer["encoding"]["y"].get("sort") is None
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg
