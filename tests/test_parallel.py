import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.parallel import plot_parallel_coordinates


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "c": [2.0, 4.0, 1.0, 3.0, 5.0],
            "group": ["X", "X", "Y", "Y", "X"],
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )


def test_plot_parallel_basic(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(sample_df, columns=["a", "b", "c"])
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 400
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_with_color(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(
        sample_df, columns=["a", "b", "c"], color_col="group"
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_normalized(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(
        sample_df, columns=["a", "b", "c"], normalize=True
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_with_transforms(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(
        sample_df,
        columns=["a", "b", "c"],
        normalize=True,
        transforms=["linear", "log", "linear"],
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_with_id(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(
        sample_df,
        columns=["a", "b", "c"],
        color_col="group",
        id_col="sample_id",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_custom_title_and_size(sample_df, snapshot, snapshot_svg):
    chart = plot_parallel_coordinates(
        sample_df,
        columns=["a", "b", "c"],
        title="My Parallel Plot",
        width=800,
        height=500,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 800
    assert chart_dict["height"] == 500
    assert chart_dict["title"] == "My Parallel Plot"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_parallel_transforms_length_mismatch(sample_df):
    with pytest.raises(ValueError, match="transforms length"):
        plot_parallel_coordinates(
            sample_df,
            columns=["a", "b", "c"],
            normalize=True,
            transforms=["linear", "log"],
        )
