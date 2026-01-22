import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.hist import plot_grouped_histogram


def test_plot_grouped_histogram_dict_input(snapshot, snapshot_svg):
    """Test plot_grouped_histogram with dict input (original example)."""

    chart = plot_grouped_histogram(
        data={
            "A": [1.2, 2.3, 1.5, 3.1, 2.8, 1.9, 2.5, 3.3, 2.1, 1.7],
            "B": [0.8, 1.5, 1.2, 2.1, 1.8, 1.3, 1.9, 2.3, 1.6, 1.1],
        },
        n_bins=30,
    )

    # Verify chart is created
    assert chart is not None

    # Verify chart properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 400

    # Snapshot tests
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_grouped_histogram_dataframe_input(snapshot, snapshot_svg):
    """Test plot_grouped_histogram with polars DataFrame input."""
    # Create test data as DataFrame
    df = pl.DataFrame(
        {
            "value": [1.2, 2.3, 1.5, 3.1, 2.8, 0.8, 1.5, 1.2, 2.1, 1.8],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )

    chart = plot_grouped_histogram(
        data=df,
        n_bins=20,
        x_title="Value",
        y_title="Count",
        value_column="value",
        group_column="group",
    )

    # Verify chart is created
    assert chart is not None

    # Verify chart properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 400

    # Snapshot tests
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_grouped_histogram_custom_columns(snapshot, snapshot_svg):
    """Test plot_grouped_histogram with custom column names."""
    df = pl.DataFrame(
        {
            "measurement": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "category": ["X", "X", "X", "Y", "Y", "Y"],
        }
    )

    chart = plot_grouped_histogram(
        data=df,
        n_bins=10,
        value_column="measurement",
        group_column="category",
        width=800,
        height=500,
        opacity=0.8,
    )

    # Verify chart is created
    assert chart is not None

    # Verify custom properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 800
    assert chart_dict["height"] == 500
    assert chart_dict["mark"]["opacity"] == 0.8

    # Snapshot tests
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_grouped_histogram_empty_data():
    """Test that function raises error with empty data."""
    with pytest.raises(ValueError, match="no valid values"):
        plot_grouped_histogram(data={"group1": []}, n_bins=10)
