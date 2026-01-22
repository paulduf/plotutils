import polars as pl
import pytest
from plotutils.hist import plot_grouped_histogram


def _normalize_chart_dict(chart_dict):
    """Normalize chart dict by removing non-deterministic data name hash and sorting data."""
    if "data" in chart_dict and "name" in chart_dict["data"]:
        # Replace the hash with a stable name
        old_name = chart_dict["data"]["name"]
        chart_dict["data"]["name"] = "data-normalized"

        # Update the datasets key to match and sort the data for deterministic ordering
        if "datasets" in chart_dict and old_name in chart_dict["datasets"]:
            dataset = chart_dict["datasets"].pop(old_name)
            # Sort by group then bin_start to ensure deterministic order
            if dataset and isinstance(dataset, list) and "group" in dataset[0]:
                dataset = sorted(dataset, key=lambda x: (x.get("group", ""), x.get("bin_start", 0)))
            chart_dict["datasets"]["data-normalized"] = dataset

    return chart_dict


def test_plot_grouped_histogram_dict_input(snapshot):
    """Test plot_grouped_histogram with dict input (original example)."""
    # Sample data from the original example
    ice_normalized = [1.2, 2.3, 1.5, 3.1, 2.8, 1.9, 2.5, 3.3, 2.1, 1.7]
    raw_measurements = [0.8, 1.5, 1.2, 2.1, 1.8, 1.3, 1.9, 2.3, 1.6, 1.1]

    chart = plot_grouped_histogram(
        data={
            "ICE uncorrelated": ice_normalized,
            "Distribution of simulated measurements (N=10)": raw_measurements,
        },
        n_bins=30,
        x_title="Q (cp/PCR)",
        y_title="Counts",
    )

    # Verify chart is created
    assert chart is not None

    # Verify chart properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 400

    # Snapshot test with normalized data names
    assert _normalize_chart_dict(chart_dict) == snapshot


def test_plot_grouped_histogram_dataframe_input(snapshot):
    """Test plot_grouped_histogram with polars DataFrame input."""
    # Create test data as DataFrame
    df = pl.DataFrame({
        "value": [1.2, 2.3, 1.5, 3.1, 2.8, 0.8, 1.5, 1.2, 2.1, 1.8],
        "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    })

    chart = plot_grouped_histogram(
        data=df,
        n_bins=20,
        x_title="Value",
        y_title="Count",
        value_column="value",
        group_column="group"
    )

    # Verify chart is created
    assert chart is not None

    # Verify chart properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 600
    assert chart_dict["height"] == 400

    # Snapshot test with normalized data names
    assert _normalize_chart_dict(chart_dict) == snapshot


def test_plot_grouped_histogram_custom_columns(snapshot):
    """Test plot_grouped_histogram with custom column names."""
    df = pl.DataFrame({
        "measurement": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "category": ["X", "X", "X", "Y", "Y", "Y"]
    })

    chart = plot_grouped_histogram(
        data=df,
        n_bins=10,
        value_column="measurement",
        group_column="category",
        width=800,
        height=500,
        opacity=0.8
    )

    # Verify chart is created
    assert chart is not None

    # Verify custom properties
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 800
    assert chart_dict["height"] == 500
    assert chart_dict["mark"]["opacity"] == 0.8

    # Snapshot test with normalized data names
    assert _normalize_chart_dict(chart_dict) == snapshot


def test_plot_grouped_histogram_empty_data():
    """Test that function raises error with empty data."""
    with pytest.raises(ValueError, match="no valid values"):
        plot_grouped_histogram(
            data={"group1": []},
            n_bins=10
        )
