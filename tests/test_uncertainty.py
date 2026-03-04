import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.uncertainty import (
    plot_confidence_scatter,
    plot_deviations,
    plot_predictions_errors,
)


def test_plot_confidence_scatter_basic(snapshot):
    """Test basic confidence scatter plot with raw data."""
    # Raw data: multiple y values per category
    df = pl.DataFrame({
        "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
        "value": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
                  2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
                  4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
    })

    chart = plot_confidence_scatter(df, x_col="category", y_col="value")
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    # SVG not snapshot-tested: extent="ci" uses bootstrap (non-deterministic)
    assert "<svg" in chart_to_svg(chart)


def test_plot_confidence_scatter_with_title(snapshot):
    """Test confidence scatter plot with custom title."""
    df = pl.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "measurement": [1.0, 1.2, 0.8, 2.0, 2.2, 1.8, 3.0, 3.2, 2.8],
    })

    chart = plot_confidence_scatter(
        df,
        x_col="group",
        y_col="measurement",
        title="Custom Title Test",
    )
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    # SVG not snapshot-tested: extent="ci" uses bootstrap (non-deterministic)
    assert "<svg" in chart_to_svg(chart)


def test_plot_confidence_scatter_custom_styling(snapshot):
    """Test confidence scatter plot with custom styling."""
    df = pl.DataFrame({
        "level": ["X"] * 5 + ["Y"] * 5 + ["Z"] * 5,
        "score": [10.0, 9.5, 10.5, 9.8, 10.2,
                  20.0, 19.5, 20.5, 19.8, 20.2,
                  30.0, 29.5, 30.5, 29.8, 30.2],
    })

    chart = plot_confidence_scatter(
        df,
        x_col="level",
        y_col="score",
        width=800,
        height=500,
        x_title="Category",
        y_title="Measurement",
        point_color="red",
    )
    chart_dict = chart.to_dict()

    assert chart_dict["width"] == 800
    assert chart_dict["height"] == 500
    assert normalize_chart_dict(chart_dict) == snapshot
    # SVG not snapshot-tested: extent="ci" uses bootstrap (non-deterministic)
    assert "<svg" in chart_to_svg(chart)


def test_plot_confidence_scatter_with_identity_line(snapshot):
    """Test confidence scatter plot with identity line."""
    df = pl.DataFrame({
        "x": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "y": [1.0, 0.9, 1.1, 0.95, 1.05,
              2.0, 1.9, 2.1, 1.95, 2.05,
              3.0, 2.9, 3.1, 2.95, 3.05],
    })

    chart = plot_confidence_scatter(df, identity_line=True)
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) == 3  # identity line + error bars + points
    assert normalize_chart_dict(chart_dict) == snapshot
    # SVG not snapshot-tested: extent="ci" uses bootstrap (non-deterministic)
    assert "<svg" in chart_to_svg(chart)


def test_plot_confidence_scatter_stdev_extent(report_theme, snapshot, snapshot_svg):
    """Test confidence scatter plot with stdev extent."""
    df = pl.DataFrame({
        "x": ["A"] * 10 + ["B"] * 10,
        "y": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.05, 0.95,
              2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.1, 1.9, 2.05, 1.95],
    })

    chart = plot_confidence_scatter(df, extent="stdev")
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_confidence_scatter_numeric_x_with_labels(snapshot):
    """Test confidence scatter plot with numeric x-axis and custom labels."""
    df = pl.DataFrame({
        "x": [1.0] * 10 + [2.0] * 10 + [3.0] * 10,
        "y": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
              2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
              4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
    })

    chart = plot_confidence_scatter(
        df,
        x_labels={1.0: "Low", 2.0: "Medium", 3.0: "High"},
    )
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    # Verify x-axis is quantitative with custom labels
    assert chart_dict["layer"][0]["encoding"]["x"]["type"] == "quantitative"
    assert normalize_chart_dict(chart_dict) == snapshot
    # SVG not snapshot-tested: extent="ci" uses bootstrap (non-deterministic)
    assert "<svg" in chart_to_svg(chart)


def test_plot_deviations_basic(report_theme, snapshot, snapshot_svg):
    """Test basic deviations plot: y - mean(y) per group, zero line."""
    df = pl.DataFrame({
        "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
        "value": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
                  2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
                  4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
    })

    chart = plot_deviations(df, x_col="category", y_col="value")
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) == 2  # points + zero line
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_deviations_with_title(snapshot, snapshot_svg):
    """Test deviations plot with custom title."""
    df = pl.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "measurement": [1.0, 1.2, 0.8, 2.0, 2.2, 1.8, 3.0, 3.2, 2.8],
    })

    chart = plot_deviations(
        df,
        x_col="group",
        y_col="measurement",
        title="Deviations Title Test",
    )
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert chart_dict["title"] == "Deviations Title Test"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_deviations_relative(snapshot, snapshot_svg):
    """Test deviations plot with relative=True: (y - mean) / mean."""
    df = pl.DataFrame({
        "x": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "y": [1.0, 0.9, 1.1, 0.95, 1.05,
              2.0, 1.9, 2.1, 1.95, 2.05,
              3.0, 2.9, 3.1, 2.95, 3.05],
    })

    chart = plot_deviations(df, x_col="x", y_col="y", relative=True)
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_deviations_with_levels(report_theme, snapshot, snapshot_svg):
    """Test deviations plot with add_levels: symmetric horizontal lines."""
    df = pl.DataFrame({
        "x": ["A"] * 10 + ["B"] * 10,
        "y": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.05, 0.95,
              2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.1, 1.9, 2.05, 1.95],
    })

    chart = plot_deviations(df, x_col="x", y_col="y", add_levels=[0.1, 0.2])
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) == 3  # points + zero line + level lines
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_deviations_numeric_x_with_labels(snapshot, snapshot_svg):
    """Test deviations plot with numeric x-axis and custom labels."""
    df = pl.DataFrame({
        "x": [1.0] * 10 + [2.0] * 10 + [3.0] * 10,
        "y": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
              2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
              4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05],
    })

    chart = plot_deviations(
        df,
        x_col="x",
        y_col="y",
        x_labels={1.0: "Low", 2.0: "Medium", 3.0: "High"},
    )
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    # Verify x-axis is quantitative with custom labels
    assert chart_dict["layer"][0]["encoding"]["x"]["type"] == "quantitative"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


# --- plot_predictions_errors tests ---


def test_plot_predictions_errors_basic(report_theme, snapshot, snapshot_svg):
    """Test basic prediction error plot with identity line."""
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8, 9.1, 10.3],
    })

    chart = plot_predictions_errors(df)
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) == 2  # identity line + points
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_custom_columns(snapshot, snapshot_svg):
    """Test prediction error plot with custom column names and title."""
    df = pl.DataFrame({
        "actual": [1.0, 2.0, 3.0, 4.0, 5.0],
        "forecast": [1.2, 1.8, 3.1, 4.3, 4.8],
    })

    chart = plot_predictions_errors(
        df,
        true_col="actual",
        pred_col="forecast",
        title="Forecast Accuracy",
        x_title="Actual Value",
        y_title="Forecast Value",
    )
    chart_dict = chart.to_dict()

    assert chart_dict["title"] == "Forecast Accuracy"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_with_color(snapshot, snapshot_svg):
    """Test prediction error plot with color column."""
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9],
        "model": ["A", "A", "A", "B", "B", "B"],
    })

    chart = plot_predictions_errors(df, color_col="model")
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    points_layer = chart_dict["layer"][1]
    assert "color" in points_layer["encoding"]
    assert points_layer["encoding"]["color"]["field"] == "model"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_with_shape(snapshot, snapshot_svg):
    """Test prediction error plot with shape column."""
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9],
        "dataset": ["train", "train", "train", "test", "test", "test"],
    })

    chart = plot_predictions_errors(df, shape_col="dataset")
    chart_dict = chart.to_dict()

    points_layer = chart_dict["layer"][1]
    assert "shape" in points_layer["encoding"]
    assert points_layer["encoding"]["shape"]["field"] == "dataset"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_color_and_shape(report_theme, snapshot, snapshot_svg):
    """Test prediction error plot with both color and shape columns."""
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8],
        "model": ["A", "A", "B", "B", "A", "A", "B", "B"],
        "split": ["train", "test", "train", "test", "train", "test", "train", "test"],
    })

    chart = plot_predictions_errors(df, color_col="model", shape_col="split")
    chart_dict = chart.to_dict()

    points_layer = chart_dict["layer"][1]
    assert "color" in points_layer["encoding"]
    assert "shape" in points_layer["encoding"]
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_shared_scale(snapshot, snapshot_svg):
    """Test that both axes share the same scale domain."""
    # true range [1, 5], pred range [2, 10] → shared domain with 2% padding
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pred": [2.0, 4.0, 6.0, 8.0, 10.0],
    })

    chart = plot_predictions_errors(df)
    chart_dict = chart.to_dict()

    points_layer = chart_dict["layer"][1]
    x_domain = points_layer["encoding"]["x"]["scale"]["domain"]
    y_domain = points_layer["encoding"]["y"]["scale"]["domain"]
    # raw range [1, 10], pad = 9 * 0.02 = 0.18
    assert x_domain == pytest.approx([0.82, 10.18])
    assert y_domain == pytest.approx([0.82, 10.18])
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_predictions_errors_custom_styling(snapshot, snapshot_svg):
    """Test prediction error plot with custom point color, size, and dimensions."""
    df = pl.DataFrame({
        "true": [1.0, 2.0, 3.0, 4.0, 5.0],
        "pred": [1.1, 1.8, 3.2, 3.9, 5.1],
    })

    chart = plot_predictions_errors(
        df,
        width=800,
        height=800,
        point_color="red",
        point_size=100,
        point_opacity=0.5,
    )
    chart_dict = chart.to_dict()

    assert chart_dict["width"] == 800
    assert chart_dict["height"] == 800
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg
