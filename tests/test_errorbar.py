import polars as pl
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.errorbar import plot_confidence_scatter


def test_plot_confidence_scatter_basic(snapshot, snapshot_svg):
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
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_confidence_scatter_with_title(snapshot, snapshot_svg):
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
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_confidence_scatter_custom_styling(snapshot, snapshot_svg):
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
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_confidence_scatter_with_identity_line(snapshot, snapshot_svg):
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
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_confidence_scatter_stdev_extent(snapshot, snapshot_svg):
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


def test_plot_confidence_scatter_numeric_x_with_labels(snapshot, snapshot_svg):
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
    assert chart_to_svg(chart) == snapshot_svg
