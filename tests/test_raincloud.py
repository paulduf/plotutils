import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.raincloud import plot_raincloud


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "value": [
                1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.4, 1.0, 0.7, 1.5,
                1.1, 0.95, 1.25, 1.05, 0.85, 1.15, 1.35, 0.75, 1.45, 1.02,
                2.0, 2.3, 1.8, 2.1, 2.5, 1.7, 2.4, 2.0, 1.6, 2.6,
                2.2, 1.9, 2.35, 2.15, 1.85, 2.05, 2.45, 1.75, 2.55, 2.02,
            ],
        }
    )


def test_plot_raincloud_basic(sample_df, snapshot, snapshot_svg):
    chart = plot_raincloud(sample_df, x_col="group", y_col="value")
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_raincloud_uniform_jitter(sample_df, snapshot, snapshot_svg):
    chart = plot_raincloud(
        sample_df, x_col="group", y_col="value", jitter="uniform"
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_raincloud_custom_options(sample_df, snapshot, snapshot_svg):
    chart = plot_raincloud(
        sample_df,
        x_col="group",
        y_col="value",
        title="My Raincloud",
        width=200,
        height=400,
        y_title="Measurement",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_raincloud_deterministic_jitter(sample_df):
    """Same seed produces identical charts."""
    chart1 = plot_raincloud(
        sample_df, x_col="group", y_col="value", jitter_seed=42
    )
    chart2 = plot_raincloud(
        sample_df, x_col="group", y_col="value", jitter_seed=42
    )
    assert chart1.to_dict() == chart2.to_dict()
