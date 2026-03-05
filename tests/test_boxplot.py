import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from plotutils.boxplot import plot_bivariate_boxes, plot_bivariate_strip


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": [0.8, 0.6, 0.9, 0.7, 0.5, 0.3, 0.4, 0.2, 0.85, 0.65, 0.35, 0.25],
            "outcome_a": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            "outcome_b": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "patient_id": [f"P{i:03d}" for i in range(12)],
        }
    )


@pytest.fixture
def missing_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "outcome_a": [1, 0, 1],
            "outcome_b": [0, 1, 0],
            "patient_id": ["M001", "M002", "M003"],
        }
    )


# ── plot_bivariate_boxes ──────────────────────────────────────────────


def test_plot_bivariate_boxes_basic(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_boxes(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_boxes_custom_titles(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_boxes(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        title="Custom Title",
        y_title="Score Value",
        x_title="Outcome A",
        color_title="Outcome B",
        width=500,
        height=400,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 500
    assert chart_dict["height"] == 400
    assert chart_dict["title"] == "Custom Title"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_boxes_with_id(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_boxes(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        id_col="patient_id",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_boxes_with_missing(sample_df, missing_df, snapshot, snapshot_svg):
    chart = plot_bivariate_boxes(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        missing_score_df=missing_df,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_boxes_with_missing_and_id(sample_df, missing_df, snapshot, snapshot_svg):
    chart = plot_bivariate_boxes(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        id_col="patient_id",
        missing_score_df=missing_df,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


# ── plot_bivariate_strip ──────────────────────────────────────────────


def test_plot_bivariate_strip_basic(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_strip_custom_titles(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        title="Strip Plot",
        y_title="Score Value",
        x_title="Outcome A",
        color_title="Outcome B",
        width=500,
        height=400,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert chart_dict["width"] == 500
    assert chart_dict["height"] == 400
    assert chart_dict["title"] == "Strip Plot"
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_strip_with_id(sample_df, snapshot, snapshot_svg):
    chart = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        id_col="patient_id",
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_strip_with_missing(sample_df, missing_df, snapshot, snapshot_svg):
    chart = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        missing_score_df=missing_df,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_strip_with_missing_and_id(sample_df, missing_df, snapshot, snapshot_svg):
    chart = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        id_col="patient_id",
        missing_score_df=missing_df,
    )
    assert chart is not None
    chart_dict = chart.to_dict()
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_bivariate_strip_deterministic_jitter(sample_df):
    """Same seed produces identical charts."""
    chart1 = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        jitter_seed=42,
    )
    chart2 = plot_bivariate_strip(
        sample_df,
        score_col="score",
        label_x_col="outcome_a",
        label_y_col="outcome_b",
        jitter_seed=42,
    )
    assert chart1.to_dict() == chart2.to_dict()
