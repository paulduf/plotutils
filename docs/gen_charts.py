"""Generate interactive HTML charts for documentation pages.

Run with:  uv run python docs/gen_charts.py
"""

import polars as pl

from plotutils.auc import plot_roc_curve
from plotutils.boxplot import plot_bivariate_boxes, plot_bivariate_strip
from plotutils.concat import hchart, vchart
from plotutils.hist import plot_grouped_histogram
from plotutils.uncertainty import (
    plot_confidence_scatter,
    plot_deviations,
    plot_predictions_errors,
)

OUT = "docs/plots"


def _save(chart, name: str) -> None:
    path = f"{OUT}/{name}"
    chart.save(path)
    print(f"  {name}")


def gen_grouped_histogram() -> None:
    chart = plot_grouped_histogram(
        data={
            "Group A": [1.2, 2.3, 1.5, 3.1, 2.8],
            "Group B": [0.8, 1.5, 1.2, 2.1, 1.8],
        },
        n_bins=30,
        x_title="Value",
        y_title="Counts",
    )
    _save(chart, "grouped_histogram.html")


def gen_confidence_scatter() -> None:
    df = pl.DataFrame(
        {
            "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
            "value": [
                1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
                2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
                4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05,
            ],
        }
    )
    chart = plot_confidence_scatter(
        df, x_col="category", y_col="value", extent="stdev"
    )
    _save(chart, "confidence_scatter.html")


def gen_deviations_basic() -> None:
    df = pl.DataFrame(
        {
            "category": ["Low"] * 10 + ["Medium"] * 10 + ["High"] * 10,
            "value": [
                1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95, 1.05,
                2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.2, 2.8, 2.45, 2.55,
                4.0, 3.8, 4.2, 3.9, 4.1, 4.0, 3.7, 4.3, 3.95, 4.05,
            ],
        }
    )
    chart = plot_deviations(df, x_col="category", y_col="value")
    _save(chart, "deviations_basic.html")


def gen_deviations_levels() -> None:
    df = pl.DataFrame(
        {
            "x": ["A"] * 10 + ["B"] * 10,
            "y": [
                1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.05, 0.95,
                2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.1, 1.9, 2.05, 1.95,
            ],
        }
    )
    chart = plot_deviations(df, x_col="x", y_col="y", add_levels=[0.1, 0.2])
    _save(chart, "deviations_levels.html")


def gen_predictions_errors_basic() -> None:
    df = pl.DataFrame(
        {
            "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8, 9.1, 10.3],
        }
    )
    chart = plot_predictions_errors(df)
    _save(chart, "predictions_errors_basic.html")


def gen_predictions_errors_color() -> None:
    df = pl.DataFrame(
        {
            "true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "pred": [1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8],
            "model": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "split": [
                "train", "test", "train", "test",
                "train", "test", "train", "test",
            ],
        }
    )
    chart = plot_predictions_errors(df, color_col="model", shape_col="split")
    _save(chart, "predictions_errors_color.html")


def gen_concat_charts() -> None:
    df = pl.DataFrame(
        {
            "true": [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                50.0, 60.0, 70.0, 75.0, 80.0, 90.0,
                50.0, 60.0, 70.0, 75.0, 80.0, 90.0,
            ],
            "pred": [
                1.1, 2.3, 2.8, 4.2, 5.1, 5.8,
                1.4, 1.7, 3.5, 3.6, 5.3, 5.5,
                52.0, 58.0, 72.0, 73.0, 82.0, 91.0,
                55.0, 55.0, 75.0, 70.0, 84.0, 93.0,
            ],
            "split": ["train"] * 12 + ["test"] * 12,
            "group": (["A"] * 6 + ["B"] * 6) * 2,
            "model": (["linear"] * 3 + ["tree"] * 3) * 4,
        }
    )

    _save(
        hchart(
            column="split",
            row="group",
            df=df,
            func=plot_predictions_errors,
            color_col="model",
            shape_col="model",
        ),
        "hchart_row_facet.html",
    )

    _save(
        vchart(
            row="split",
            column="group",
            df=df,
            func=plot_predictions_errors,
            color_col="model",
            shape_col="model",
        ),
        "vchart_column_facet.html",
    )

    df_split = pl.DataFrame(
        {
            "true": [
                1.0, 2.0, 3.0, 4.0,
                10.0, 20.0, 30.0, 40.0,
                50.0, 60.0, 70.0, 80.0,
            ],
            "pred": [
                1.2, 1.8, 3.2, 3.9,
                11.0, 19.0, 31.0, 39.0,
                52.0, 58.0, 72.0, 78.0,
            ],
            "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
        }
    )
    _save(
        hchart(
            column="split",
            df=df_split,
            func=plot_predictions_errors,
            width=300,
            height=300,
        ),
        "hchart_three_splits.html",
    )


def gen_boxplot_charts() -> None:
    df = pl.DataFrame(
        {
            "score": [
                0.8, 0.6, 0.9, 0.7, 0.5, 0.3, 0.4, 0.2,
                0.85, 0.65, 0.35, 0.25,
            ],
            "outcome_a": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            "outcome_b": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "patient_id": [f"P{i:03d}" for i in range(12)],
        }
    )
    _save(
        plot_bivariate_boxes(
            df,
            score_col="score",
            label_x_col="outcome_a",
            label_y_col="outcome_b",
        ),
        "boxplot_boxes.html",
    )
    _save(
        plot_bivariate_strip(
            df,
            score_col="score",
            label_x_col="outcome_a",
            label_y_col="outcome_b",
            id_col="patient_id",
        ),
        "boxplot_strip.html",
    )


def gen_auc_chart() -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    labels = rng.integers(0, 2, size=n)
    scores = rng.normal(loc=labels * 1.2, scale=1.0)
    scores = np.clip(
        (scores - scores.min()) / (scores.max() - scores.min()), 0, 1
    )

    df = pl.DataFrame({"score": scores.tolist(), "label": labels.tolist()})
    chart = plot_roc_curve(
        df,
        score_col="score",
        label_col="label",
        specificity_levels=[0.95, 0.90, 0.80],
        width=480,
        height=440,
    )
    _save(chart, "auc_chart.html")


if __name__ == "__main__":
    print("Generating interactive charts...")
    gen_grouped_histogram()
    gen_confidence_scatter()
    gen_deviations_basic()
    gen_deviations_levels()
    gen_predictions_errors_basic()
    gen_predictions_errors_color()
    gen_concat_charts()
    gen_boxplot_charts()
    gen_auc_chart()
    print("Done.")
