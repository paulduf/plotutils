"""Generate interactive HTML chart snippets for documentation pages.

Run with:  uv run python docs/gen_charts.py

Each Altair chart is saved as a small HTML snippet (a <div> + <script>
calling vegaEmbed) that can be included directly in Markdown pages via
pymdownx.snippets.  The Vega / Vega-Lite / Vega-Embed libraries are
loaded once at the page level through ``extra_javascript`` in mkdocs.yml.
"""

import json

import polars as pl

from plotutils.auc import plot_roc_curve
from plotutils.forest import plot_forest
from plotutils.raincloud import plot_raincloud
from plotutils.parallel import plot_parallel_coordinates
from plotutils.boxplot import plot_bivariate_boxes, plot_bivariate_strip
from plotutils.concat import hchart, vchart
from plotutils.hist import plot_grouped_histogram
from plotutils.uncertainty import (
    plot_confidence_scatter,
    plot_deviations,
    plot_predictions_errors,
)
from plotutils.upset import plot_upset

OUT = "docs/plots"


def _save(chart, name: str) -> None:
    """Write an embeddable Vega-Lite snippet (div + script).

    The Vega libraries are loaded at the bottom of the page by MkDocs, so
    the inline ``<script>`` must wait until ``vegaEmbed`` is available.
    We register a tiny ``DOMContentLoaded`` listener that polls for it.
    """
    stem = name.removesuffix(".html").replace(".", "_").replace("-", "_")
    div_id = f"vis-{stem}"
    spec_json = json.dumps(json.loads(chart.to_json()))
    snippet = (
        f'<div id="{div_id}"></div>\n'
        f"<script>\n"
        f"  (function() {{\n"
        f"    function render() {{\n"
        f'      vegaEmbed("#{div_id}", {spec_json}, {{"actions": false}});\n'
        f"    }}\n"
        f"    if (typeof vegaEmbed !== 'undefined') {{ render(); }}\n"
        f"    else {{ document.addEventListener('DOMContentLoaded', function check() {{\n"
        f"      if (typeof vegaEmbed !== 'undefined') render();\n"
        f"      else setTimeout(check, 50);\n"
        f"    }}); }}\n"
        f"  }})();\n"
        f"</script>\n"
    )
    path = f"{OUT}/{name}"
    with open(path, "w") as f:
        f.write(snippet)
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


def gen_parallel_charts() -> None:
    df = pl.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3, 6.5, 5.8],
            "sepal_width": [3.5, 3.0, 3.2, 3.3, 2.8, 2.7],
            "petal_length": [1.4, 1.4, 4.7, 4.4, 4.6, 5.1],
            "petal_width": [0.2, 0.2, 1.4, 1.3, 1.5, 1.9],
            "species": [
                "setosa", "setosa", "versicolor",
                "versicolor", "virginica", "virginica",
            ],
        }
    )
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    _save(
        plot_parallel_coordinates(df, columns=cols, color_col="species"),
        "parallel_basic.html",
    )
    _save(
        plot_parallel_coordinates(
            df, columns=cols, color_col="species", normalize=True
        ),
        "parallel_normalized.html",
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


def gen_auc_reports() -> None:
    from plotutils.auc import AUCReport
    from plotutils.datasets import load_binary_diabetes, load_synthetic

    # ── Synthetic ──────────────────────────────────────────────────────
    ds = load_synthetic()
    AUCReport(
        ds.df, variables=ds.variables, outcomes=ds.outcomes, id_col="patient_id"
    ).to_html(path=f"{OUT}/auc_report_synthetic.html")
    print("  auc_report_synthetic.html")

    # ── Synthetic with anti-correlated variable ────────────────────────
    ds_rev = load_synthetic(anti_correlated=True)
    AUCReport(
        ds_rev.df, variables=ds_rev.variables, outcomes=ds_rev.outcomes,
        id_col="patient_id", auto_reverse=True,
    ).to_html(path=f"{OUT}/auc_report_reversed.html")
    print("  auc_report_reversed.html")

    # ── Synthetic with anti-correlated outcomes ───────────────────────
    ds_anti_out = load_synthetic(anti_correlated_outcomes=True)
    AUCReport(
        ds_anti_out.df, variables=ds_anti_out.variables,
        outcomes=ds_anti_out.outcomes,
        id_col="patient_id", auto_reverse=True,
    ).to_html(path=f"{OUT}/auc_report_anti_outcomes.html")
    print("  auc_report_anti_outcomes.html")

    # ── Synthetic with missing data ────────────────────────────────────
    ds_miss = load_synthetic(missing=True)
    AUCReport(
        ds_miss.df, variables=ds_miss.variables, outcomes=ds_miss.outcomes,
        id_col="patient_id",
    ).to_html(path=f"{OUT}/auc_report_missing.html")
    print("  auc_report_missing.html")

    # ── Diabetes ───────────────────────────────────────────────────────
    ds_diab = load_binary_diabetes()
    AUCReport(
        ds_diab.df, variables=ds_diab.variables, outcomes=ds_diab.outcomes,
        id_col="patient_id",
    ).to_html(path=f"{OUT}/auc_report_diabetes.html")
    print("  auc_report_diabetes.html")


def gen_forest_basic() -> None:
    df = pl.DataFrame(
        {
            "subgroup": ["Overall", "Age < 65", "Age ≥ 65", "Male", "Female"],
            "hr": [0.78, 0.72, 0.85, 0.80, 0.75],
            "low": [0.62, 0.55, 0.65, 0.61, 0.57],
            "high": [0.98, 0.94, 1.11, 1.05, 0.99],
        }
    )
    chart = plot_forest(
        df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        x_title="Hazard Ratio",
    )
    _save(chart, "forest_basic.html")


def gen_forest_effect() -> None:
    df = pl.DataFrame(
        {
            "subgroup": [
                "Overall", "Age < 65", "Age ≥ 65",
                "Male", "Female", "High risk", "Low risk",
            ],
            "hr": [0.78, 0.72, 0.85, 0.80, 0.75, 0.68, 0.91],
            "low": [0.62, 0.55, 0.65, 0.61, 0.57, 0.50, 0.72],
            "high": [0.98, 0.94, 1.11, 1.05, 0.99, 0.92, 1.15],
            "population": [
                "All", "Young", "Older",
                "Male", "Female", "High risk", "Low risk",
            ],
        }
    )
    chart = plot_forest(
        df,
        center_col="hr",
        low_col="low",
        high_col="high",
        label_col="subgroup",
        null_value=1.0,
        min_effect=1.25,
        color_col="population",
        x_title="Hazard Ratio",
        title="Treatment effect by subgroup",
    )
    _save(chart, "forest_effect.html")


def gen_raincloud() -> None:
    df = pl.DataFrame(
        {
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            "value": [
                1.0 + i * 0.05 for i in range(30)
            ] + [
                2.0 + i * 0.06 for i in range(30)
            ] + [
                1.5 + i * 0.04 for i in range(30)
            ],
        }
    )
    chart = plot_raincloud(df, x_col="group", y_col="value")
    _save(chart, "raincloud.html")


def gen_upset_charts() -> None:
    df = pl.DataFrame(
        {
            "Drama": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
            "Comedy": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
            "Action": [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "Sci-Fi": [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )
    _save(plot_upset(df, title="Movie genre overlaps"), "upset_basic.html")
    _save(
        plot_upset(
            df,
            sort_by=["degree", "frequency"],
            min_degree=1,
            n_intersections=10,
            title="Degree ≥ 1, sorted by degree then frequency",
        ),
        "upset_filtered.html",
    )
    _save(
        plot_upset(df, show_set_sizes=False, title="No set-size bars"),
        "upset_no_sizes.html",
    )


if __name__ == "__main__":
    print("Generating interactive charts...")
    gen_upset_charts()
    gen_grouped_histogram()
    gen_confidence_scatter()
    gen_deviations_basic()
    gen_deviations_levels()
    gen_predictions_errors_basic()
    gen_predictions_errors_color()
    gen_concat_charts()
    gen_boxplot_charts()
    gen_parallel_charts()
    gen_auc_chart()
    gen_auc_reports()
    gen_forest_basic()
    gen_forest_effect()
    gen_raincloud()
    print("Done.")
