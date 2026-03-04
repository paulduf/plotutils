"""
Linked chart demo: hover a point on the scatter plot to update the line chart.

The left chart shows mean values per category.
The right chart shows the monthly time series for the hovered category.
"""

import altair as alt
import numpy as np
import polars as pl

OUTPUT = "demos/_output/linked-mouseover.html"

rng = np.random.default_rng(42)

categories = ["Alpha", "Beta", "Gamma", "Delta"]
months = pl.date_range(
    pl.date(2024, 1, 1), pl.date(2024, 12, 1), interval="1mo", eager=True
)

# One row per (category, month)
df = pl.DataFrame(
    [
        {
            "category": cat,
            "date": date,
            "value": float(rng.normal(loc=i * 10, scale=3)),
        }
        for i, cat in enumerate(categories)
        for date in months
    ]
)

summary = df.group_by("category").agg(pl.col("value").mean())

# --- Selection ----------------------------------------------------------
hover = alt.selection_point(fields=["category"], on="mouseover", empty="none")

# --- Left: scatter of mean values per category --------------------------
scatter = (
    alt.Chart(summary)
    .mark_circle(size=120)
    .encode(
        x=alt.X("category:N", title="Category"),
        y=alt.Y("value:Q", title="Mean value", scale=alt.Scale(zero=False)),
        color=alt.condition(hover, "category:N", alt.value("lightgray")),
        tooltip=["category:N", alt.Tooltip("value:Q", format=".2f")],
    )
    .add_params(hover)
    .properties(title="Hover a category", width=250, height=300)
)

# --- Right: time series for the hovered category ------------------------
line = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Value"),
        color="category:N",
        tooltip=["category:N", "date:T", alt.Tooltip("value:Q", format=".2f")],
    )
    .transform_filter(hover)
    .properties(title="Time series (hovered category)", width=400, height=300)
)

chart = scatter | line

chart.save(OUTPUT)
print(f"Saved to {OUTPUT}")
