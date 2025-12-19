import altair as alt
import importlib
import pytest

import polars as pl


def test_report_theme_registration(snapshot):
    # Remove the theme if already registered (for test idempotency)
    if "report_theme" in alt.theme.names():
        alt.theme.enable("default")
        alt.theme._names.remove("report_theme")

    # Import the theme module (should register and enable the theme)
    importlib.import_module("plotutils.themes")

    # Check that the theme is registered and enabled
    assert "report_theme" in alt.theme.names(), (
        "Theme 'report_theme' should be registered."
    )
    assert alt.theme.active == "report_theme", "Theme 'report_theme' should be active."

    # Check that the theme config is applied to a chart
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = alt.Chart(df).mark_line()
    config = chart.to_dict().get("config", {})
    assert config.get("view", {}).get("stroke") == "transparent"
    assert config.get("view", {}).get("fill") == "transparent"
    assert config.get("background") == "transparent"
    assert config == snapshot
