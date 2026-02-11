import altair as alt
import importlib

import polars as pl


def test_report_theme_registration(snapshot):
    # Import the theme module (registers and enables the theme)
    importlib.import_module("plotutils.themes")
    alt.theme.enable("report_theme")

    # Check that the theme is registered and enabled
    assert "report_theme" in alt.theme.names()
    assert alt.theme.active == "report_theme"

    # Check that the theme config is applied to a chart
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    chart = alt.Chart(df).mark_line()
    config = chart.to_dict().get("config", {})
    assert config.get("view", {}).get("stroke") == "transparent"
    assert config.get("view", {}).get("fill") == "transparent"
    assert config.get("background") == "transparent"
    assert config == snapshot
