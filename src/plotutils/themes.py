import altair as alt


@alt.theme.register("report_theme", enable=True)
def custom_report_theme() -> alt.theme.ThemeConfig:
    """Define a custom Altair theme for the report."""
    return {
        "config": {
            "view": {
                "stroke": "transparent",
                "fill": "transparent",
            },
            "background": "transparent",
            "title": {
                "fontSize": 18,
                "anchor": "start",
            },
            "axis": {
                "titleFontSize": 14,
                "labelFontSize": 12,
            },
            "legend": {
                "titleFontSize": 14,
                "labelFontSize": 12,
            },
            "header": {
                "titleFontSize": 14,
                "labelFontSize": 13,
            },
        }
    }
