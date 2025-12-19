import altair as alt


@alt.theme.register("report_theme", enable=True)
def custom_report_theme() -> alt.theme.ThemeConfig:
    """Define a custom Altair theme for the report."""
    return {
        # Global configuration that applies to all charts
        "config": {
            # 1. Set the view properties (transparent background, fixed width)
            "view": {
                # "width": 800,
                "stroke": "transparent",
                "fill": "transparent",
            },
            "background": "transparent",
            # 2. You can add other global styles here (e.g., axis, title)
            # Example:
            # "title": {
            #     "fontSize": 16,
            #     "anchor": "start"
            # }
        }
    }
