import altair.datasets as datasets
from plotutils import plot_error_bars


def test_plot_error_bars_chart_dict(snapshot):
    # Use the cars dataset with polars engine
    df = datasets.data.cars(engine="polars")
    chart = plot_error_bars(df, x="Cylinders", y="Weight_in_lbs")
    chart_dict = chart.to_dict()
    # Syrupy: assert chart_dict matches the snapshot
    assert chart_dict == snapshot
