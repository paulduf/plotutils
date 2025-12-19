import altair as alt


def plot_error_bars(df, x, y):
    error_bars = (
        alt.Chart(df)
        .mark_errorbar(extent="ci")
        .encode(x=alt.X(x, scale=alt.Scale(zero=False)), y=alt.Y(y))
    )

    points = (
        alt.Chart(df)
        .mark_point(filled=True, color="black")
        .encode(
            x=alt.X(x, aggregate="mean"),
            y=alt.Y(y),
        )
    )

    return error_bars + points
