"""ROC curve with optional specificity-level annotations."""

import polars as pl
import altair as alt


def _compute_roc(
    df: pl.DataFrame,
    score_col: str,
    label_col: str,
    reverse_score: bool = False,
) -> pl.DataFrame:
    """Return a DataFrame with columns (threshold, sensitivity, specificity).

    Includes boundary points at (sensitivity=0, specificity=1) and
    (sensitivity=1, specificity=0) with null threshold.

    Ties (multiple samples sharing the same score) are handled correctly:
    all tied samples are grouped into one threshold step before the cumulative
    TP/FP counts are computed.
    """
    n_pos = int((df[label_col] == 1).sum())
    n_neg = int((df[label_col] == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes (0 and 1) must be present in label_col.")

    curve = (
        df.select([score_col, label_col])
        .group_by(score_col)
        .agg(
            (pl.col(label_col) == 1).sum().alias("tp_step"),
            (pl.col(label_col) == 0).sum().alias("fp_step"),
        )
        .sort(score_col, descending=not reverse_score)
        .with_columns(
            pl.col("tp_step").cum_sum().alias("tp"),
            pl.col("fp_step").cum_sum().alias("fp"),
        )
        .select(
            pl.col(score_col).cast(pl.Float64).alias("threshold"),
            (pl.col("tp") / n_pos).alias("sensitivity"),
            ((n_neg - pl.col("fp")) / n_neg).alias("specificity"),
        )
    )

    boundary = pl.DataFrame(
        {
            "threshold": pl.Series([None, None], dtype=pl.Float64),
            "sensitivity": [0.0, 1.0],
            "specificity": [1.0, 0.0],
        }
    )

    return pl.concat([boundary.head(1), curve, boundary.tail(1)])


def _compute_auc(roc_df: pl.DataFrame) -> float:
    """Trapezoidal AUC under the specificity-sensitivity curve.

    Equivalent to the standard AUC of the ROC (see note in the docstring
    of :func:`plot_roc_curve`).
    """
    return float(
        roc_df.sort("sensitivity")
        .with_columns(
            [
                pl.col("sensitivity").diff().alias("dx"),
                ((pl.col("specificity") + pl.col("specificity").shift(1)) / 2).alias(
                    "avg_spec"
                ),
            ]
        )
        .filter(pl.col("dx").is_not_null())
        .select((pl.col("dx") * pl.col("avg_spec")).sum())
        .item()
    )


def plot_roc_curve(
    df: pl.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    specificity_levels: list[float] | None = None,
    title: str = "",
    width: int = 400,
    height: int = 400,
    curve_color: str = "steelblue",
    **kwargs,
) -> alt.LayerChart:
    """Plot a ROC curve with sensitivity on the x-axis and specificity on the y-axis.

    The area under this curve is mathematically equal to the standard AUC
    (area under the FPR / TPR ROC curve).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing a score column and a binary label column.
    score_col : str
        Column with classifier scores (higher → more likely positive).
    label_col : str
        Column with binary ground-truth labels (0 = negative, 1 = positive).
    specificity_levels : list[float] or None
        Target specificity values to annotate on the curve.  For each level a
        dashed horizontal line is drawn from the y-axis to the curve, then a
        dashed vertical line goes down to the x-axis.  The intersection point
        shows a tooltip with the closest threshold (cutoff) in the data.
    title : str
        Chart title.  When empty, defaults to ``"ROC curve  (AUC = …)"``.
    width, height : int
        Chart dimensions in pixels.
    curve_color : str
        CSS color for the ROC curve.
    **kwargs
        Additional keyword arguments are passed to `_compute_roc` (e.g. `reverse_score=True` if lower scores indicate more likely positive).

    Returns
    -------
    alt.LayerChart
    """
    alt.data_transformers.disable_max_rows()

    roc_df = _compute_roc(df, score_col, label_col, **kwargs)
    auc = _compute_auc(roc_df)
    chart_title = alt.Title(title, subtitle=f"ROC curve  (AUC = {auc:.3f})")

    # --- Reference diagonal (random classifier) -----------------------
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]}))
        .mark_line(color="gray", strokeDash=[4, 4], opacity=0.6)
        .encode(x="x:Q", y="y:Q")
    )

    # --- Main ROC curve -----------------------------------------------
    # Drop the threshold column: boundary points have null threshold and
    # the curve only needs sensitivity/specificity for its x/y encoding.
    curve = (
        alt.Chart(roc_df.select(["sensitivity", "specificity"]))
        .mark_line(color=curve_color)
        .encode(
            x=alt.X(
                "sensitivity:Q", title="Sensitivity", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "specificity:Q", title="Specificity", scale=alt.Scale(domain=[0, 1])
            ),
            tooltip=[
                alt.Tooltip("sensitivity:Q", format=".3f"),
                alt.Tooltip("specificity:Q", format=".3f"),
            ],
        )
    )

    layers: list[alt.Chart | alt.LayerChart] = [diag, curve]

    # --- Specificity-level markers ------------------------------------
    if specificity_levels:
        roc_with_thresh = roc_df.filter(pl.col("threshold").is_not_null())

        # Resolve each target level to the closest curve point with
        # specificity >= spec_level (never below the target).
        # Among equally-close points, prefer the highest sensitivity.
        level_data: list[tuple[float, float, float, float]] = []
        for spec_level in specificity_levels:
            candidates = roc_with_thresh.filter(pl.col("specificity") >= spec_level)
            if candidates.is_empty():
                max_spec = float(roc_with_thresh["specificity"].max())  # type: ignore[arg-type]
                raise ValueError(
                    f"No curve point achieves specificity ≥ {spec_level:.3f}. "
                    f"Maximum specificity in data is {max_spec:.3f}."
                )
            row = (
                candidates.with_columns(
                    (pl.col("specificity") - spec_level).alias("_diff")
                )
                .sort(["_diff", "sensitivity"], descending=[False, True])
                .row(0, named=True)
            )
            level_data.append(
                (spec_level, row["sensitivity"], row["specificity"], row["threshold"])
            )

        # Line segments: 2 rows per segment, grouped by seg_id via detail encoding.
        # All y-coordinates use actual_spec so that lines and marker are consistent.
        # The legend label shows actual_spec; target is kept for the tooltip.
        seg_rows: list[dict] = []
        pt_rows: list[dict] = []
        for i, (spec_level, sens, actual_spec, threshold) in enumerate(level_data):
            label = f"{actual_spec:.3f}"
            target = f"{spec_level:.2f}"
            seg_rows += [
                # Horizontal: (0, actual_spec) → (sens, actual_spec)
                {
                    "seg": f"h{i}",
                    "x": 0.0,
                    "y": actual_spec,
                    "level": label,
                    "target": target,
                    "threshold": threshold,
                },
                {
                    "seg": f"h{i}",
                    "x": sens,
                    "y": actual_spec,
                    "level": label,
                    "target": target,
                    "threshold": threshold,
                },
                # Vertical: (sens, actual_spec) → (sens, 0)
                {
                    "seg": f"v{i}",
                    "x": sens,
                    "y": actual_spec,
                    "level": label,
                    "target": target,
                    "threshold": threshold,
                },
                {
                    "seg": f"v{i}",
                    "x": sens,
                    "y": 0.0,
                    "level": label,
                    "target": target,
                    "threshold": threshold,
                },
            ]
            pt_rows.append(
                {
                    "sensitivity": sens,
                    "specificity": actual_spec,
                    "level": label,
                    "target": target,
                    "threshold": threshold,
                }
            )

        seg_df = pl.DataFrame(seg_rows)
        pt_df = pl.DataFrame(pt_rows)

        level_lines = (
            alt.Chart(seg_df)
            .mark_line(strokeDash=[5, 3])
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("level:N", title="Specificity"),
                detail="seg:N",
            )
        )

        level_pts = (
            alt.Chart(pt_df)
            .mark_point(size=80, filled=True)
            .encode(
                x=alt.X("sensitivity:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("specificity:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("level:N", title="Specificity"),
                tooltip=[
                    alt.Tooltip("target:N", title="Target specificity"),
                    alt.Tooltip("level:N", title="Actual specificity"),
                    alt.Tooltip("sensitivity:Q", format=".3f"),
                    alt.Tooltip("threshold:Q", title="Cutoff", format=".4g"),
                ],
            )
        )

        layers += [level_lines, level_pts]

    return (
        alt.layer(*layers)
        .properties(title=chart_title, width=width, height=height)
        .configure_axis(gridColor="gray", gridDash=[3, 3], gridOpacity=0.5)
        .configure_view(strokeWidth=0)
    )
