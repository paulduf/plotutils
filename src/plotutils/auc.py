"""ROC curve with optional specificity-level annotations."""

import json
from typing import Literal

import polars as pl
import altair as alt

from plotutils.boxplot import plot_bivariate_boxes, plot_bivariate_strip


def _coerce_label(series: pl.Series) -> pl.Series:
    """Return a 0/1 Int8 copy of *series*, accepting any binary input dtype.

    - Numeric (int/float) with values in {0, 1}: returned as Int8.
    - Boolean: True → 1, False → 0.
    - String / Categorical / Enum / other: sorted unique values are mapped;
      the lexicographically smaller value becomes 0, the larger becomes 1.

    Raises ``ValueError`` if the series does not contain exactly 2 unique values.
    """
    dtype = series.dtype
    if dtype == pl.Boolean:
        return series.cast(pl.Int8)
    if dtype.is_numeric():
        return series.cast(pl.Int8)
    # Non-numeric: coerce via string representation.
    uniq = sorted(series.cast(pl.String).unique().to_list())
    if len(uniq) != 2:
        raise ValueError(
            f"label column must have exactly 2 unique values; got {uniq!r}"
        )
    return (series.cast(pl.String) == uniq[1]).cast(pl.Int8)


def _compute_roc(
    df: pl.DataFrame,
    score_col: str,
    label_col: str,
    reverse_score: bool = False,
) -> pl.DataFrame:
    """AUROC computation with `polars` backend. Returns a DataFrame with columns (threshold, sensitivity, specificity).

    Includes boundary points at (sensitivity=0, specificity=1) and
    (sensitivity=1, specificity=0) with null threshold.

    Ties (multiple samples sharing the same score) are handled correctly:
    all tied samples are grouped into one threshold step before the cumulative
    TP/FP counts are computed.

    The *label_col* can be any binary column: integer 0/1, boolean, or a
    string / categorical column with exactly two distinct values (the
    lexicographically larger value is treated as the positive class).
    """
    # Normalise label to 0/1 Int8 so downstream comparisons always work.
    label_col_int = "__label__"
    df = df.with_columns(_coerce_label(df[label_col]).alias(label_col_int))

    n_pos = int((df[label_col_int] == 1).sum())
    n_neg = int((df[label_col_int] == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes (0 and 1) must be present in label_col.")

    curve = (
        df.select([score_col, label_col_int])
        .group_by(score_col)
        .agg(
            (pl.col(label_col_int) == 1).sum().alias("tp_step"),
            (pl.col(label_col_int) == 0).sum().alias("fp_step"),
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


def _compute_pauc(
    roc_df: pl.DataFrame,
    range_min: float,
    range_max: float,
    focus: Literal["sensitivity", "specificity"] = "specificity",
    mcclish: bool = True,
) -> float:
    """Partial AUC over a restricted range, with optional McClish correction.

    Uses the same trapezoidal rule as :func:`_compute_auc`, so
    ``_compute_pauc(roc, 0.0, 1.0, focus)`` equals ``_compute_auc(roc)``
    for both focus modes.

    Parameters
    ----------
    roc_df : pl.DataFrame
        Output of :func:`_compute_roc` (columns: threshold, sensitivity, specificity).
    range_min, range_max : float
        The interval ``[range_min, range_max]`` of the metric named by ``focus``.
        For ``focus="specificity"`` this is a specificity range (e.g. ``0.8, 1.0``);
        for ``focus="sensitivity"`` it is a sensitivity range.
    focus : {"specificity", "sensitivity"}
        Which axis the partial range applies to.
    mcclish : bool
        Apply the McClish (1989) standardisation so that the result lies in
        ``[0.5, 1]``: 0.5 = random classifier, 1 = perfect.

    Returns
    -------
    float
    """
    if focus == "sensitivity":
        # Clip the ROC curve to sensitivity ∈ [range_min, range_max] and apply
        # the same formula as _compute_auc: ∑ Δsens · avg_spec.
        boundary = pl.DataFrame(
            {"sensitivity": [range_min, range_max], "specificity": [None, None]}
        )
        clipped = (
            pl.concat([roc_df.select(["sensitivity", "specificity"]), boundary])
            .sort("sensitivity")
            .with_columns(pl.col("specificity").interpolate_by("sensitivity"))
            .filter(
                pl.col("specificity").is_not_null()
                & (pl.col("sensitivity") >= range_min)
                & (pl.col("sensitivity") <= range_max)
            )
        )
        pauc = float(
            clipped.with_columns(
                [
                    pl.col("sensitivity").diff().alias("dx"),
                    (
                        (pl.col("specificity") + pl.col("specificity").shift(1)) / 2
                    ).alias("avg_y"),
                ]
            )
            .filter(pl.col("dx").is_not_null())
            .select((pl.col("dx") * pl.col("avg_y")).sum())
            .item()
        )
        if mcclish:
            max_area = range_max - range_min
            min_area = (range_max - range_min) - 0.5 * (range_max**2 - range_min**2)
            if max_area > min_area:
                pauc = 0.5 * (1.0 + (pauc - min_area) / (max_area - min_area))

    else:  # specificity focus
        # Integrate sensitivity over FPR ∈ [1-range_max, 1-range_min]:
        # ∑ Δfpr · avg_sens, clipped with interpolated boundaries.
        low_fpr = 1.0 - range_max
        high_fpr = 1.0 - range_min
        boundary = pl.DataFrame(
            {"fpr": [low_fpr, high_fpr], "sensitivity": [None, None]}
        )
        clipped = (
            pl.concat(
                [
                    roc_df.with_columns(
                        (1.0 - pl.col("specificity")).alias("fpr")
                    ).select(["fpr", "sensitivity"]),
                    boundary,
                ]
            )
            .sort("fpr")
            .with_columns(pl.col("sensitivity").interpolate_by("fpr"))
            .filter(
                pl.col("sensitivity").is_not_null()
                & (pl.col("fpr") >= low_fpr)
                & (pl.col("fpr") <= high_fpr)
            )
        )
        pauc = float(
            clipped.with_columns(
                [
                    pl.col("fpr").diff().alias("dx"),
                    (
                        (pl.col("sensitivity") + pl.col("sensitivity").shift(1)) / 2
                    ).alias("avg_y"),
                ]
            )
            .filter(pl.col("dx").is_not_null())
            .select((pl.col("dx") * pl.col("avg_y")).sum())
            .item()
        )
        if mcclish:
            max_area = high_fpr - low_fpr
            min_area = 0.5 * (high_fpr**2 - low_fpr**2)
            if max_area > min_area:
                pauc = 0.5 * (1.0 + (pauc - min_area) / (max_area - min_area))

    return pauc


def _lerp_hex(t: float, c1: str = "#006d2c", c2: str = "#74c476") -> str:
    """Linearly interpolate between two hex colours.

    Parameters
    ----------
    t : float
        Interpolation factor in [0, 1].  0 → ``c1`` (dark green), 1 → ``c2`` (light green).
    c1, c2 : str
        Hex colour strings (``#rrggbb``).
    """

    def _parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = _parse(c1)
    r2, g2, b2 = _parse(c2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_roc_curve(
    df: pl.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    specificity_levels: list[float] | None = None,
    title: str = "",
    width: int = 400,
    height: int = 400,
    curve_color: str = "steelblue",
    id_col: str | None = None,
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
    id_col : str or None
        Optional column name containing patient / sample identifiers.  When
        provided, hovering over any threshold step on the curve reveals the
        ID(s) of the patient(s) whose score equals that cutoff (ties are shown
        as a comma-separated list).
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
        .mark_line(color="#444", strokeWidth=0.75, opacity=0.4)
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

    # --- Per-threshold patient ID tooltip (optional) ------------------
    if id_col is not None:
        # For each unique score value (= threshold step), collect the IDs of
        # all patients at that score.  Ties produce a comma-separated list.
        id_by_threshold = (
            df.select([score_col, id_col])
            .with_columns(pl.col(score_col).cast(pl.Float64).alias("threshold"))
            .group_by("threshold")
            .agg(pl.col(id_col).cast(pl.Utf8).sort().str.join(", ").alias("_ids"))
        )
        roc_interactive = (
            roc_df.filter(pl.col("threshold").is_not_null())
            .join(id_by_threshold, on="threshold", how="left")
            .select(["sensitivity", "specificity", "threshold", "_ids"])
        )
        id_points = (
            alt.Chart(roc_interactive)
            .mark_point(opacity=0, size=300, filled=True)
            .encode(
                x=alt.X("sensitivity:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("specificity:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    alt.Tooltip("sensitivity:Q", format=".3f"),
                    alt.Tooltip("specificity:Q", format=".3f"),
                    alt.Tooltip("threshold:Q", format=".4g", title="Cutoff"),
                    alt.Tooltip("_ids:N", title="Patient ID(s)"),
                ],
            )
        )
        layers.append(id_points)

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

        # Pre-compute pAUC for each level: range [spec_level, 1.0] with McClish.
        pauc_by_target: dict[str, float] = {
            f"{sl:.2f}": _compute_pauc(
                roc_df, sl, 1.0, focus="specificity", mcclish=True
            )
            for sl, *_ in level_data
        }

        # Line segments: 2 rows per segment, grouped by seg_id via detail encoding.
        # All y-coordinates use actual_spec so that lines and marker are consistent.
        # The legend label shows actual_spec only; pAUC is in the tooltip.
        seg_rows: list[dict] = []
        pt_rows: list[dict] = []
        for i, (spec_level, sens, actual_spec, threshold) in enumerate(level_data):
            target = f"{spec_level:.2f}"
            label = f"{actual_spec:.3f}"
            pauc_val = pauc_by_target[target]
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
                    "pauc": pauc_val,
                }
            )

        seg_df = pl.DataFrame(seg_rows)
        pt_df = pl.DataFrame(pt_rows)

        # Build a deterministic green→grey colour scale.
        # Sort by spec_level descending so the highest level gets green (t=0)
        # and the lowest gets grey (t=1).
        sorted_levels = sorted(level_data, key=lambda x: x[0], reverse=True)
        n_lvl = len(sorted_levels)
        color_domain: list[str] = []
        color_range: list[str] = []
        for idx, (_, _, actual_spec, _) in enumerate(sorted_levels):
            t = idx / max(n_lvl - 1, 1)
            color_domain.append(f"{actual_spec:.3f}")
            color_range.append(_lerp_hex(t))
        color_scale = alt.Scale(domain=color_domain, range=color_range)

        # Shade the "high-specificity" region: x∈[0,1], y∈[min_level, 1].
        # Use the green end of the gradient for a subtle filled band.
        shade_y_lo = min(spec_level for spec_level, *_ in level_data)
        shade_df = pl.DataFrame(
            {"x1": [0.0], "x2": [1.0], "y1": [shade_y_lo], "y2": [1.0]}
        )
        shade = (
            alt.Chart(shade_df)
            .mark_rect(color=color_range[0], opacity=0.08)
            .encode(
                x=alt.X("x1:Q", scale=alt.Scale(domain=[0, 1])),
                x2="x2:Q",
                y=alt.Y("y1:Q", scale=alt.Scale(domain=[0, 1])),
                y2="y2:Q",
            )
        )
        # Insert shade before the diagonal so it sits in the background.
        layers.insert(0, shade)

        level_lines = (
            alt.Chart(seg_df)
            .mark_line(strokeDash=[5, 3])
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("level:N", scale=color_scale, title="Specificity"),
                detail="seg:N",
            )
        )

        level_pts = (
            alt.Chart(pt_df)
            .mark_point(size=80, filled=True)
            .encode(
                x=alt.X("sensitivity:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("specificity:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("level:N", scale=color_scale, title="Specificity"),
                tooltip=[
                    alt.Tooltip("target:N", title="Target specificity"),
                    alt.Tooltip("level:N", title="Actual specificity"),
                    alt.Tooltip("sensitivity:Q", format=".3f"),
                    alt.Tooltip("threshold:Q", title="Cutoff", format=".4g"),
                    alt.Tooltip("pauc:Q", title="pAUC (McClish)", format=".3f"),
                ],
            )
        )

        layers += [level_lines, level_pts]

    return (
        alt.layer(*layers)
        .properties(title=chart_title, width=width, height=height)
        .configure_axis(
            gridColor="#444", gridWidth=0.75, gridDash=[3, 3], gridOpacity=0.4
        )
        .configure_view(strokeWidth=0)
    )


# ── Linked AUC report ─────────────────────────────────────────────────────────


class AUCReport:
    """Pre-computed linked AUC explorer that renders to a self-contained HTML page.

    Parameters
    ----------
    df : pl.DataFrame
        Data with one column per variable (continuous score) and one per
        outcome (binary 0/1 label), plus an optional ID column.
    variables : list[str]
        Column names of the continuous score variables.
    outcomes : list[str]
        Column names of the binary outcome variables.
    id_col : str or None
        Optional column with sample / patient identifiers.  When provided,
        hovering over points in the ROC and distribution charts reveals the ID.
    kind : {"box", "strip"}
        Distribution chart style: grouped boxplot or jittered strip plot.
    specificity_levels : list[float] or None
        Target specificity levels to annotate on each ROC curve.  Levels
        unreachable by a given curve are silently skipped.
        Defaults to ``[0.8, 0.9]``.
    chart_width, chart_height : int
        Pixel dimensions of each chart panel (ROC and distribution).
        The scatter panel is square with side ``chart_width + 20``.

    Examples
    --------
    One-liner for Quarto / Jupyter::

        from plotutils.auc import AUCReport
        AUCReport(df, variables, outcomes, id_col="patient_id").to_html("report.html")
    """

    _DEFAULT_SPEC_LEVELS: list[float] = [0.8, 0.9, 0.99]

    def __init__(
        self,
        df: pl.DataFrame,
        variables: list[str],
        outcomes: list[str],
        id_col: str | None = None,
        kind: Literal["box", "strip"] = "box",
        specificity_levels: list[float] | None = None,
        chart_width: int = 320,
        chart_height: int = 290,
    ) -> None:
        self._df = df
        self._variables = variables
        self._outcomes = outcomes
        self._id_col = id_col
        self._kind = kind
        self._target_spec = (
            specificity_levels
            if specificity_levels is not None
            else self._DEFAULT_SPEC_LEVELS
        )
        self._w = chart_width
        self._h = chart_height
        self._plot_dist = (
            plot_bivariate_boxes if kind == "box" else plot_bivariate_strip
        )

        # Pre-compute all chart specs (expensive, done once).
        self._auc_df = self._build_auc_df()
        self._roc_specs = self._build_roc_specs()
        self._dist_specs = self._build_dist_specs()

    # ── Internal builders ──────────────────────────────────────────────────

    def _build_auc_df(self) -> pl.DataFrame:
        rows = []
        for var in self._variables:
            row: dict = {"variable": var}
            for out in self._outcomes:
                df_vo = self._df.select([var, out]).rename({var: "score", out: "label"})
                roc = _compute_roc(df_vo, "score", "label")
                row[f"auc_{out}"] = round(_compute_auc(roc), 4)
                # pAUC columns: one per specificity level, None if unreachable.
                max_spec = float(
                    roc.filter(pl.col("threshold").is_not_null())["specificity"].max()  # type: ignore[arg-type]
                )
                for sl in self._target_spec:
                    col = f"pauc_{out}_q{int(sl * 100)}"
                    if sl <= max_spec:
                        row[col] = round(
                            _compute_pauc(
                                roc, sl, 1.0, focus="specificity", mcclish=True
                            ),
                            4,
                        )
                    else:
                        row[col] = None
            rows.append(row)
        return pl.DataFrame(rows)

    def _build_roc_specs(self) -> dict[str, dict]:
        specs: dict[str, dict] = {}
        for var in self._variables:
            for out in self._outcomes:
                select_cols = [self._id_col, var, out] if self._id_col else [var, out]
                df_vo = self._df.select(select_cols).rename(
                    {var: "score", out: "label"}
                )
                roc_check = _compute_roc(
                    df_vo.select(["score", "label"]), "score", "label"
                )
                max_spec = float(
                    roc_check.filter(pl.col("threshold").is_not_null())[
                        "specificity"
                    ].max()  # type: ignore[arg-type]
                )
                achievable = [s for s in self._target_spec if s <= max_spec] or None
                chart = plot_roc_curve(
                    df_vo,
                    score_col="score",
                    label_col="label",
                    id_col=self._id_col,
                    specificity_levels=achievable,
                    title=var,
                    width=self._w,
                    height=self._h,
                )
                specs[f"{var}|{out}"] = json.loads(chart.to_json())
        return specs

    def _build_dist_specs(self) -> dict[str, dict]:
        specs: dict[str, dict] = {}
        for var in self._variables:
            for out_x in self._outcomes:
                for out_y in self._outcomes:
                    d: dict = {
                        "score": self._df[var].to_list(),
                        "label_x": self._df[out_x].to_list(),
                        "label_y": self._df[out_y].to_list(),
                    }
                    if self._id_col:
                        d[self._id_col] = self._df[self._id_col].to_list()
                    chart = self._plot_dist(
                        pl.DataFrame(d),
                        score_col="score",
                        label_x_col="label_x",
                        label_y_col="label_y",
                        id_col=self._id_col,
                        title=var,
                        x_title=out_x.replace("_", " "),
                        color_title=out_y.replace("_", " "),
                        y_title="score",
                        width=self._w,
                        height=self._h,
                    )
                    specs[f"{var}|{out_x}|{out_y}"] = json.loads(chart.to_json())
        return specs

    @staticmethod
    def _auc_lookup(signal: str, outcomes: list[str]) -> str:
        """Nested Vega ternary to pick the right AUC datum field by signal value."""
        expr = f"datum.auc_{outcomes[-1]}"
        for out in reversed(outcomes[:-1]):
            expr = f"{signal} == '{out}' ? datum.auc_{out} : {expr}"
        return expr

    def _build_scatter(self) -> alt.Chart:
        outcomes = self._outcomes
        outcome_labels = [o.replace("_", " ") for o in outcomes]
        scatter_size = self._w + 20

        outcome_x = alt.param(
            name="outcome_x",
            value=outcomes[0],
            bind=alt.binding_select(
                options=outcomes, labels=outcome_labels, name="X-axis: "
            ),
        )
        outcome_y = alt.param(
            name="outcome_y",
            value=outcomes[1],
            bind=alt.binding_select(
                options=outcomes, labels=outcome_labels, name="Y-axis: "
            ),
        )
        click_sel = alt.selection_point(
            fields=["variable"], name="var_sel", on="click", empty=False
        )

        return (
            alt.Chart(self._auc_df)
            .transform_calculate(
                x_auc=self._auc_lookup("outcome_x", outcomes),
                y_auc=self._auc_lookup("outcome_y", outcomes),
            )
            .mark_point(filled=True)
            .encode(
                x=alt.X(
                    "x_auc:Q",
                    title="AUC (X outcome)",
                    scale=alt.Scale(domain=[0.4, 1.05]),
                ),
                y=alt.Y(
                    "y_auc:Q",
                    title="AUC (Y outcome)",
                    scale=alt.Scale(domain=[0.4, 1.05]),
                ),
                color=alt.condition(
                    click_sel,
                    alt.Color("variable:N", legend=None),
                    alt.value("lightgray"),
                ),
                size=alt.condition(click_sel, alt.value(180), alt.value(80)),
                tooltip=[
                    alt.Tooltip("variable:N"),
                    alt.Tooltip("x_auc:Q", format=".3f", title="AUC (X)"),
                    alt.Tooltip("y_auc:Q", format=".3f", title="AUC (Y)"),
                ],
            )
            .add_params(click_sel, outcome_x, outcome_y)
            .properties(
                title=alt.Title(
                    "AUC – AUC scatter",
                    subtitle="Click a variable to reveal its charts →",
                ),
                width=scatter_size,
                height=scatter_size,
            )
            .configure_axis(
                gridColor="#444", gridWidth=0.75, gridDash=[3, 3], gridOpacity=0.4
            )
            .configure_view(strokeWidth=0)
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def to_html(self, path: str | None = None) -> str:
        """Render the linked AUC explorer as a self-contained HTML page.

        Parameters
        ----------
        path : str or None
            If provided, write the HTML to this file path in addition to
            returning it.

        Returns
        -------
        str
            Full HTML page as a string (suitable for ``IPython.display.HTML``
            or Quarto ``{python}`` cells with ``output: asis``).
        """
        dist_kind_label = "Boxplot" if self._kind == "box" else "Strip plot"
        roc_specs_js = json.dumps(self._roc_specs)
        dist_specs_js = json.dumps(self._dist_specs)
        scatter_js = json.dumps(json.loads(self._build_scatter().to_json()))

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Linked AUC Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@6.1.0"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
  <style>
    body {{
      font-family: sans-serif;
      padding: 20px 28px;
      background: #fafafa;
      color: #222;
    }}
    h1 {{ font-size: 18px; margin: 0 0 4px; }}
    p.hint {{ color: #777; font-size: 13px; margin: 0 0 18px; }}

    #grid {{
      display: grid;
      grid-template-columns: auto auto;
      grid-template-rows: auto auto;
      gap: 20px;
      align-items: start;
    }}

    .cell-label {{
      font-size: 12px;
      font-weight: bold;
      color: #666;
      margin-bottom: 4px;
      min-height: 16px;
    }}

    .slot {{
      border: 2px dashed #ddd;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #bbb;
      font-size: 13px;
      box-sizing: border-box;
      width: 370px;
      min-height: 320px;
    }}
  </style>
</head>
<body>
  <h1>Linked AUC Explorer</h1>
  <p class="hint">
    Select outcomes for the scatter axes. Click a variable point to populate the other three charts.
  </p>

  <div id="grid">

    <!-- top-left: scatter -->
    <div>
      <div class="cell-label">AUC scatter</div>
      <div id="scatter"></div>
    </div>

    <!-- top-right: ROC for outcome X -->
    <div>
      <div class="cell-label" id="lbl-roc0">ROC — outcome X</div>
      <div class="slot" id="roc0"><span>← click a variable</span></div>
    </div>

    <!-- bottom-left: distribution chart -->
    <div>
      <div class="cell-label" id="lbl-dist">{dist_kind_label} — score distribution</div>
      <div class="slot" id="dist"><span>← click a variable</span></div>
    </div>

    <!-- bottom-right: ROC for outcome Y -->
    <div>
      <div class="cell-label" id="lbl-roc1">ROC — outcome Y</div>
      <div class="slot" id="roc1"><span>← click a variable</span></div>
    </div>

  </div>

<script>
const ROC  = {roc_specs_js};
const DIST = {dist_specs_js};
const SCATTER = {scatter_js};

let activeVar = null;

function embed(id, spec) {{
  vegaEmbed('#' + id, spec, {{actions: false}});
}}

function update(view) {{
  if (!activeVar) return;
  const ox = view.signal('outcome_x');
  const oy = view.signal('outcome_y');

  // ROC charts
  const oxLabel = ox.replace(/_/g, ' ');
  const oyLabel = oy.replace(/_/g, ' ');
  document.getElementById('lbl-roc0').textContent = 'ROC — ' + oxLabel + ' (' + activeVar + ')';
  document.getElementById('lbl-roc1').textContent = 'ROC — ' + oyLabel + ' (' + activeVar + ')';
  const roc0 = ROC[activeVar + '|' + ox];
  const roc1 = ROC[activeVar + '|' + oy];
  if (roc0) embed('roc0', roc0);
  if (roc1) embed('roc1', roc1);

  // Distribution chart
  document.getElementById('lbl-dist').textContent =
    '{dist_kind_label} — ' + activeVar + ' score × ' + oxLabel + ' × ' + oyLabel;
  const dist = DIST[activeVar + '|' + ox + '|' + oy];
  if (dist) embed('dist', dist);
}}

vegaEmbed('#scatter', SCATTER, {{actions: false}}).then(function(r) {{
  const view = r.view;

  view.addEventListener('click', function(event, item) {{
    if (item && item.datum && item.datum.variable) {{
      activeVar = item.datum.variable;
      update(view);
    }}
  }});

  view.addSignalListener('outcome_x', function() {{ update(view); }});
  view.addSignalListener('outcome_y', function() {{ update(view); }});
}});
</script>
</body>
</html>
"""
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html
