import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from sklearn.metrics import roc_auc_score

from plotutils.auc import _compute_auc, _compute_pauc, _compute_roc, plot_roc_curve

# Perfect classifier: positives all score above negatives, no overlap.
_DF_PERFECT = pl.DataFrame(
    {
        "score": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
        "label": [1,    1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0   ],
    }
)

# Noisy classifier: score distributions overlap so AUC < 1.
# Positives (label=1): 10 samples, some with low scores.
# Negatives (label=0): 9 samples, two with scores above several positives.
_DF_NOISY = pl.DataFrame(
    {
        "score": [
            0.95, 0.88, 0.80, 0.73, 0.65, 0.60, 0.48, 0.40, 0.35, 0.28,
            0.82, 0.70, 0.55, 0.42, 0.25, 0.18, 0.12, 0.05, 0.02,
        ],
        "label": [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    }
)


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_compute_roc_perfect_classifier():
    """AUC should be 1.0 when positives and negatives are perfectly separated."""
    roc = _compute_roc(_DF_PERFECT, "score", "label")
    auc = _compute_auc(roc)
    assert abs(auc - 1.0) < 1e-9


def test_compute_roc_noisy_auc_range():
    """AUC for the noisy dataset should be strictly between 0.5 and 1.0."""
    roc = _compute_roc(_DF_NOISY, "score", "label")
    auc = _compute_auc(roc)
    assert 0.5 < auc < 1.0


def test_compute_roc_random_classifier():
    """AUC should be ≈ 0.5 when scores are identical for all samples."""
    df = pl.DataFrame({"score": [0.5] * 10, "label": [1, 0] * 5})
    roc = _compute_roc(df, "score", "label")
    auc = _compute_auc(roc)
    assert abs(auc - 0.5) < 0.05


def test_compute_roc_includes_boundary_points():
    """ROC DataFrame must start at (sensitivity=0, specificity=1)
    and end at (sensitivity=1, specificity=0)."""
    roc = _compute_roc(_DF_NOISY, "score", "label")
    assert roc["sensitivity"][0] == 0.0
    assert roc["specificity"][0] == 1.0
    assert roc["sensitivity"][-1] == 1.0
    assert roc["specificity"][-1] == 0.0


def test_compute_roc_raises_on_single_class():
    """Raise ValueError when only one class is present."""
    df = pl.DataFrame({"score": [0.9, 0.8, 0.7], "label": [1, 1, 1]})
    with pytest.raises(ValueError, match="Both classes"):
        _compute_roc(df, "score", "label")


# ---------------------------------------------------------------------------
# AUC agreement with scikit-learn
# ---------------------------------------------------------------------------

def _sklearn_auc(df: pl.DataFrame) -> float:
    return float(roc_auc_score(df["label"].to_list(), df["score"].to_list()))


def _our_auc(df: pl.DataFrame) -> float:
    return _compute_auc(_compute_roc(df, "score", "label"))


def test_auc_matches_sklearn_perfect():
    """AUC = 1.0 — both implementations must agree."""
    assert abs(_our_auc(_DF_PERFECT) - _sklearn_auc(_DF_PERFECT)) < 1e-9


def test_auc_matches_sklearn_noisy():
    """AUC < 1 with overlapping distributions — both implementations must agree."""
    assert abs(_our_auc(_DF_NOISY) - _sklearn_auc(_DF_NOISY)) < 1e-9


def test_auc_matches_sklearn_balanced():
    """Balanced, near-random scores — both implementations must agree."""
    df = pl.DataFrame({
        "score": [0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6, 0.1, 0.5, 0.45],
        "label": [1,   0,   0,   1,   1,   0,   0,   1,   1,   0  ],
    })
    assert abs(_our_auc(df) - _sklearn_auc(df)) < 1e-9


def test_auc_matches_sklearn_ties():
    """Tied scores across classes — both implementations must agree."""
    df = pl.DataFrame({
        "score": [0.8, 0.8, 0.5, 0.5, 0.5, 0.2, 0.2],
        "label": [1,   0,   1,   1,   0,   0,   1  ],
    })
    assert abs(_our_auc(df) - _sklearn_auc(df)) < 1e-9


# ---------------------------------------------------------------------------
# Chart snapshot tests  (all use _DF_NOISY for a realistic non-perfect curve)
# ---------------------------------------------------------------------------


def test_plot_roc_curve_basic(snapshot, snapshot_svg):
    """Basic ROC curve without specificity-level markers."""
    chart = plot_roc_curve(_DF_NOISY, score_col="score", label_col="label")
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_roc_curve_with_levels(snapshot, snapshot_svg):
    """ROC curve with three specificity-level annotations.

    Checks that the intersection markers sit exactly on the dashed lines,
    i.e. both use actual_spec (not the target spec_level).
    """
    chart = plot_roc_curve(
        _DF_NOISY,
        score_col="score",
        label_col="label",
        specificity_levels=[0.90, 0.75, 0.50],
    )
    chart_dict = chart.to_dict()

    assert "layer" in chart_dict
    # With levels: shade + diag + curve + level_lines + level_pts
    assert len(chart_dict["layer"]) == 5
    assert normalize_chart_dict(chart_dict) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def _get_level_datasets(chart):
    """Return (seg_data, pt_data) from a chart's inline datasets."""
    datasets = chart.to_dict().get("datasets", {})
    seg_data = next(v for v in datasets.values() if v and "seg" in v[0])
    pt_data = next(v for v in datasets.values() if v and "level" in v[0] and "sensitivity" in v[0])
    return seg_data, pt_data


def test_plot_roc_curve_level_marker_alignment():
    """Intersection point y must equal the horizontal segment's y (actual_spec)."""
    chart = plot_roc_curve(
        _DF_NOISY,
        score_col="score",
        label_col="label",
        specificity_levels=[0.90, 0.75, 0.50],
    )
    seg_data, pt_data = _get_level_datasets(chart)

    for pt in pt_data:
        # Match via target (the requested level string)
        target = pt["target"]
        h_end = next(r for r in seg_data if r["target"] == target and r["seg"].startswith("h") and r["x"] > 0)
        assert pt["specificity"] == h_end["y"], (
            f"Target {target}: point y={pt['specificity']} != horizontal segment y={h_end['y']}"
        )


def test_plot_roc_curve_actual_spec_geq_target():
    """Actual specificity at each marker must be >= the requested target."""
    chart = plot_roc_curve(
        _DF_NOISY,
        score_col="score",
        label_col="label",
        specificity_levels=[0.90, 0.75, 0.50],
    )
    _, pt_data = _get_level_datasets(chart)

    for pt in pt_data:
        actual = pt["specificity"]
        target = float(pt["target"])
        assert actual >= target, f"Actual {actual:.3f} < target {target:.2f}"


def test_plot_roc_curve_level_legend_shows_actual():
    """The legend label (level field) must reflect the actual specificity, not the target."""
    chart = plot_roc_curve(
        _DF_NOISY,
        score_col="score",
        label_col="label",
        specificity_levels=[0.90, 0.75, 0.50],
    )
    _, pt_data = _get_level_datasets(chart)

    for pt in pt_data:
        # label is actual_spec formatted to 3 d.p.; pAUC is in the tooltip, not the legend
        assert pt["level"] == f"{pt['specificity']:.3f}"
        assert "pAUC" not in pt["level"]
        # label must differ from target when they don't match exactly
        if pt["specificity"] != float(pt["target"]):
            assert pt["level"] != pt["target"]


def test_plot_roc_curve_unreachable_level_raises():
    """Raise ValueError when no curve point achieves the requested specificity."""
    # Top-scored sample is a negative → max achievable specificity < 1
    df = pl.DataFrame({"score": [0.9, 0.8, 0.7, 0.6], "label": [0, 1, 1, 0]})
    with pytest.raises(ValueError, match="specificity"):
        plot_roc_curve(df, score_col="score", label_col="label", specificity_levels=[0.6])


def test_plot_roc_curve_custom_title(snapshot, snapshot_svg):
    """Custom title is used verbatim (no auto AUC suffix)."""
    chart = plot_roc_curve(
        _DF_NOISY,
        score_col="score",
        label_col="label",
        title="My custom title",
    )
    title = chart.to_dict()["title"]
    text = title.get("text", title) if isinstance(title, dict) else title
    assert text == "My custom title"
    assert normalize_chart_dict(chart.to_dict()) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_roc_curve_auc_in_default_title():
    """Default subtitle includes the AUC value."""
    chart = plot_roc_curve(_DF_NOISY, score_col="score", label_col="label")
    title = chart.to_dict()["title"]
    # title is rendered as an alt.Title dict {text, subtitle}
    subtitle = title.get("subtitle", "") if isinstance(title, dict) else title
    assert "AUC" in subtitle


def test_plot_roc_curve_dimensions():
    """Custom width and height are reflected in the chart properties."""
    chart = plot_roc_curve(_DF_NOISY, score_col="score", label_col="label", width=300, height=250)
    d = chart.to_dict()
    assert d["width"] == 300
    assert d["height"] == 250


# ---------------------------------------------------------------------------
# Partial AUC — analytical reference tests
# ---------------------------------------------------------------------------
# For a *perfect* classifier (all positives score above all negatives):
#   specificity focus [0.8, 1.0]: FPR range [0, 0.2], TPR = 1 everywhere
#     raw pAUC  = 0.20    (width of the FPR band)
#     mcclish   = 1.0     (scaled to [0.5, 1])
#   sensitivity focus [0.8, 1.0]: TPR range [0.8, 1.0], FPR = 0 everywhere
#     raw pAUC  = 0.20    (width of the TPR band)
#     mcclish   = 1.0
# For a *random* classifier (TPR = FPR), mcclish → 0.5 for any range.


def _roc_perfect() -> pl.DataFrame:
    df = pl.DataFrame({"score": [0.9] * 6 + [0.1] * 6, "label": [1] * 6 + [0] * 6})
    return _compute_roc(df, "score", "label")


def _roc_random() -> pl.DataFrame:
    """Near-random: alternating labels sorted by score → AUC ≈ 0.5."""
    scores = list(range(20, 0, -1))
    labels = [i % 2 for i in range(20)]
    df = pl.DataFrame({"score": scores, "label": labels})
    return _compute_roc(df, "score", "label")


def test_pauc_perfect_spec_focus_raw():
    assert abs(_compute_pauc(_roc_perfect(), 0.8, 1.0, "specificity", mcclish=False) - 0.2) < 1e-9


def test_pauc_perfect_spec_focus_mcclish():
    assert abs(_compute_pauc(_roc_perfect(), 0.8, 1.0, "specificity", mcclish=True) - 1.0) < 1e-9


def test_pauc_perfect_sens_focus_raw():
    assert abs(_compute_pauc(_roc_perfect(), 0.8, 1.0, "sensitivity", mcclish=False) - 0.2) < 1e-9


def test_pauc_perfect_sens_focus_mcclish():
    assert abs(_compute_pauc(_roc_perfect(), 0.8, 1.0, "sensitivity", mcclish=True) - 1.0) < 1e-9


def test_pauc_random_spec_focus_mcclish():
    """McClish-corrected pAUC should be ≈ 0.5 for a random classifier."""
    assert abs(_compute_pauc(_roc_random(), 0.8, 1.0, "specificity", mcclish=True) - 0.5) < 0.05


def test_pauc_random_sens_focus_mcclish():
    assert abs(_compute_pauc(_roc_random(), 0.8, 1.0, "sensitivity", mcclish=True) - 0.5) < 0.05


def test_pauc_range_lt_full_auc():
    """pAUC over a sub-range must be strictly less than the full AUC."""
    roc = _compute_roc(_DF_NOISY, "score", "label")
    full = _compute_auc(roc)
    partial = _compute_pauc(roc, 0.8, 1.0, "specificity", mcclish=False)
    assert partial < full


def test_pauc_mcclish_between_half_and_one():
    """After McClish correction the result must lie in [0.5, 1]."""
    roc = _compute_roc(_DF_NOISY, "score", "label")
    for focus in ("specificity", "sensitivity"):
        val = _compute_pauc(roc, 0.8, 1.0, focus, mcclish=True)  # type: ignore[arg-type]
        assert 0.5 <= val <= 1.0, f"{focus} focus gave {val}"


def test_pauc_full_range_matches_auc():
    """pAUC over [0, 1] (no mcclish) must equal _compute_auc for both focuses."""
    roc = _compute_roc(_DF_NOISY, "score", "label")
    full_auc = _compute_auc(roc)
    for focus in ("specificity", "sensitivity"):
        pauc = _compute_pauc(roc, 0.0, 1.0, focus, mcclish=False)  # type: ignore[arg-type]
        assert abs(pauc - full_auc) < 1e-9, f"{focus}: pauc={pauc} auc={full_auc}"



# ---------------------------------------------------------------------------
# AUC agreement with pauc package (standard AUC only — partial_auc in v0.2.0
# has a sorting bug and is not used for cross-validation here)
# ---------------------------------------------------------------------------


def test_auc_matches_pauc_package():
    """Standard AUC must match pauc.ROC.auc to floating-point precision."""
    pauc_pkg = pytest.importorskip("pauc")
    for df in (_DF_PERFECT, _DF_NOISY):
        our = _compute_auc(_compute_roc(df, "score", "label"))
        pkg = pauc_pkg.ROC(df["label"].to_list(), df["score"].to_list()).auc
        assert abs(our - pkg) < 1e-9, f"our={our} pkg={pkg}"


# ---------------------------------------------------------------------------
# Categorical / boolean label support in _compute_roc
# ---------------------------------------------------------------------------


def test_compute_roc_boolean_label():
    """Boolean label column (True=positive) must give the same AUC as int 0/1."""
    df_int  = pl.DataFrame({"score": _DF_PERFECT["score"], "label": _DF_PERFECT["label"]})
    df_bool = pl.DataFrame({"score": _DF_PERFECT["score"], "label": _DF_PERFECT["label"].cast(pl.Boolean)})
    auc_int  = _compute_auc(_compute_roc(df_int,  "score", "label"))
    auc_bool = _compute_auc(_compute_roc(df_bool, "score", "label"))
    assert abs(auc_int - auc_bool) < 1e-9


def test_compute_roc_string_label():
    """String label column must produce the same AUC as the equivalent int encoding.

    Convention: lexicographically larger string = positive class.
    Here 'pos' > 'neg', so 'pos' maps to 1.
    """
    scores = _DF_PERFECT["score"].to_list()
    labels_int = _DF_PERFECT["label"].to_list()          # 1 = positive
    labels_str = ["pos" if l == 1 else "neg" for l in labels_int]
    df_int = pl.DataFrame({"score": scores, "label": labels_int})
    df_str = pl.DataFrame({"score": scores, "label": labels_str})
    auc_int = _compute_auc(_compute_roc(df_int, "score", "label"))
    auc_str = _compute_auc(_compute_roc(df_str, "score", "label"))
    assert abs(auc_int - auc_str) < 1e-9


def test_compute_roc_categorical_label():
    """Polars Categorical dtype must be accepted and give the same AUC."""
    scores = _DF_PERFECT["score"].to_list()
    labels_int = _DF_PERFECT["label"].to_list()
    labels_cat = pl.Series(["pos" if l == 1 else "neg" for l in labels_int]).cast(pl.Categorical)
    df_int = pl.DataFrame({"score": scores, "label": labels_int})
    df_cat = pl.DataFrame({"score": pl.Series(scores), "label": labels_cat})
    auc_int = _compute_auc(_compute_roc(df_int, "score", "label"))
    auc_cat = _compute_auc(_compute_roc(df_cat, "score", "label"))
    assert abs(auc_int - auc_cat) < 1e-9


def test_compute_roc_string_label_wrong_class_order():
    """Verify that a string where the positive class sorts FIRST is handled.

    'active' < 'inactive' lexicographically, so 'inactive' → 1.
    We build the data so that high scores go with 'inactive' (= positive),
    which must yield AUC = 1.0.
    """
    df = pl.DataFrame({
        "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        "label": ["inactive", "inactive", "inactive", "active", "active", "active"],
    })
    auc = _compute_auc(_compute_roc(df, "score", "label"))
    assert abs(auc - 1.0) < 1e-9


def test_compute_roc_string_label_three_values_raises():
    """More than two unique string values must raise ValueError."""
    df = pl.DataFrame({"score": [0.9, 0.5, 0.1], "label": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="2 unique values"):
        _compute_roc(df, "score", "label")
