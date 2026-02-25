import polars as pl
import pytest
from conftest import chart_to_svg, normalize_chart_dict
from sklearn.metrics import roc_auc_score

from plotutils.auc import _compute_auc, _compute_roc, plot_roc_curve

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
    # With levels: diag + curve + level_lines + level_pts
    assert len(chart_dict["layer"]) == 4
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
        # label is formatted as actual_spec with 3 decimal places
        assert pt["level"] == f"{pt['specificity']:.3f}"
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
    assert chart.to_dict()["title"] == "My custom title"
    assert normalize_chart_dict(chart.to_dict()) == snapshot
    assert chart_to_svg(chart) == snapshot_svg


def test_plot_roc_curve_auc_in_default_title():
    """Default title includes the AUC value."""
    chart = plot_roc_curve(_DF_NOISY, score_col="score", label_col="label")
    title = chart.to_dict()["title"]
    assert "AUC" in title


def test_plot_roc_curve_dimensions():
    """Custom width and height are reflected in the chart properties."""
    chart = plot_roc_curve(_DF_NOISY, score_col="score", label_col="label", width=300, height=250)
    d = chart.to_dict()
    assert d["width"] == 300
    assert d["height"] == 250
