import polars as pl
import pytest
from plotutils.upset import _preprocess_upset, plot_upset


@pytest.fixture
def membership_df() -> pl.DataFrame:
    """8 rows covering all 2^3 combinations of three sets."""
    return pl.DataFrame(
        {
            "setA": [0, 0, 0, 0, 1, 1, 1, 1],
            "setB": [0, 0, 1, 1, 0, 0, 1, 1],
            "setC": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def weighted_df() -> pl.DataFrame:
    """DataFrame with unequal intersection sizes for sort/filter tests."""
    return pl.DataFrame(
        {
            "A": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            "B": [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            "C": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


SET_COLS_3 = ["setA", "setB", "setC"]


# ── preprocessing: intersection counts ────────────────────────────────


def test_preprocess_intersection_counts(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3)
    # All 8 combinations present, each exactly once.
    assert data.intersection_df.shape[0] == 8
    assert data.intersection_df["cardinality"].to_list() == [1] * 8


def test_preprocess_degree(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3)
    degrees = data.intersection_df.sort(SET_COLS_3)["degree"].to_list()
    # Sorted by (setA, setB, setC): 000→0, 001→1, 010→1, 011→2, 100→1, 101→2, 110→2, 111→3
    assert degrees == [0, 1, 1, 2, 1, 2, 2, 3]


# ── preprocessing: filtering ─────────────────────────────────────────


def test_preprocess_min_degree_filter(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3, min_degree=2)
    degrees = data.intersection_df["degree"].to_list()
    assert all(d >= 2 for d in degrees)
    assert data.intersection_df.shape[0] == 4  # degree 2 (×3) + degree 3 (×1)


def test_preprocess_max_degree_filter(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3, max_degree=1)
    degrees = data.intersection_df["degree"].to_list()
    assert all(d <= 1 for d in degrees)
    assert data.intersection_df.shape[0] == 4  # degree 0 (×1) + degree 1 (×3)


# ── preprocessing: sorting ────────────────────────────────────────────


def test_preprocess_sort_frequency_default_descending(weighted_df):
    """Default sort_order for frequency is descending (biggest first)."""
    data = _preprocess_upset(weighted_df, ["A", "B", "C"], sort_by="frequency")
    cards = data.intersection_df["cardinality"].to_list()
    assert cards == sorted(cards, reverse=True)


def test_preprocess_sort_frequency_ascending(weighted_df):
    data = _preprocess_upset(weighted_df, ["A", "B", "C"], sort_by="frequency", sort_order="ascending")
    cards = data.intersection_df["cardinality"].to_list()
    assert cards == sorted(cards)


def test_preprocess_sort_degree_default_ascending(weighted_df):
    """Default sort_order for degree is ascending (simplest first)."""
    data = _preprocess_upset(weighted_df, ["A", "B", "C"], sort_by="degree")
    degrees = data.intersection_df["degree"].to_list()
    for i in range(len(degrees) - 1):
        assert degrees[i] <= degrees[i + 1]


def test_preprocess_sort_multi_key_default_dirs(weighted_df):
    """sort_by=["degree", "frequency"] uses ascending for degree, descending for frequency."""
    data = _preprocess_upset(
        weighted_df, ["A", "B", "C"],
        sort_by=["degree", "frequency"],
    )
    rows = data.intersection_df.select("degree", "cardinality").rows()
    for i in range(len(rows) - 1):
        d1, c1 = rows[i]
        d2, c2 = rows[i + 1]
        if d1 == d2:
            # Within the same degree: frequency descending.
            assert c1 >= c2, f"row {i} card {c1} < row {i+1} card {c2}"
        else:
            # Across degrees: ascending.
            assert d1 < d2, f"row {i} degree {d1} >= row {i+1} degree {d2}"


def test_preprocess_sort_multi_key_uniform_order(weighted_df):
    """A single sort_order string applies to all keys."""
    data = _preprocess_upset(
        weighted_df, ["A", "B", "C"],
        sort_by=["degree", "frequency"], sort_order="ascending",
    )
    rows = data.intersection_df.select("degree", "cardinality").rows()
    for i in range(len(rows) - 1):
        assert rows[i] <= rows[i + 1]


def test_preprocess_sort_multi_key_per_key_order(weighted_df):
    """A list of sort_order strings sets direction per key."""
    data = _preprocess_upset(
        weighted_df, ["A", "B", "C"],
        sort_by=["degree", "frequency"],
        sort_order=["descending", "ascending"],
    )
    rows = data.intersection_df.select("degree", "cardinality").rows()
    for i in range(len(rows) - 1):
        d1, c1 = rows[i]
        d2, c2 = rows[i + 1]
        if d1 == d2:
            assert c1 <= c2, f"row {i} card {c1} > row {i+1} card {c2}"
        else:
            assert d1 > d2, f"row {i} degree {d1} <= row {i+1} degree {d2}"


# ── preprocessing: n_intersections ────────────────────────────────────


def test_preprocess_n_intersections(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3, n_intersections=3)
    assert data.intersection_df.shape[0] == 3


# ── preprocessing: set sizes ─────────────────────────────────────────


def test_preprocess_set_sizes(membership_df):
    data = _preprocess_upset(membership_df, SET_COLS_3)
    sizes = dict(
        zip(
            data.set_sizes_df["set_name"].to_list(),
            data.set_sizes_df["set_size"].to_list(),
        )
    )
    assert sizes == {"setA": 4, "setB": 4, "setC": 4}


def test_preprocess_set_sizes_unequal(weighted_df):
    data = _preprocess_upset(weighted_df, ["A", "B", "C"])
    sizes = dict(
        zip(
            data.set_sizes_df["set_name"].to_list(),
            data.set_sizes_df["set_size"].to_list(),
        )
    )
    assert sizes == {"A": 7, "B": 8, "C": 4}


# ── preprocessing: boolean columns ───────────────────────────────────


def test_preprocess_boolean_columns():
    df = pl.DataFrame({"x": [True, False, True], "y": [False, True, True]})
    data = _preprocess_upset(df, ["x", "y"])
    assert data.intersection_df.shape[0] == 3  # (T,F), (F,T), (T,T)
    assert data.set_sizes_df.filter(pl.col("set_name") == "x")["set_size"].item() == 2


# ── preprocessing: validation ─────────────────────────────────────────


def test_preprocess_invalid_column_type():
    df = pl.DataFrame({"a": [1, 0], "b": ["x", "y"]})
    with pytest.raises(TypeError, match="Column 'b' must be integer or boolean"):
        _preprocess_upset(df, ["a", "b"])


# ── integration: chart structure ──────────────────────────────────────


def test_plot_upset_basic(membership_df):
    chart = plot_upset(membership_df, SET_COLS_3, title="Basic UpSet")
    d = chart.to_dict()
    assert "hconcat" in d  # set sizes + main column
    assert d["title"] == "Basic UpSet"


def test_plot_upset_no_set_sizes(membership_df):
    chart = plot_upset(membership_df, SET_COLS_3, show_set_sizes=False)
    d = chart.to_dict()
    assert "vconcat" in d
    assert "hconcat" not in d


def test_plot_upset_custom_dims(membership_df):
    chart = plot_upset(membership_df, SET_COLS_3, width=800, height=400, show_set_sizes=False)
    d = chart.to_dict()
    bar = d["vconcat"][0]
    assert bar["width"] == 800
    assert bar["height"] == 400


def test_plot_upset_sort_multi_key(weighted_df):
    chart = plot_upset(weighted_df, ["A", "B", "C"], sort_by=["degree", "frequency"])
    d = chart.to_dict()
    assert "hconcat" in d


def test_plot_upset_with_filtering(weighted_df):
    chart = plot_upset(weighted_df, ["A", "B", "C"], min_degree=2, n_intersections=3)
    d = chart.to_dict()
    assert "hconcat" in d


def test_plot_upset_set_cols_subset(membership_df):
    """Using only 2 of 3 columns."""
    chart = plot_upset(membership_df, ["setA", "setB"])
    d = chart.to_dict()
    assert "hconcat" in d


def test_plot_upset_all_columns_default(membership_df):
    """set_cols=None uses every column in the DataFrame."""
    chart = plot_upset(membership_df)
    d = chart.to_dict()
    assert "hconcat" in d


def test_plot_upset_invalid_column_type_integration():
    df = pl.DataFrame({"a": [1, 0], "b": ["x", "y"]})
    with pytest.raises(TypeError, match="Column 'b' must be integer or boolean"):
        plot_upset(df, ["a", "b"])
