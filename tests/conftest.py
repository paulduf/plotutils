import altair as alt
import pytest
import vl_convert as vlc
from syrupy.extensions.image import SVGImageSnapshotExtension


@pytest.fixture(autouse=True)
def _reset_altair_theme():
    """Reset Altair theme to default before each test for deterministic SVG rendering.

    Without this, test_theme.py enables a global theme that leaks into subsequent
    tests, making SVG output depend on test execution order.
    """
    alt.theme.enable("default")
    yield
    alt.theme.enable("default")


@pytest.fixture
def report_theme():
    """Enable the custom report theme for tests that produce doc SVGs."""
    import plotutils.themes  # noqa: F401 — registers the theme
    alt.theme.enable("report_theme")
    yield
    alt.theme.enable("default")


@pytest.fixture
def snapshot_svg(snapshot):
    """Syrupy snapshot fixture configured for SVG files.

    Usage:
        def test_my_chart(snapshot_svg):
            svg_content = chart_to_svg(my_chart)
            assert svg_content == snapshot_svg
    """
    return snapshot.use_extension(SVGImageSnapshotExtension)


def normalize_chart_dict(chart_dict: dict) -> dict:
    """Normalize chart dict by removing non-deterministic data name hashes.

    Altair generates hash-based data names that change between runs.
    This function replaces them with stable names for snapshot testing.
    """
    normalized = chart_dict.copy()

    if "datasets" in normalized:
        new_datasets = {}
        name_mapping = {}
        for i, key in enumerate(sorted(normalized["datasets"].keys())):
            new_name = f"data-normalized-{i}"
            name_mapping[key] = new_name
            dataset = normalized["datasets"][key]
            if isinstance(dataset, list) and dataset and isinstance(dataset[0], dict):
                dataset = sorted(dataset, key=lambda x: tuple(sorted(x.items())))
            new_datasets[new_name] = dataset
        normalized["datasets"] = new_datasets

        def update_data_refs(obj, mapping):
            if isinstance(obj, dict):
                return {
                    k: mapping[v] if k == "name" and v in mapping else update_data_refs(v, mapping)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [update_data_refs(item, mapping) for item in obj]
            return obj

        normalized = update_data_refs(normalized, name_mapping)

    return normalized


def chart_to_svg(chart) -> str:
    """Convert an Altair chart to SVG string.

    Note: SVG snapshot testing only works for charts with deterministic rendering.
    Charts using mark_errorbar with extent="ci" (the default) are non-deterministic
    because Vega computes confidence intervals via bootstrap resampling with an
    unseeded RNG. This produces ~1-2px coordinate jitter between calls, even within
    the same process. For those charts, test only the Vega-Lite dict snapshot.
    """
    return vlc.vegalite_to_svg(chart.to_dict())
