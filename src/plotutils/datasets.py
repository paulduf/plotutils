"""Reusable toy datasets for demos and documentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Dataset:
    """Container returned by every ``load_*`` helper."""

    df: pl.DataFrame
    variables: list[str]
    outcomes: list[str]


def mask_missing_data(
    df: pl.DataFrame,
    *,
    pattern: str = "mcar",
    columns: list[str],
    seed: int = 7,
) -> pl.DataFrame:
    """Introduce missing values into *columns* of *df*.

    Parameters
    ----------
    df:
        Input dataframe (not mutated).
    pattern:
        Missing-data mechanism.  Only ``"mcar"`` (Missing Completely At
        Random) is currently supported.
    columns:
        Column names to mask.
    seed:
        Random seed for reproducibility.
    """
    if pattern != "mcar":
        raise ValueError(f"Unsupported missing-data pattern: {pattern!r}")

    rng = np.random.default_rng(seed)
    n = len(df)

    for col in columns:
        n_miss = int(rng.integers(5, 26))
        miss_idx = rng.choice(n, size=n_miss, replace=False)
        vals = df[col].to_list()
        for i in miss_idx:
            vals[i] = None
        df = df.with_columns(pl.Series(col, vals, dtype=df[col].dtype))

    return df


def load_synthetic(*, missing: bool = False, seed: int = 42) -> Dataset:
    """Five synthetic biomarker scores with three binary outcomes.

    Parameters
    ----------
    missing:
        If *True*, randomly mask values in all variable and outcome columns.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = 400

    variables = [f"var_{i}" for i in range(5)]
    outcomes = [f"outcome_{i}" for i in range(3)]

    labels = {out: rng.integers(0, 2, size=n) for out in outcomes}
    strengths = rng.uniform(0.3, 1.5, size=(len(variables), len(outcomes)))

    data: dict = {
        "patient_id": [f"P{i + 1:03d}" for i in range(n)],
        **{out: lbl.tolist() for out, lbl in labels.items()},
    }
    for i, var in enumerate(variables):
        score = sum(strengths[i, j] * labels[out] for j, out in enumerate(outcomes))
        score += rng.normal(0, 1.0, size=n)
        data[var] = score.tolist()

    df = pl.DataFrame(data)

    if missing:
        df = mask_missing_data(
            df, pattern="mcar", columns=variables + outcomes,
        )

    return Dataset(df=df, variables=variables, outcomes=outcomes)


def load_binary_diabetes(*, missing: bool = False) -> Dataset:
    """Sklearn diabetes dataset with binarised progression outcomes.

    The continuous progression target is binarised at three quartile
    thresholds (Q25, Q50, Q75).

    Parameters
    ----------
    missing:
        If *True*, randomly mask values in all variable and outcome columns.
    """
    from sklearn.datasets import load_diabetes  # type: ignore[import-untyped]

    X, y = load_diabetes(return_X_y=True)
    variables = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    outcomes = ["prog_q75", "prog_q50", "prog_q25"]

    data: dict = {
        "patient_id": [f"P{i + 1:03d}" for i in range(len(y))],
        "prog_q75": (y > q75).astype(int).tolist(),
        "prog_q50": (y > q50).astype(int).tolist(),
        "prog_q25": (y > q25).astype(int).tolist(),
    }
    for j, var in enumerate(variables):
        data[var] = X[:, j].tolist()

    df = pl.DataFrame(data)

    if missing:
        df = mask_missing_data(
            df, pattern="mcar", columns=variables + outcomes,
        )

    return Dataset(df=df, variables=variables, outcomes=outcomes)
