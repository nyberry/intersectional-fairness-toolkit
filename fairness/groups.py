"""
fairness.groups
===============

Utilities for constructing protected groups and intersectional group labels.

This module creates intersectional group labels, group count summaries, and small-group warnings.

Typical usage
-------------
>>> protected = ["Sex", "age_group"]
>>> groups, group_map, counts = create_intersectional_groups(df.loc[X_test.index], protected)
>>> counts
Sex=1|age_group=young    127
Sex=1|age_group=older    103
Sex=0|age_group=older     26
Sex=0|age_group=young     20
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class GroupingResult:
    """
    Result container for intersectional grouping.

    Attributes
    ----------
    groups:
        List of group labels aligned with the input DataFrame rows.
    group_map:
        Mapping from label -> {attribute: value} for interpretability.
    counts:
        Series of group sizes (index = label, value = count).
    protected_cols:
        The protected columns used to build the group labels.
    """
    groups: list[str]
    group_map: dict[str, dict[str, Hashable]]
    counts: pd.Series
    protected_cols: tuple[str, ...]


def validate_protected_columns(df: pd.DataFrame, protected: Sequence[str]) -> None:
    """
    Validate that the requested protected columns exist in the DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    protected:
        List of protected column names.

    Raises
    ------
    ValueError
        If protected is empty or columns are missing.
    """
    if not protected:
        raise ValueError("protected must be a non-empty list of column names")

    missing = [c for c in protected if c not in df.columns]
    if missing:
        raise ValueError(f"Protected columns not found: {missing}")


def _normalise_value(val: object, *, missing: str = "NA") -> Hashable:
    """
    Normalise values used in group labels.

    - Converts NaN/None to a sentinel string (default: 'NA')
    - Leaves other values unchanged

    Parameters
    ----------
    val:
        Input value from the DataFrame.
    missing:
        Replacement used when val is missing.

    Returns
    -------
    Hashable
        Normalised value suitable for label creation.
    """
    if pd.isna(val):
        return missing
    return val  # type: ignore[return-value]


def create_group_label(
    row: pd.Series,
    protected: Sequence[str],
    *,
    sep: str = "|",
    kv_sep: str = "=",
    missing: str = "NA",
) -> str:
    """
    Create a single intersectional group label for a row.

    Example:
        Sex=1|age_group=older

    Parameters
    ----------
    row:
        A Series containing at least the protected columns.
    protected:
        Ordered list of protected column names.
    sep:
        Separator between attributes.
    kv_sep:
        Separator between key and value.
    missing:
        Placeholder used for missing values.

    Returns
    -------
    str
        Intersectional group label.
    """
    parts: list[str] = []
    for col in protected:
        val = _normalise_value(row[col], missing=missing)
        parts.append(f"{col}{kv_sep}{val}")
    return sep.join(parts)


def create_intersectional_groups(
    df: pd.DataFrame,
    protected: Sequence[str],
    *,
    sep: str = "|",
    kv_sep: str = "=",
    missing: str = "NA",
    sort_counts: bool = True,
) -> Tuple[list[str], dict[str, dict[str, Hashable]], pd.Series]:
    """
    Create intersectional group labels from protected attributes.

    Parameters
    ----------
    df:
        DataFrame containing protected columns.
    protected:
        Column names to intersect (e.g., ["Sex", "age_group"]).
    sep:
        Separator between attributes in labels.
    kv_sep:
        Separator between key and value in labels.
    missing:
        Placeholder for missing values.
    sort_counts:
        If True, counts are returned sorted descending.

    Returns
    -------
    groups:
        List of group labels aligned with df rows.
    group_map:
        Mapping label -> {attribute: value} for interpretability.
    counts:
        Group sizes as a pandas Series.

    Notes
    -----
    Alignment is preserved: `groups[i]` corresponds to the i-th row of df.
    When used with `df.loc[X_test.index]`, this guarantees alignment with y_pred.
    """
    validate_protected_columns(df, protected)

    protected = tuple(protected)
    groups: list[str] = []
    group_map: dict[str, dict[str, Hashable]] = {}

    # Build labels row-wise to preserve ordering/alignment
    for _, row in df[list(protected)].iterrows():
        label = create_group_label(row, protected, sep=sep, kv_sep=kv_sep, missing=missing)
        groups.append(label)

        if label not in group_map:
            mapping = {col: _normalise_value(row[col], missing=missing) for col in protected}
            group_map[label] = mapping

    counts = pd.Series(groups, name="group").value_counts()
    if not sort_counts:
        # Preserve first-seen order rather than frequency order
        counts = counts.reindex(pd.Index(dict.fromkeys(groups).keys()))

    return groups, group_map, counts


def warn_small_groups(
    counts: pd.Series,
    *,
    min_size: int = 20,
) -> Optional[str]:
    """
    Generate a warning message if any intersectional group has fewer than min_size samples.

    Parameters
    ----------
    counts:
        Group size Series (as returned by create_intersectional_groups()).
    min_size:
        Minimum recommended sample size per group.

    Returns
    -------
    Optional[str]
        Warning message string if small groups exist, otherwise None.

    Notes
    -----
    Small groups can produce unstable fairness estimates (including DF epsilon),
    even with smoothing. Users may wish to:
    - reduce the number of protected attributes
    - merge rare categories
    - use stronger smoothing
    """
    small = counts[counts < min_size]
    if small.empty:
        return None

    items = ", ".join([f"{idx} (n={int(n)})" for idx, n in small.items()])
    return f"Small intersectional groups detected (<{min_size}): {items}"
