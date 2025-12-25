"""
fairness.pipeline
=================

A convenience "one-stop" workflow for demos and internal testing.

This module is intentionally **not** part of the core fairness-metric API.
It exists to help quickly:
1) load a dataset
2) apply optional fairness-oriented transforms (e.g., add_age_group)
3) preprocess to model-ready numeric features (one-hot encoding)
4) create a train/test split
5) train a simple classifier to produce y_pred
6) build group_dict (and/or intersectional labels) aligned to the test set

The fairness toolkit remains model-agnostic; any model can be used externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any, Dict, Hashable, Tuple

import pandas as pd

from fairness.data import load_csv, make_dataset_bundle
from fairness.preprocess import apply_transforms, preprocess_tabular, make_train_test_split, SplitData
from fairness.groups import create_intersectional_groups


@dataclass(frozen=True)
class PipelineResult:
    """Outputs from the demo pipeline.

    Attributes
    ----------
    df_raw:
        Raw loaded DataFrame.
    df_fair:
        DataFrame after fairness-oriented transforms (e.g., age binning).
    df_model:
        Model-ready numeric DataFrame (after one-hot encoding etc.).
    split:
        Train/test split container with X_train, X_test, y_train, y_test.
    model:
        Fitted model object (e.g., scikit-learn estimator).
    y_pred:
        Predictions for X_test (aligned with split.X_test and split.y_test).
    groups:
        Intersectional group label per test sample (aligned with y_pred).
    group_map:
        Mapping from label -> {attribute: value} (interpretability).
    counts:
        Group sizes for the test set.
    group_dict:
        Dict of protected attribute -> list of values per test sample (aligned with y_pred).
        This matches the input format used by some alternative fairness functions.
    """

    df_raw: pd.DataFrame
    df_fair: pd.DataFrame
    df_model: pd.DataFrame
    split: SplitData
    model: Any
    y_pred: Any
    groups: list[str]
    group_map: dict[str, dict[str, Hashable]]
    counts: pd.Series
    group_dict: Dict[str, list]


def run_demo_pipeline(
    *,
    csv_path: str,
    target_col: str,
    protected_cols: Sequence[str],
    fairness_transforms: Optional[Sequence[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
    drop_from_X: Sequence[str] = (),
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
    model: Optional[Any] = None,
    model_fit_kwargs: Optional[dict] = None,
    predict_proba: bool = False,
) -> PipelineResult:
    """Run an end-to-end demo workflow and return aligned outputs.

    Parameters
    ----------
    csv_path:
        Path to the input CSV.
    target_col:
        Name of the target column (binary 0/1 recommended for the demo).
    protected_cols:
        Names of protected attributes to build intersectional groups from.
        These columns must exist after fairness_transforms are applied.
    fairness_transforms:
        Optional list of DataFrame->DataFrame transforms applied *before* model preprocessing.
        Example: [add_age_group, lambda df: map_binary_column(df, ...)]
    drop_from_X:
        Columns to exclude from model features (in addition to target_col). Commonly includes
        fairness-only derived protected columns such as "age_group".
    test_size, random_state, stratify:
        Train/test split configuration.
    model:
        A scikit-learn style estimator implementing .fit() and .predict() (and optionally .predict_proba()).
        If None, uses LogisticRegression(max_iter=1000).
    model_fit_kwargs:
        Optional kwargs passed to model.fit(...).
    predict_proba:
        If True, returns probability of class 1 (model.predict_proba) instead of hard labels.

    Returns
    -------
    PipelineResult
        Container with all aligned objects needed by downstream fairness metrics.
    """
    df_raw = load_csv(csv_path)

    # 1) fairness-oriented transforms (optional)
    df_fair = df_raw
    if fairness_transforms:
        df_fair = apply_transforms(df_fair, fairness_transforms)

    missing = [c for c in protected_cols if c not in df_fair.columns]
    if missing:
        raise ValueError(f"Protected columns missing after transforms: {missing}")


    # 2) model-oriented preprocessing (one-hot etc.)
    df_model = preprocess_tabular(df_fair, drop_cols=drop_from_X)

    # 3) split for modelling
    split = make_train_test_split(
        df_model,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 4) fit model and predict
    if model is None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])

    fit_kwargs = model_fit_kwargs or {}
    model.fit(split.X_train, split.y_train, **fit_kwargs)

    if predict_proba:
        if not hasattr(model, "predict_proba"):
            raise ValueError("predict_proba=True but model has no predict_proba method")
        y_pred = model.predict_proba(split.X_test)[:, 1]
    else:
        y_pred = model.predict(split.X_test)

    # 5) protected attributes for *test set* in matching row order
    protected_test = df_fair.loc[split.X_test.index, list(protected_cols)]

    # group_dict format (per-attribute lists) for colleagues' functions
    group_dict = {col: protected_test[col].tolist() for col in protected_cols}

    # also compute intersectional string labels (your toolkit format)
    groups, group_map, counts = create_intersectional_groups(protected_test, protected=protected_cols)

    # defensive alignment check
    if len(groups) != len(split.X_test) or len(groups) != len(split.y_test):
        raise ValueError("Alignment error: groups, X_test, and y_test lengths differ")

    return PipelineResult(
        df_raw=df_raw,
        df_fair=df_fair,
        df_model=df_model,
        split=split,
        model=model,
        y_pred=y_pred,
        groups=groups,
        group_map=group_map,
        counts=counts,
        group_dict=group_dict,
    )
