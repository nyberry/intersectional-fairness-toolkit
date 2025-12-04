import numpy as np
import pandas as pd

def differential_fairness(
    y_pred: pd.Series,
    groups: pd.Series,
    prior: float = 1e-6
) -> float:
    """
    Compute the empirical Differential Fairness (ε) value using hard counts.

    Differential Fairness measures the maximum difference in how often different
    intersectional groups receive positive predictions from a model.
    Lower ε indicates fairer behaviour, while larger ε indicates greater disparity.

    Parameters
    ----------
    y_pred : pandas.Series or array-like
        Predicted binary labels (0 or 1). Must be the same length as `groups`.

    groups : pandas.Series
        Intersectional group labels for each individual. Returned from 
        `create_intersectional_groups`.

    prior : float, optional (default = 1e-6)
        Small smoothing constant added to avoid zero probabilities and log(0)
        issues, especially for small groups.

    Returns
    -------
    epsilon : float
        The differential fairness value ε. 
        Interpretation: exp(ε) approximates the maximum ratio of positive
        prediction rates between any two groups.

    Raises
    ------
    ValueError
        If inputs have different lengths or y_pred contains values other than 0/1.

    Notes
    -----
    This implements the empirical DF formulation:
        ε = max_{i,j} | log(P(y=1|s_i)) - log(P(y=1|s_j)) |
    """

    # ---- Input Validation ----
    y_pred = pd.Series(y_pred)
    groups = pd.Series(groups)

    if len(y_pred) != len(groups):
        raise ValueError("y_pred and groups must have the same length.")

    if not set(y_pred.unique()).issubset({0, 1}):
        raise ValueError("y_pred must contain only binary values: 0 or 1.")

    # ---- Compute P(y=1 | group) with smoothing ----
    df = pd.DataFrame({"y_pred": y_pred, "group": groups})

    group_counts = df.groupby("group")["y_pred"].count()
    positive_counts = df.groupby("group")["y_pred"].sum()

    # Smoothed probabilities
    probs = (positive_counts + prior) / (group_counts + 2 * prior)

    # If only one group exists → perfectly fair
    if len(probs) <= 1:
        return 0.0

    # ---- Compute all log differences ----
    log_probs = np.log(probs.astype(float).values)


    # Compute max absolute difference between all pairs (i,j)
    # ε = max_{i,j} | log(p_i) - log(p_j) |
    epsilon = float(
        np.max(np.abs(log_probs[:, None] - log_probs[None, :]))
    )

    return epsilon
