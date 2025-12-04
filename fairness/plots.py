import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_group_fairness(y_pred: pd.Series, groups: pd.Series):
    """
    Plot the positive prediction rate (P(y_pred=1 | group)) for each 
    intersectional group.

    This visualization helps identify which groups the model treats differently,
    and complements the Differential Fairness (ε) summary statistic.

    Parameters
    ----------
    y_pred : pandas.Series or array-like
        Predicted binary labels (0 or 1). Must be same length as `groups`.

    groups : pandas.Series or array-like
        Intersectional group labels for each row (e.g., "sex=1|age=older").
        Typically generated via create_intersectional_groups().

    Returns
    -------
    None
        Displays a matplotlib bar chart.

    Raises
    ------
    ValueError
        If lengths mismatch or predictions are not binary.
    """

    # ---- Convert to Series ----
    y_pred = pd.Series(y_pred)
    groups = pd.Series(groups)

    # ---- Validation ----
    if len(y_pred) != len(groups):
        raise ValueError("y_pred and groups must have the same length.")

    if not set(y_pred.unique()).issubset({0, 1}):
        raise ValueError("y_pred must contain only 0 or 1 binary values.")

    # ---- Compute positive prediction rates ----
    df = pd.DataFrame({"y_pred": y_pred, "group": groups})

    group_rates = df.groupby("group")["y_pred"].mean().sort_values()

    # ---- Plot ----
    plt.figure(figsize=(10, 5))
    group_rates.plot(kind="bar", color="steelblue", edgecolor="black")

    plt.title("Positive Prediction Rate by Intersectional Group", fontsize=14)
    plt.xlabel("Intersectional Group", fontsize=12)
    plt.ylabel("P(y_pred = 1 | group)", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
