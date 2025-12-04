import pandas as pd

def create_intersectional_groups(
    df: pd.DataFrame, 
    protected_attributes: list[str]
):
    """
    Create intersectional group labels from one or more protected attributes.

    This function combines the specified protected attribute columns into a 
    single human-readable group label for each row (e.g. "sex=1|age_group=older").
    These labels can then be used to analyse fairness across intersectional groups.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset. Must contain all columns listed in protected_attributes.

    protected_attributes : list of str
        The column names representing protected characteristics, such as 
        ["sex", "age_group"]. These columns should exist in the dataframe.

    Returns
    -------
    group_labels : pandas.Series
        A Series (length = len(df)) where each entry is a string label 
        identifying the intersectional group for that row.

    group_map : dict
        A dictionary mapping each unique group label to a list of row indices 
        belonging to that group. Example:
            {
                "sex=1|age_group=older": [3, 7, 12, ...],
                "sex=0|age_group=young": [0, 4, 9, ...]
            }

    counts : pandas.Series
        The number of samples (rows) in each intersectional group.

    Raises
    ------
    ValueError
        If protected_attributes is empty.

    KeyError
        If any of the specified protected attributes do not exist in df.
    """

    # ---- Validation ----
    if not protected_attributes:
        raise ValueError("protected_attributes list cannot be empty.")

    missing_cols = [col for col in protected_attributes if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following protected attributes are missing from the dataframe: {missing_cols}"
        )

    # ---- Create human-readable intersectional group labels ----
    # Example output: "sex=1|age_group=older"
    group_labels = df[protected_attributes].apply(
        lambda row: "|".join([f"{col}={row[col]}" for col in protected_attributes]),
        axis=1
    )

    # ---- Create group → row index mapping ----
    group_map = {
        group: df.index[group_labels == group].tolist()
        for group in group_labels.unique()
    }

    # ---- Count how many individuals are in each group ----
    counts = group_labels.value_counts()

    return group_labels, group_map, counts
