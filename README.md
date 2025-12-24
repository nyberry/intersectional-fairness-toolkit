# HPDM139_assignment

## Data Loading and processing module

This module provides utilities for loading and preparing tabular health datasets for fairness analysis.

It is designed to be dataset-agnostic, meaning it can work with different datasets. It is also model-agnostic, meaning the outputs produced should be suitable for any classifier.


## Overview of module


`data.py` handles dataset loading.

`preprocess.py` handles preprocessing, and train / test splitting.

`groups.py` creates intersectional group labels from protected attributes

Together, these functions take the raw dataset, process it, and supply the protected-group inputs needed for fairness metrics functions.


## Loading a dataset


```python
from data import load_csv
df = load_csv("fairness/data/heart.csv")
```

Dataset-specific loaders are added as wrappers around this function as needed.

## Preprocessing

Preprocessing prepares a raw dataset for both model training and fairness analysis.

There are 3 steps:

### 1. Fairness-oriented preprocessing

Before fairness analysis, the dataset must be prepared so that protected attributes can be meaningfully compared across groups.

In particular, continuous protected attributes (such as age) should be converted into discrete categories. Using raw continuous values would create a large number of tiny groups (e.g. age = 47 vs age = 48), making group-level fairness comparisons unreliable and difficult to interpret

For example, to discretise age:

```python
from preprocess import add_age_group
df = add_age_group(df)
```

This creates a categorical variable (young, old) suitable for intersectional grouping.

In summary, this preprocessing step takes as input a pandas DataFrame containing a continuous age column (eg. `age`), and bins the age values into a smaller number of categories. It outputs a new dataframe that retains all original columns, and adds a new categorical colunm (eg. `age_group`)


### 2. Model-oriented preprocessing

Here the data is prepared for machine learning.

```python
from preprocess import preprocess_tabular
df_model = preprocess_tabular(df)
```

The output of this step, `df_model`, is a pandas DataFrame containing only numeric features. Categorical variables are converted into binary indicator columns using one-hot encoding, while numeric features are left unchanged.

This DataFrame is compatible with scikit-learn, which expects a 2-dimensional array-like input of shape (n_samples, n_features).


### 3. Data partitioning (train / test split)

The pre-processed dataset is split into seperate training and test sets. This ensures that the model is trained on one subset of the data and evaluated on a different, unseen subset.

```python
from preprocess import make_train_test_split

split = make_train_test_split(
    df_model,
    target_col="HeartDisease",
    drop_cols=("age_group",),
    test_size=0.3,
    random_state=42
)
```
The column specified by target_col (e.g. `HeartDisease`) is extracted as the outcome variable `y`.

Protected attributes derived and used only for fairness analysis (sych as age_group) are dropeed from model inputs, ensuring that the model does not directly use these derived attributes while making predictions.

A proportion of the data (e.g. 30%) is held as a test set, and the split is reproducible due to the fixed random_state.

The function returns a container with four aligned components:
- `X_train`, the feature matrix used to train a model
- `X_test`, the feature matrix used to generate predictions
- `y_train`, True outcome labels corresponding to X_train
- `y_test`, True outcome labels corresponding to X_test 

Row indices are preserved across all outputs.

## Creating intersectional groups

Fairness analysis often requires examining how a model behaves not only across individual protected attributes (such as sex or age), but across their intersections. For example, a model may behave differently for older women than for younger men, even if average performance by sex alone appears acceptable.

Intersectional groups are constructed by combining multiple protected attributes into a single group label for each individual.

```python
from groups import create_intersectional_groups

groups, group_map, counts = create_intersectional_groups(
    df.loc[split.X_test.index],
    protected=["Sex", "age_group"]
)
```

The DataFrame is indexed using split.X_test.index to ensure that only individuals in the test set are considered, and that the group labels are aligned with the model’s predictions.

For each test-set individual, a label is constructed by combining the specified protected attributes.

eg. `Sex=1|age_group=older`

The function also counts how many individuals belong to each intersectional group, which is important for interpreting fairness metrics and identifying small groups.

The outputs of this function are:

`groups`, a list of intersectional group labels, with one label per individual in the test set.
The order of this list matches the order of the test-set rows used to generate model predictions.

This means that for each test-set individual i:

```
groups[i], the protected group of individual i
y_pred[i], the model’s prediction for individual i
y_test[i], the true outcome for individual i
```

`group_map`, a mapping from each group label to the underlying protected attribute values.

`counts`, a summary of how many individuals in the test set belong to each intersectional group.
