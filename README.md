# HPDM139_assignment

## Data Loading and processing

This module provides dataset-agnostic utilities for loading, preprocessings and preparing tabular health datasets for fairness analysis.

This module provides the front-end of the fairness workflow, producing clean, aligned inputs for fairness metrics and visualisation components implemented elsewhere in the package.

It is designed to be Dataset-agnostic: works with multiple datasets (eg UCI Heart Disease), and Model-agnostic: produces outputs suitable for any classifier


## Overview of modules

'data.py'

Handles dataset loading

'preprocess.py'

Handles preprocessing and train / test splitting

'groups.py'

creates intersectional group labels from protected attributes

Together, these modules prepare the two inputs required by fairness metrics:

(test individual i)
 ├─ groups[i]      → protected group
 ├─ y_pred[i]      → model decision


## Loading a dataset

'''python
from data import load_csv

df = load_csv("data/heart.csv")
'''

(dataset-specific loaders could be added as wrappers around this function as needed)

## Preprocessing

Any protected attributes which are *continuous* should be made discrete before fairness analysis

'''python

from preprocess import add_age_group

df = add_age_group(df)
'''

This creates a categorical variable (young, old) suitable for intersectional grouping

### preparing data for machine learning

'''python
from preprocess import preprocess_tabular

X = preprocess_tabular(df)
'''

This fuction converts categorical columns to numeri using one-hot encoding. It leaves numeric features unchanges.

It produces a dataframe compatible with scikit-learn

### train / test split

'''python
from preprocess import make_train_test_split

split = make_train_test_split(
    df,
    target_col="HeartDisease",
    drop_cols=("age_group",),
    test_size=0.3,
    random_state=42
)
'''

Protected attributes derived and used only for fairness analysis (sych as age_group) are dropeed from model inputs

outputs are:
X_train, X_test
y_train, y_test

### Creating intersectional groups

These groups are constructed from protected attributes such as sex and age group

'''python
from groups import create_intersectional_groups

groups, group_map, counts = create_intersectional_groups(
    df.loc[split.X_test.index],
    protected=["Sex", "age_group"]
)
'''

### Outputs

- 'groups' is a list of readable labels for the intersectional groups. Example:

'''
Sex=1|age_group=old
'''
- 'group_map' maps from group labels to attribute values
- 'counts' gives the sample size per intersectional group

### Guarnanteeong alignment

the protected attributes used for fairness analysis are selected using

'''python
df.loc[X_test.index]
'''
this ensures that groups[i] refers to the same indivisual as y_pred[i]

## Typical usage

'''python
# Load and preprocess
df = load_csv("data/heart.csv")
df = add_age_group(df)
df = preprocess_tabular(df)

# Split
split = make_train_test_split(df)

# Train model (an example)
model.fit(split.X_train, split.y_train)
y_pred = model.predict(split.X_test)

# Create intersectional groups
protected_test = df.loc[split.X_test.index]
groups, _, counts = create_intersectional_groups(
    protected_test,
    ["Sex", "age_group"]
)

# Pass to fairness metric
epsilon, probs = differential_fairness(y_pred, groups)
'''

### convenient helper function

“For convenience, fairness.pipeline.prepare_fairness_inputs() bundles loading/preprocessing/group creation into a single call.”

### Supported datasets


The pipeline has been designed to support any tabular dataset with:

- a clearly defined target variable
- one or more protected attributes

The dataset used for demonstration is UCI Heart Disease

Dataset-specific logic can be implemented with small adapters, not inside the core pipeline.