# Tutorial

This toolkit helps you check whether a machine-learning model treats different groups of people fairly when working with health data stored in tables (for example, CSV files).

It lets you:

- combine protected characteristics such as sex and age to form intersectional groups (for example, female and over 55),
- calculate fairness metrics that show how model performance differs between those groups.

The toolkit is not tied to any specific dataset or model. You can use it with different health datasets and with any machine-learning model that produces predictions.

---

In this tutorial we will perform an end-to-end intersectional fairness evaluation on a health dataset.

We will:

1. Install the package
2. Load a real clinical dataset (heart.csv)
3. Prepare protected attributes and intersectional groups
4. Train a simple machine-learning model
5. Build an evaluation table for fairness analysis
6. Evaluate fairness using three key metrics
7. Visualise fairness using three key plots

No prior fairness expertise is required.

## 1. Installation

It is recommended to install the toolkit in a clean Python environment to avoid
dependency conflicts.

### Option A: Using conda

```bash
conda create -n fairness python=3.12
conda activate fairness
pip install intersectional-fairness-toolkit
```

### Option B: Using `venv`

On MacOS/ Linux:

```bash
python -m venv fairness-env
source fairness-env/bin/activate 
```

On Windows:
```bash
python -m venv fairness-env
fairness-env\Scripts\activate 
```

Then, install the toolkit:

```bash
pip install intersectional-fairness-toolkit
```


## 2. Load a clinical dataset

For this tutorial, we will use the `heart failure prediction dataset`. This dataset contains 11 features that can be used to predict possible heart disease.

Download the dataset:

```python
import pandas as pd
from fairness.data import load_heart_csv

url = "https://raw.githubusercontent.com/Raiet-Bekirov/HPDM139_assignment/main/data/heart.csv"
df = load_heart_csv(url)
```

This dataset includes:

- clinical features (age, cholesterol, blood pressure, etc.)
- a binary outcome (HeartDisease)
- protected attributes such as Sex and Age

## 3. Add protected attributes and prepare features for modelling


```python
from fairness.preprocess import add_age_group, preprocess_tabular, make_train_test_split

df = add_age_group(df)
df_model = preprocess_tabular(df)
split = make_train_test_split(
    df_model,
    target_col="HeartDisease",
    stratify=True
)
```

The `add_age_group` function creates an `age_group` column by binning age into broad, clinically relevant categories (for example: under 55, and 55+).

This gives us two protected attributes for fairness analysis:
`Sex` (already present in the dataset), and `Age group` (derived from the Age column)

The `preprocess_tabular` function prepares the dataset so it can be used by a machine-learning model. It cleans up the data and converts it into a format the model can understand, for example by turning categories into numbers and making sure everything is laid out consistently.

The `make_train_test_split` function then divides the data into two parts:

- a training set, which the model learns from, and
- a test set, which is used to check how well the model performs on new, unseen data.

It returns these as `X_train`, `X_test`, `y_train`, and `y_test`, which are ready to be passed into a machine-learning model.

## 4. Train a machine learning model

The toolkit is designed to be model agnostic. Here we train a simple model using logistic regression.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(split.X_train, split.y_train)
y_pred = model.predict(split.X_test)

```

The model pipeline includes:

- feature scaling (to ensure numerical stability),
- a logistic regression classifier.

The trained model is then used to generate predictions on the test set.

## 5. Build an evaluation table for fairness analysis

```python
from fairness.groups import make_eval_df

df_test = df.loc[split.X_test.index]

eval_df = make_eval_df(
    df_test=df_test,
    protected=["Sex", "age_group"],
    y_pred=y_pred,
    y_true=split.y_test.to_numpy(),
)
```

The `make_eval_df` function combines:

- protected attributes
- model predictions
- true outcomes

into a single evaluation table (`eval_df`).

Each row represents one individual in the test set, with all information required for intersectional fairness analysis.


## 6. Fairness Metrics

Many fairness metrics operate on lists, for example:

- a list of group labels
- a list of predictions
- a list of true outcomes

These can be extracted from `eval_df` using helper functions in `adapters.py`:

```python
from fairness.adapters import unpack_eval_df, make_subject_labels_dict

subject_labels, predictions, true_statuses = unpack_eval_df(eval_df)

subject_labels_dict = make_subject_labels_dict(
    df_test,
    protected_cols=["Sex", "age_group"]
)
```

### 6a. Intersectional Accuracy

This asks the question: does the model performance differ accross intersectional groups?

```python
from fairness.metrics import (
    all_intersect_accs,
    max_intersect_acc_diff,
    max_intersect_acc_ratio,
)

accs = all_intersect_accs(subject_labels_dict, predictions, true_statuses)
accs
```

### 6b. Maximum accuracy difference across intersectional groups

This asks, what is the performance gap between best and worst groups?


```python

max_gap = max_intersect_acc_diff(
    subject_labels_dict=subject_labels_dict,
    predictions=predictions,
    true_statuses=true_statuses
)

max_gap
```

### 6c. Maximum accuracy ratio across intersectional groups

This asks, how many times better does the best group perform compared to the worst group?

```python
max_ratio_log = max_intersect_acc_ratio(
    subject_labels_dict=subject_labels_dict,
    predictions=predictions,
    true_statuses=true_statuses,
    natural_log=True
)

max_ratio_log
```

---

These are three examples of fairness metrics which can be computed using this toolkit. A full list of metrics functions provided by the toolkit is available at `api_reference.md`

## 7. Fairness Visualisations 

Visualisations can make fairness issues interpretable.

### 7a. Accuracy by Intersectional Group

```python
from fairness.visualisation import plot_group_accuracies

plot_group_accuracies(accs)
```

This visualisation highlights groups with systematically poorer performance.

### 7b. Group Size vs Performance

```python
from fairness.visualisation import plot_group_size_vs_accuracy

plot_group_size_vs_accuracy(accs, subject_labels_dict)
```

This visualisation shows whether poor performance may be driven by small sample sizes.

### 7c. Fairness Summary Plot

```python
from fairness.visualisation import plot_fairness_summary

plot_fairness_summary(accs)
```

This visualisation provides an overview suitable for reports and presentations.

--- 

These are three examples of fairness visualisations which can be produced using this toolkit. A full list of visualisation functions provided by the toolkit is available at `api_reference.md`

## Interpreting the Results

When looking at the fairness results, it helps to ask a few simple questions:

- Do some combinations of characteristics (for example, older women or younger men) get worse results than others?
- Are the differences big enough to matter in a real clinical setting, rather than just being statistically different?
- Could some poor results be explained by very small numbers of people in certain groups?

If fairness issues are identified, possible ways to address them include:

- collecting more data for under-represented groups,
- changing the decision threshold used by the model,
- trying a different type of model that behaves more consistently across groups.

## Summary

In this tutorial we demonstrated how to:

- Install and use the toolkit
- Load a real clinical dataset
- Create intersectional protected groups
- Train a simple ML model
- Evaluate fairness using metrics
- Visualise disparities using plots