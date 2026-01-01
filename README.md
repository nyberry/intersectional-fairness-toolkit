# Intersectional Fairness Toolkit for Health Machine Learning

This project provides a Python package for evaluating intersectional fairness in machine learning models applied to tabular health datasets. It supports the construction of intersectional protected groups (e.g. sex × age group) and the computation of fairness metrics such as Differential Fairness.

The toolkit is designed to be dataset-agnostic and model-agnostic, making it suitable for use in a wide range of health data science workflows.


## Fairness

Fairness in machine learning refers to the ethical requirement that models do not produce systematically biased or discriminatory outcomes for individuals or groups. In the context of health data science, unfair models may lead to unequal access to diagnosis, treatment, or follow-up.


Bias can arise when model predictions differ across protected attributes such as sex, age, ethnicity, or disability status. A growing number of tools exist to help researchers and practitioners evaluate fairness in machine learning models; however, many focus on single protected attributes in isolation.


## Intersectional Fairness

This package provides tools which allow researchers to evaluate fairness across intersections of protected attributes, rather than considering each attribute independently.

For example, instead of checking:
- men vs women
- younger vs older patients

We check:
- young women
- older women
- young men
- older men

This approach is motivated by the observation that unfairness can be hidden when outcomes are averaged over broad groups. Disparities often emerge at the intersections of attributes, where individuals may experience compounded or 'double' disadvantage. For instance, older women may be treated less favourably than either women or older patients considered as marginal groups alone.

In this package, intersectional groups are evaluated using differential fairness metrics to quantify worst-case disparities in model outcomes.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Raiet-Bekirov/HPDM139_assignment.git
cd HPDM139_assignment
pip install -e .
```

Alternatively, install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/intersectional-fairness-toolkit==0.1.0
```


## Example usage

The example below demonstrates a typical workflow:
loading a clinical dataset, training a simple classifier, and evaluating
intersectional fairness metrics. The toolkit is model-agnostic and can be used
with any scikit-learn–compatible estimator.

```python
from fairness.data import load_heart_csv
from fairness.preprocess import add_age_group, preprocess_tabular, make_train_test_split
from fairness.groups import make_eval_df
from fairness.adapters import unpack_eval_df
from fairness.metrics import group_acc_ratio 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) Load dataset and add protected attributes
df = load_heart_csv("data/heart.csv")
df = add_age_group(df)

# 2) Prepare features for modelling
df_model = preprocess_tabular(df)

# 3) Train/test split
split = make_train_test_split(df_model, target_col="HeartDisease", stratify=True)

# 4) Train a simple model (example)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])
model.fit(split.X_train, split.y_train)
y_pred = model.predict(split.X_test)

# 5) Align predictions, labels, and protected attributes
df_test = df.loc[split.X_test.index] 
eval_df = make_eval_df(
    df_test=df_test,
    protected=["Sex", "age_group"],
    y_pred=y_pred,
    y_true=split.y_test.to_numpy(),
)

# 6) Compute intersectional fairness metric

subject_labels, predictions, true_statuses = unpack_eval_df(eval_df)

acc = group_acc_ratio(
    "Sex=0|age_group=older",
    "Sex=1|age_group=older",
    subject_labels,
    predictions,
    true_statuses,
    natural_log=True
)
print("Accuracy ratio:", acc)
```

A complete end-to-end example using the UCI Heart Disease dataset is provided at [`examples/uci_heart_demo.ipynb`](examples/uci_heart_demo.ipynb)

## Documentation

Additional documentation is available [here](https://raiet-bekirov.github.io/HPDM139_assignment/), including:

- [Tutorial](https://raiet-bekirov.github.io/HPDM139_assignment/tutorial.md) - a step-by-step workflow explanation
- [API reference](https://raiet-bekirov.github.io/HPDM139_assignment/api_reference.md) - documentation of each function
- [Design decisions](https://raiet-bekirov.github.io/HPDM139_assignment/design_decisions.md) - rationale behind design choices



## Project context

This package was developed as part of the HPDM139 module (Health Data Science) at the University of Exeter.

Agendas and minutes of team meetings are in the [Team portfolio](https://raiet-bekirov.github.io/HPDM139_assignment/team_portfolio/team_portfolio/contents.md) 


## License

Apache-2.0 license
