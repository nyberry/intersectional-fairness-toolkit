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


## Example usage

```python
from fairness.data import load_heart_csv
from fairness.preprocess import add_age_group, preprocess_tabular, make_train_test_split
from fairness.groups import make_eval_df
from fairness.metrics.metrics import group_accuracy 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) Load + fairness-oriented preprocessing
df = load_heart_csv("data/heart.csv")
df = add_age_group(df)

# 2) Preprocessing for ML
df_model = preprocess_tabular(df)

# 3) Train/test split
split = make_train_test_split(df_model, target_col="HeartDisease", stratify=True)

# 4) Train a simple model (example only; toolkit is model-agnostic)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])
model.fit(split.X_train, split.y_train)
y_pred = model.predict(split.X_test)

# 5) Build aligned evaluation DataFrame for fairness metrics
df_test = df.loc[split.X_test.index] 
eval_df = make_eval_df(
    df_test=df_test,
    protected=["Sex", "age_group"],
    y_pred=y_pred,
    y_true=split.y_test.to_numpy(),
)

# 6) Example group metric: accuracy for older females
older_female = "Sex=0|age_group=older"
acc = group_accuracy(older_female, eval_df)
print("Accuracy (older female):", round(acc, 3))
```

A complete end-to-end example using the UCI Heart Disease dataset is provided at [`examples/uci_heart_demo.ipynb`](examples/uci_heart_demo.ipynb)

## Documentation

Additional documentation is available in the `docs/` directory:

- [`docs/tutorial.md`](docs/tutorial.md) – step-by-step workflow explanation
- [`docs/api_reference.md`](docs/api_reference.md) – documentation of each function
- [`docs/design_decisions.md`](docs/design_decisions.md) – rationale behind design choices


## Project context

This package was developed as part of the HPDM139 module (Health Data Science) at the University of Exeter.


## License

MIT License
