"""
Example Fairness Evaluation Script
====================================================

This script demonstrates a ML workflow using the
Heart Failure Prediction dataset.

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

 It shows:

1. Loading labelled clinical data (X = features, y = target)
2. Splitting into training & test sets
3. Preprocessing by scaling numeric features + encoding categorical features
4. Training a logistic regression classifier
5. Evaluating predictive performance (accuracy, confusion matrix, sensitivity)
6. Assessing fairness using Demographic Parity Difference (DPD) across sex

This script is purposefully simple and intended as a possible starting point for
our HPDM139 group fairness-toolkit project.

There'd be work to do to go from here to a generic tookit, with
generic loaders, preprocessors, models, and fairness metrics.

Hope this gives a feeling for the overall workflow

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split







# ==========================================================
# 1. Load dataset
# ==========================================================
"""
X : pandas DataFrame
    Contains the input clinical features such as age, sex,
    chest pain type, resting blood pressure, cholesterol etc.

y : pandas Series
    Binary label indicating presence (1) or absence (0) of heart disease.
"""

df = pd.read_csv("../data/heart.csv")

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nFirst 10 values of y:")
print(y.head(10))


# ==========================================================
# 2. Train/test split
# ==========================================================
"""
Splits the dataset so that:

- 80% is used for training
- 20% is reserved for testing (unseen evaluation)
- 'stratify=y' keeps class balance the same across both sets.
This ensures fair and reproducible evaluation.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)


# ==========================================================
# 3. Identify categorical vs numeric features
# ==========================================================
"""
Most ML models cannot directly handle categorical strings such as:
    - "Male"
    - "ATA"
    - "ST"
So here they are lsited for one-hot encoding later.

Numeric features will be scaled (standardised).
"""

categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)


# ==========================================================
# 4. Build preprocessing pipeline
# ==========================================================
"""
ColumnTransformer allows applying:
    - StandardScaler to numeric columns
    - OneHotEncoder to categorical columns

This creates a clean, reproducible preprocessing step that is applied
consistently during training and testing.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)


# ==========================================================
# 5. Build full model pipeline
# ==========================================================
"""
Pipeline ensures preprocessing happens automatically before training
and before generating predictions.

Here let's use Logistic Regression as a baseline classifier.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================================
# 6. Train (fit) the model
# ==========================================================
"""
Runs the preprocessing + logistic regression training.
"""
model.fit(X_train, y_train)


# ==========================================================
# 7. Evaluate predictive performance
# ==========================================================
"""
Computes:
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

These metrics reflect predictive performance, not fairness.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))


# ==========================================================
# 8. Fairness Evaluation: Demographic Parity Difference
# ==========================================================
"""
Demographic Parity (DP):
-----------------------
Measures whether the model predicts positive outcomes at different
rates for different groups — in this example, men vs women.

DP Difference = | P(Ŷ=1 | Sex=M) – P(Ŷ=1 | Sex=F) |

Interpretation:
- 0.0       → perfectly fair
- < 0.05    → very low disparity
- 0.05-0.10 → mild disparity
- 0.10-0.20 → moderate disparity
- > 0.20    → concerning fairness disparity 

A large difference means the model predicts heart disease more often
for one sex than the other, *regardless of true disease status*.
"""

# Extract the protected attribute (sex)
sex = np.array(X_test["Sex"])
y_pred = np.array(y_pred)

men_mask = (sex == "M")
women_mask = (sex == "F")

positive_rate_men = y_pred[men_mask].mean()
positive_rate_women = y_pred[women_mask].mean()

print("Positive prediction rate (men):", positive_rate_men)
print("Positive prediction rate (women):", positive_rate_women)

dp_diff = abs(positive_rate_men - positive_rate_women)
print("Demographic Parity Difference:", dp_diff)