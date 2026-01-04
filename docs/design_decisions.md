## Design Decisions

This document outlines the key design decisions made in the development of the Intersectional Fairness Toolkit.

### 1. Purpose and Scope

The purpose of this project is to provide a lightweight, transparent, and extensible Python toolkit for evaluating intersectional fairness in machine-learning models applied to tabular health datasets.

The toolkit is intended to:

- Support health data scientists and researchers
- Demonstrate best practice software engineering for an MSc-level project


### 2. Intended Use and Audience

The main intended users are:

- Health data science students
- Researchers evaluating fairness in clinical prediction models
- Practitioners exploring bias in tabular health data

The toolkit assumes:

- A binary classification setting, which is common in healthcare
- Predictions (y_pred) are generated externally by any model

Users may want to use their own metrics, models, or datasets. As a result, the toolkit is designed to be model-agnostic and dataset-agnostic.

### 3. Separation of Concerns

A core design principle was separation of responsibilities across modules.

Fairness analysis typically involves:
- Data loading and preprocessing
- Model training (external)
- Fairness evaluation
- Visualisation

To keep the system flexible and maintainable:

- Model training is not part of the core fairness API
- Data preparation is kept distinct from metric computation
- Metrics operate on simple, explicit inputs

This makes it easier to swap models, reuse metrics, and test components independently.

### 4. Package Structure

The project is implemented as a Python package using a src/ layout:

src/
  fairness/
    data.py
    preprocess.py
    groups.py
    adapters.py
    metrics/
    plots/
    demo/

Key design choices:

- using src/ Prevents accidental imports from the repository root.

- Each module has a single responsibility (e.g. groups.py only handles group construction).

- Functions accept and return plain Python objects (DataFrame, lists, dicts)

This structure supports reusability, testing and long-term extensibility

### 5. Data Flow Into Fairness Metrics

A key design decision was to standardise how data is passed into fairness metrics.

Rather than passing many loosely related arrays, the toolkit constructs a single evaluation DataFrame `eval_df` with aligned columns:

- subject_label, the intersectional protected group
- y_pred, the model prediction
- y_true, the true outcome

Each row corresponds to one individual in the test set. This maintains alignment between predictions and protected attributes.

Adapters are provided to unpack this DataFrame into formats required by different metric implementations.

### 6. Intersectional Group Construction

Intersectional groups are defined by combining multiple protected attributes (e.g. Sex & age_group).

Design decisions include:

- Human-readable group labels (e.g. Sex=0|age_group=older)
- Explicit handling of missing values
- Row-by-row alignment with predictions

The toolkit avoids hard-coding protected attributes, allowing users to define the intersections they wish to examine.

### 7. Choice of Fairness Metrics

The toolkit supports group-based and intersectional fairness metrics, including:

- Accuracy differences and ratios
- False negative rate (FNR)
- False positive rate (FPR)
- Omission and discovery rates
- Maximum intersectional disparities
- Differential-style worst-case ratios

These metrics are of clinical importance, and allow examination of worst-case harms, as wekl as averages.

Metrics are implemented as functions, making them easy to test and reuse.

### 8. Visualisation Strategy

Visualisation code is separated into its own module (plots/).

Design goals:

- Accept outputs from metric functions
- Support quick identification of high-risk groups

### 9. Documentation Strategy

Tools

- MkDocs with Material theme
- MkDocStrings for API documentation from docstrings
- Hosted via GitHub Pages

Structure

- index.md, an overview
- tutorial.md, an end-to-end example
- api_reference.md,  generated from code
- design_decisions.md, this document
- team_portfolio/, evidence of collaboration


### 10. Collaboration and Extensibility

The toolkit was designed to support parallel development. Metrics, visualisations, and pipelines can evolve independently.

New datasets and metrics can be added without refactoring core code.


### 11. Packaging and Distribution

The project is configured using pyproject.toml and is installable via:

pip install -e .

The group agreed to:

- Publish initially to TestPyPI
- Progress to PyPI once stable
