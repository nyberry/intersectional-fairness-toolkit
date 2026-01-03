# Intersectional Fairness Toolkit

This documentation describes a Python toolkit for evaluating intersectional fairness
in machine learning models applied to tabular health datasets.

The toolkit supports construction of intersectional protected groups (e.g. sex × age group), alignment of model predictions with protected attributes, and computation of fairness metrics.

It is designed to be dataset-agnostic and model-agnostic, and can be integrated
into a range of health data science workflows.

---

## What is intersectional fairness?

Fairness issues may not appear when analysing protected attributes in isolation (e.g. sex or age). Disparities often emerge at the intersections of attributes (([Foulds, 2019](https://arxiv.org/abs/1807.08362))).

For example, a model may perform similarly for: men vs women, or for: younger vs older patients;  while still performing substantially worse for: older women vs younger men.

This toolkit enables systematic evaluation of such intersectional effects.

---

## Typical workflow

A standard workflow using this package is:

1. Load a tabular health dataset
2. Apply fairness-oriented preprocessing (e.g. age binning)
3. Prepare model-ready features
4. Train any classification model
5. Construct an evaluation DataFrame aligned with predictions
6. Compute fairness metrics across protected groups
7. Visualise the outputs

---

## Getting started

If you are new to the toolkit, start here:

- **[Tutorial](tutorial.md)**  
  A step-by-step walkthrough using the UCI Heart Disease dataset.

- **[API reference](api_reference.md)**  
  Detailed documentation for all modules and functions  

- **[Design decisions](design_decisions.md)**  
  Rationale behind design and methodological choices  

---

## Project context

This toolkit was developed as part of the **HPDM139 Health Data Science** module
at the University of Exeter.

It is intended for educational and research use in fairness-aware machine learning.
