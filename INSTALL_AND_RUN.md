# Installation and Running Instructions

This document explains how to install and run the *Intersectional Fairness Toolkit* from the a ZIP file.

## List of contents of the ZIP

The ZIP file contains the following directories and files:

- `pyproject.toml`  
  Package configuration file defining the project metadata and build system.

- `environment.yml`  
  Conda environment specification allowing the software environment to be fully recreated.

- `INSTALL_AND_RUN.md`  
  Step-by-step instructions to install, test, and run the package.

- `README.md`  
  Overview of the project, its purpose, and guidance on using the toolkit.

- `LICENSE`  
  Open-source license for the project.

- `src/`  
  Source code for the Python package:
  - `fairness/` – core package modules for data handling, group construction, fairness metrics, and visualisation.

- `tests/`  
  Automated unit tests for core functionality and fairness metric calculations.

- `docs/`  
  Documentation source files used to generate the project documentation site, including:
  - tutorial and user guide
  - API reference
  - design decisions
  - team portfolio and project documentation

- `examples/`  
  Jupyter notebooks demonstrating how to use the toolkit in practice, including:
  - loading clinical tabular data
  - computing fairness metrics
  - visualising results

- `data/`  
  Example dataset and accompanying documentation used in demonstrations.

- `mkdocs.yml`  
  Configuration file for building the project documentation using MkDocs.

- `binder/`  
  Binder configuration files to allow the notebooks to be run in an online Binder environment.


---

## 1. Create the Python environment

This project uses **conda** for dependency management.

From the root of the submission directory, run:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate fairness
```

## 2. Install the package

Install the package in editable mode:

```bash
pip install -e .
```

You should now be able to import the package in Python:

```bash
python -c "import fairness; print('fairness imported successfully')"
```

## 3. Run the tests

To verify that the package is working correctly, run:

```bash
pytest
```

All tests should pass without errors.

## 4. Run example notebooks

Example Jupyter notebooks demonstrating the toolkit are provided in the `examples/` directory.

Start Jupyter:

```bash
jupyter lab
```

Then open one of the following notebooks:
- `examples/uci_heart_demo.ipynb`
- `examples/metrics_demo.ipynb`
- `examples/single_metrics_demo.ipynb`
- `examples/visualisation_demo.ipynb`

These notebooks demonstrate:
- loading and preprocessing data
- constructing intersectional groups
- computing fairness metrics
- visualising and interpreting results.


## 5. View the documentation

All documentation fot this package is available at [https://raiet-bekirov.github.io/HPDM139_assignment/](https://raiet-bekirov.github.io/HPDM139_assignment/)

Documentation source files are located in the docs/ directory. To build and view the documentation locally:

```bash
mkdocs build
mkdocs serve
```

Then open the displayed local URL in a web browser

