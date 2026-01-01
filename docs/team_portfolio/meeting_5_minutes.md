## Minutes – Differential Fairness Toolkit Project

Meeting #5

Date: 27th December 2025
Time: 10:00 AM
Location: Online

## Attendees
- Nick
- Becky
- Kayla
- Raiet

### 1. Welcome and Purpose of Meeting

The purpose of the meeting was to:

Review and compare each member’s prototype pipeline

Agree on a shared design for the fairness toolkit

Decide which components should be merged into the common development branch

### 2. Review of Individual Pipelines

Each group member presented a minimal working pipeline covering:

Dataset loading

Preprocessing steps

Fairness metric computation

Model training and evaluation

Clarity and usability of outputs

### Discussion summary:

All pipelines successfully demonstrated.

Common patterns were identified across pipelines, particularly in data loading, preprocessing, and evaluation alignment.

Some duplication of logic was noted, especially around group handling and evaluation data structures.

Agreement that a standardised pipeline structure would improve clarity and reuse.

### Agreement on Core Toolkit Design

Following discussion, the group agreed on the following design principles:

A clean, modular package structure separating data loading, preprocessing, grouping, metrics, and visualisation.

A minimal, well-documented v0.1 feature set focused on correctness and clarity rather than breadth.

Inclusion of example notebooks and documentation as components of the toolkit.

###  Decisions on What to Merge into Main

The group agreed that:

Only reviewed, working components would be merged into the shared main branch.

Focus would be on core functionality rather than optimisation.

### Next Steps and Work Allocation

The following responsibilities were agreed:

- Becky & Kayla: Continue development of fairness metric functions, including group-level and intersectional metrics.

- Nick. Develop project documentation. Finalise and maintain the overall package structure. Ensure examples and tutorials clearly demonstrate intended usage.

- Raiet. Develop visualisation components to support interpretation of fairness metrics.

- All: use docstrings to explain the scope of each function

Additionally:

- The group agreed to publish the package initially to TestPyPI to validate installation and imports.

- Pending successful testing, the package will later be published to PyPI.


### Date of Next Meeting

Final meeting scheduled for:

- Saturday 3rd January 2026
- 10:00 AM (Online)