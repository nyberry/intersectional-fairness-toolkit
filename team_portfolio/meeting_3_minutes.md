
# **Minutes of Meeting – Differential Fairness Toolkit Project**

**Date:** Friday 28th November
**Time:** 4:00 PM
**Location:** Online
**Present:** All four group members (Nick, Kayla, Becky, Raiet)

---

## **1. Purpose of the Meeting**

To discuss early ideas and research for implementing a differential fairness toolkit for the HPDM139 group project.

---

## **2. Discussion Summary**

### **2.1 Differential Fairness – Approach**

* The group explored how differential fairness can be implemented in a simple, reusable Python package suitable for health data science workflows.
* Key considerations included:

  * How to design functions that compute differential fairness metrics.
  * How to apply fairness evaluation to typical health datasets (e.g., structured tabular data).
  * Making the package easy for other students/health data scientists to import and use.

### **2.2 Review of Intersectional Fairness Literature**

* **Kayla shared a key research paper** on intersectional fairness, explaining the concept of protecting multiple marginalised groups defined by combinations of attributes (e.g., age × sex × ethnicity).
* The group agreed that intersectional approaches should be included in the toolkit:

  * Handling intersectional subgroups efficiently
  * Avoiding small-group instability
  * Considering privacy-preserving or smoothed estimators

### **2.3 Exploration of Reference GitHub Repo**

* The group examined an existing GitHub repository implementing differential fairness.
* Noted as a useful reference for:

  * Code structure
  * Metric definitions
  * Potential API patterns
* Agreed not to copy directly, but to take inspiration for organising our own package.

---

## **3. Actions Agreed**

* **All members** to read and (try!) to understand the shared intersectional fairness paper in detail.
* **Each member** to come back next week with:

  * Ideas on how we might structure our package (modules, functions, workflow).
  * Thoughts on which metrics and visualisations to prioritise.
  * A *firm plan for how we will divide work* and deliver the package by the **9th January** project deadline.

---

## **4. Date of Next Meeting**

4th December 2025, 10.30am, online.
