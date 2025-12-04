## HPDM139 First Group Project Meeting Notes 

13th November 2025 
Present: Raiet Bekirov, Nick Berry, Becky Griffiths, Kayla Yasmine.

## Project Ideas 

### A Package for Dealing with Missing Data 

Could include functions that take a csv/pandas df with missing values (NAs) and return csvs with no missing values; the NA entries can be dealt with using a variety of methods – e.g. complete case analysis (deleting all rows with any NAs), mean/median/mode imputation (replacing any NAs with the mean/median/mode value of the column), and multiple imputation (a more complicated approach that replaces NAs with values predicted from other columns) 

Could also include visualisation functions to see how ‘missingness’ of a chosen variables varies according to another (categorical) variable – e.g. allowing user to inspect whether the proportion of patients with missing blood pressure readings varies by gender, IMD quintile etc. 

### A Package for Evaluating Algorithmic Fairness in Clinical Prediction Modules 

Could produce metrics assessing how well clinical prediction models perform in different groups and detecting whether the model exhibits e.g. racial or gender bias. 

Could look at the impact of class imbalance. 

Could use the performance of different prediction models on the UCI heart disease dataset as an example. 

### A Package for Performing Meta-Analysis in Python 

Could make a user-friendly package for performing the kind of meta-analyses used in Cochrane systematic reviews from input study results. There are good packages for this in R like meta and metafor but not a. Python option with as many features that’s as easy to use. The package could produce visualisations like funnel plots to assess for publication bias and summary forest plots. 

### A Package for Statistical Process Control 

Statistical process control charts are widely used within NHS hospitals to visualise variation in quantities like ED admissions over time. The NHS has a preferred ‘plot the dots’ style for these charts but there is no Python tool that allows users to easily create charts in this style (source: https://nhsrplotthedots.nhsrcommunity.com/articles/other-spc-tools.html). 

 