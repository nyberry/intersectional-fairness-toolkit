# HPDM139_assignment

### What our group must deliver:

A zip file that contains our python package and some documents such as a README, installation guide and a user tutorial.

### When our group must deliver it:

9th Jan 2026

### What is our python package?

A small toolbox that we built, along with simple instructions on how to use the toos.

Our toolbox (package) will help a health scientist check whether a machine learning model treats different groups of patients fairly, especiallywhen groups overlap (eg. older women).

### What might our python package contain?

```bash
fairness_toolkit/
    fairness_toolkit/
        __init__.py
        groups.py                 # creates intersectional groups
        differential_fairness.py  # calculates DF
        metrics.py                # simple fairness metrics
        plots.py                  # fairness visualisations
        data/
            heart.csv             # demo dataset
    README.md                     # indluding installation instructions
    tutorial.ipynb                # including a worked example?
    environment.yml
    tests/
        test_groups.py
        test_df.py
    team_portolio/
        meeting_1_agenda.md
        meeting_1_minutes.md
        meeting_2... etc
```

The python package is a folder that contains our code (fairness_toolkit). As a minimum it might contain:

1. a simple function called groups.py that sorts the patients into their overlapping groups. It takes the dataset, takes the protected characteristics such as sex and age group, and outputs intersectional groups like "sex=F|age=old". 

2. a function called differential_fairness.py which calculates the DF measure, usinhg the formula from Kayla's paper. This checks how differently a model treats each group.

The 'model' is just a column of predicted labels (0 = no heart disease, 1 = heart disease). We do not need to implement or train a machine learning model within this package. We simply accept the predictions the user provides.

(HOWEVER... to demonstrate the package we might use a simple ML model like logistic regression from scikit-learn)

3. a function called plots.py, draws a picture so people can see who's treated badly.

The repo should also contain:

- README.md, the front end of the project. Explains the problem we are solving. describes the package and what each file does. Installation instructions (how to install the package and how to install the deoendencies). 

- Tutorial/ user guide tutorial.ipnyb. A jupyter notebook showig how to load the dataset, creat intersectional groups, run differential fairness, plot results.

- Dependency managemeny (environmeny.yml)

- Testing











## Structure
- fairness_toolkit/: package code
- team_portfolio/: meeting minutes etc
- tests/: automated tests
- tutorial.ipynb: demo / tutorial notebook
- READNE.md
- environment.yml

## Development
Branch strategy
- main: stable, deliverable code only
- dev: active development
- feature branches: small chunks of work

## Suggested workflow:\Workflow
1. Create branch from dev
2. Make changes
3. Push branch
4. Open a Pull Request into dev
5. Two group members review
6. Merge into dev
7. When a milestone is reached, merge dev into main
