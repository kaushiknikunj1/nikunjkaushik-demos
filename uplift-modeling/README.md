# Uplift Modeling — T-Learner Demo

A self-contained, runnable demonstration of uplift modeling on a synthetic RCT dataset. Identifies the four behavioural segments — Persuadables, Sure Things, Lost Causes, and Do-Not-Disturbs — that a standard propensity model collapses into one.

## What it does

1. Generates a 200K-user synthetic randomised-control-trial dataset with a heterogeneous treatment effect (some users react positively, some negatively, some not at all).
2. Fits a **T-Learner** (two XGBoost classifiers — one for treated, one for control).
3. Validates with the **Qini curve** (the right tool for uplift — AUC doesn't apply).
4. Benchmarks against a standard propensity model and random targeting.
5. Segments users into Persuadables / Sure Things / Lost Causes / Do-Not-Disturbs.

## Run it

```bash
pip install -r requirements.txt
python uplift_modeling.py
```

Output: console summary + `qini_curve.png`.

## Why uplift, not propensity

A propensity model tells you who's *likely to convert*. An uplift model tells you whose *conversion probability goes up because of the campaign*. Most teams build the first and discover — months later, after a holdout — that they've been buying conversions they already owned.

The shift in question changes everything: the model, the validation, the targeting logic, and the budget conversation.

## Related

- [Full case study with results](https://nikunjkaushik.com/projects/uplift-modeling.html)
- [Blog post](https://nikunjkaushik.com/blog/uplift-modeling.html)
