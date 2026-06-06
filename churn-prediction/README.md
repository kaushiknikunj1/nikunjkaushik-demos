# Churn Prediction — XGBoost + SHAP

Self-contained, runnable demo of a production-style churn prediction pipeline. Generates a synthetic fintech user dataset, trains an XGBoost classifier, and uses SHAP to convert the risk score into actionable, per-user drivers — what the CRM team actually needs.

## What it does

1. Generates 100K synthetic users with realistic fintech features (transaction patterns, feature depth, engagement signals).
2. Trains XGBoost with `scale_pos_weight` for class imbalance.
3. Validates with AUC + top-decile precision + top-two-decile recall (the metrics that actually map to a CRM workflow).
4. Computes SHAP values — global feature importance + per-user top drivers.
5. Prints sample personalised drivers for the top 5 at-risk users.

## Run it

```bash
pip install -r requirements.txt
python churn_prediction.py
```

Output: console summary + `shap_importance.png`.

## Why SHAP matters here

A churn risk score on its own is useless to a CRM team. "User 12345 is 87% likely to churn" doesn't tell anyone what to *do*. SHAP turns the score into "User 12345 is at risk because of: failed transaction ratio, low feature depth, days since last transaction" — which maps directly to a campaign trigger.

Interpretability isn't a nice-to-have. It's a business requirement.

## Related

- [Full case study](https://nikunjkaushik.com/projects/churn-prediction.html)
- [Blog post](https://nikunjkaushik.com/blog/churn-prediction-ml.html)
