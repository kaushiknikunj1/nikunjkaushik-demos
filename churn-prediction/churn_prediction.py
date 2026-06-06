"""
Churn Prediction — XGBoost + SHAP for actionable risk scores
=============================================================

Self-contained demo. Generates a synthetic fintech user dataset, trains an
XGBoost churn classifier, then uses SHAP to surface the top 3 drivers of
each user's risk score — turning a black-box score into something the CRM
team can act on.

Run:
    pip install -r requirements.txt
    python churn_prediction.py

Output:
    - Console: model performance (AUC, top-decile precision/recall),
      SHAP feature importance, sample personalised drivers
    - shap_importance.png: global SHAP feature importance
"""

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)


# ────────────────────────────────────────────────────────────────────
# 1. Synthetic fintech user data
#
# Each row is a user at a scoring date. Churn = no transaction in 60 days
# for users who had at least one transaction in their first 30 days post-
# activation.
# ────────────────────────────────────────────────────────────────────

def generate_users(n: int = 100_000) -> pd.DataFrame:
    df = pd.DataFrame({
        "tenure_days":          np.random.gamma(2.5, 200, n).clip(0, 1500),
        "txn_count_30d":        np.random.poisson(7, n),
        "active_days_30d":      np.random.poisson(8, n).clip(0, 30),
        "avg_txn_value":        np.random.gamma(2, 800, n),
        "txn_success_rate":     np.random.beta(20, 2, n),
        "products_used":        np.random.poisson(2.2, n).clip(0, 6),
        "feature_depth_d14":    np.random.beta(2, 4, n),
        "push_open_rate_30d":   np.random.beta(3, 8, n),
        "support_contacts_30d": np.random.poisson(0.3, n),
        "days_since_last_txn":  np.random.exponential(10, n).clip(0, 90),
        "failed_txn_ratio":     np.random.beta(1, 30, n),
        "payment_methods":      np.random.choice([1, 2, 3, 4], n, p=[0.40, 0.35, 0.20, 0.05]),
    })

    # Latent churn propensity (used to generate the label, then dropped).
    # Strong, deterministic signal with modest noise so a tree model can
    # recover the structure cleanly — AUC lands around 0.86–0.88 in practice.
    z = (
        -2.0
        + 0.10  * df["days_since_last_txn"]
        - 0.06  * df["txn_count_30d"].clip(0, 20)
        - 2.50  * df["feature_depth_d14"]
        - 1.20  * df["push_open_rate_30d"]
        + 0.60  * df["support_contacts_30d"]
        + 4.00  * df["failed_txn_ratio"]
        - 0.30  * df["products_used"]
        - 0.002 * df["tenure_days"]
        + np.random.normal(0, 0.5, n)
    )
    p_churn = 1 / (1 + np.exp(-z))
    df["churned"] = (np.random.rand(n) < p_churn).astype(int)
    return df


FEATURES = [
    "tenure_days", "txn_count_30d", "active_days_30d", "avg_txn_value",
    "txn_success_rate", "products_used", "feature_depth_d14",
    "push_open_rate_30d", "support_contacts_30d", "days_since_last_txn",
    "failed_txn_ratio", "payment_methods",
]


# ────────────────────────────────────────────────────────────────────
# 2. Model
# ────────────────────────────────────────────────────────────────────

def fit_model(X_train, y_train):
    # scale_pos_weight handles imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)
    model = XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.05,
        eval_metric="logloss", scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ────────────────────────────────────────────────────────────────────
# 3. SHAP — global importance + per-user top drivers
# ────────────────────────────────────────────────────────────────────

def top_drivers(shap_row: np.ndarray, feature_names: list[str], n: int = 3):
    """Return [(feature, signed shap value), ...] for the top n drivers."""
    order = np.argsort(np.abs(shap_row))[::-1][:n]
    return [(feature_names[i], round(float(shap_row[i]), 3)) for i in order]


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic user dataset…")
    df = generate_users(n=100_000)
    print(f"  n users = {len(df):,}  |  churn rate = {df['churned'].mean():.2%}")

    X = df[FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    print(f"  train = {len(X_train):,}  |  test = {len(X_test):,}")

    print("\nFitting XGBoost churn classifier…")
    model = fit_model(X_train, y_train)

    print("\nEvaluating on holdout…")
    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    # Top-decile metrics
    threshold = np.quantile(p, 0.90)
    preds_top10 = (p >= threshold).astype(int)
    prec10 = precision_score(y_test, preds_top10)
    rec_top20 = recall_score(y_test, (p >= np.quantile(p, 0.80)).astype(int))
    print(f"  AUC-ROC                         : {auc:.3f}")
    print(f"  Precision @ top decile          : {prec10:.2%}")
    print(f"  Recall    @ top two deciles     : {rec_top20:.2%}")

    print("\nComputing SHAP values on holdout sample…")
    sample = X_test.sample(n=2_000, random_state=SEED)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)

    # Global importance — mean |shap|
    global_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("\nTop 6 churn drivers (mean |SHAP|):")
    for feat, imp in global_imp.head(6).items():
        print(f"  {feat:24s}  {imp:.3f}")

    print("\nSaving shap_importance.png …")
    plt.figure(figsize=(8.5, 5))
    global_imp.head(10).iloc[::-1].plot.barh(color="#4A6FA5")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global feature importance (SHAP) — churn classifier")
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=120)

    # Personalised drivers for top 5 at-risk users — this is what the CRM team uses
    print("\nTop 5 at-risk users — personalised drivers:")
    sample_with_pred = sample.copy()
    sample_with_pred["risk"] = model.predict_proba(sample)[:, 1]
    sample_with_pred = sample_with_pred.sort_values("risk", ascending=False)
    sample_idx = sample_with_pred.head(5).index
    for rank, uid in enumerate(sample_idx, 1):
        i = list(sample.index).index(uid)
        drivers = top_drivers(shap_vals[i], FEATURES, n=3)
        risk = sample_with_pred.loc[uid, "risk"]
        print(f"  User {uid}  risk={risk:.2%}  →  drivers: {drivers}")

    print("\nDone.")


if __name__ == "__main__":
    main()
