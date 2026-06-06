"""
Uplift Modeling — T-Learner with Qini validation
=================================================

A self-contained demo of uplift modeling on a synthetic RCT campaign dataset.
Distinguishes Persuadables (real ROI), Sure Things (would convert anyway),
Lost Causes, and Do-Not-Disturbs (negative uplift — campaign hurts them).

Run:
    pip install -r requirements.txt
    python uplift_modeling.py

Output:
    - Console summary: model performance, segment sizes, top-decile uplift
    - qini_curve.png: Qini curve (uplift model vs propensity baseline vs random)
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)


# ────────────────────────────────────────────────────────────────────
# 1. Synthetic RCT dataset
#
# 500K eligible users, 70/30 treated/control split.
# Each user has features that drive both baseline conversion AND
# treatment response. The four segments emerge naturally from how
# we construct the treatment effect.
# ────────────────────────────────────────────────────────────────────

def generate_rct(n_users: int = 500_000, treatment_share: float = 0.7) -> pd.DataFrame:
    df = pd.DataFrame({
        "tenure_days":     np.random.gamma(2.5, 200, n_users).clip(0, 2000),
        "product_count":   np.random.poisson(2.5, n_users).clip(0, 8),
        "txns_30d":        np.random.poisson(8, n_users),
        "avg_txn_value":   np.random.gamma(2, 500, n_users),
        "last_login_days": np.random.exponential(7, n_users).clip(0, 60),
        "tier":            np.random.choice([0, 1, 2], n_users, p=[0.65, 0.28, 0.07]),
        "cohort_age":      np.random.gamma(3, 90, n_users).clip(0, 1000),
        "feature_depth":   np.random.beta(2, 5, n_users),
    })

    # Baseline conversion propensity — high-tenure, high-engagement users more likely
    baseline = (
        0.06
        + 0.30 * df["feature_depth"]
        + 0.12 * (df["tier"] / 2)
        + 0.06 * (df["product_count"] / 8)
        - 0.10 * (df["last_login_days"] / 60)
    ).clip(0.01, 0.85)

    # Treatment effect — sharply heterogeneous so segments are realistic:
    #   - Mid-engagement, low-tenure users → positive uplift (Persuadables)
    #   - Very loyal/high-engagement users → negative uplift (Do-Not-Disturbs)
    #   - Inactive users → ~zero uplift (Lost Causes)
    treatment_effect = (
        -0.02
        + 0.18 * (1 - df["feature_depth"])              # less loyal = more persuadable
        + 0.10 * (1 - df["tier"] / 2)                   # lower tiers persuadable
        - 0.25 * (df["feature_depth"] > 0.55).astype(int)  # very loyal = DO NOT DISTURB
        - 0.15 * (df["last_login_days"] / 60)           # very inactive = zero
        + np.random.normal(0, 0.04, n_users)            # noise
    )
    treatment_effect = treatment_effect.clip(-0.15, 0.30)

    # Assign treatment randomly (the RCT)
    df["treated"] = (np.random.rand(n_users) < treatment_share).astype(int)

    # Realised conversion probability per user
    p_convert = baseline + df["treated"] * treatment_effect
    p_convert = p_convert.clip(0, 1)
    df["converted"] = (np.random.rand(n_users) < p_convert).astype(int)

    # Hide the ground-truth uplift from the model — only used later for evaluation
    df["_true_uplift"] = treatment_effect
    return df


# ────────────────────────────────────────────────────────────────────
# 2. T-Learner — two XGBoost models, uplift = treated − control prediction
# ────────────────────────────────────────────────────────────────────

FEATURES = [
    "tenure_days", "product_count", "txns_30d", "avg_txn_value",
    "last_login_days", "tier", "cohort_age", "feature_depth",
]


def fit_t_learner(df: pd.DataFrame):
    df_t = df[df["treated"] == 1]
    df_c = df[df["treated"] == 0]

    common = dict(
        max_depth=5, n_estimators=300, learning_rate=0.05,
        eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
    )
    model_t = XGBClassifier(**common).fit(df_t[FEATURES], df_t["converted"])
    model_c = XGBClassifier(**common).fit(df_c[FEATURES], df_c["converted"])
    return model_t, model_c


def uplift_predict(model_t, model_c, X: pd.DataFrame) -> np.ndarray:
    p_t = model_t.predict_proba(X)[:, 1]
    p_c = model_c.predict_proba(X)[:, 1]
    return p_t - p_c


# ────────────────────────────────────────────────────────────────────
# 3. Qini curve — the right way to validate uplift models
#
# AUC doesn't apply here. We're predicting a counterfactual, not a label.
# Qini sorts users by predicted uplift, then plots cumulative incremental
# conversions vs share of users targeted. Random = diagonal. Better model
# = curve rising above diagonal.
# ────────────────────────────────────────────────────────────────────

def qini_curve(df: pd.DataFrame, score_col: str):
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    d["cum_treated"]      = (d["treated"]).cumsum()
    d["cum_control"]      = (1 - d["treated"]).cumsum()
    d["cum_convert_t"]    = (d["converted"] * d["treated"]).cumsum()
    d["cum_convert_c"]    = (d["converted"] * (1 - d["treated"])).cumsum()
    # Adjusted incremental conversions (Radcliffe's Qini)
    n_t = d["cum_treated"].replace(0, np.nan)
    n_c = d["cum_control"].replace(0, np.nan)
    d["qini"] = (d["cum_convert_t"] - d["cum_convert_c"] * (n_t / n_c)).fillna(0)
    return d["qini"].values


def qini_auc(qini: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting diagonal,
    normalised to the population size."""
    n = len(qini)
    x = np.linspace(0, 1, n)
    diag = np.linspace(0, qini[-1], n)
    # np.trapezoid in NumPy >= 2.0, falls back to np.trapz for older versions
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_above_diag = trapezoid(qini - diag, x)
    return float(area_above_diag / n)   # normalised per-user lift


# ────────────────────────────────────────────────────────────────────
# 4. Segment the population — Persuadables, Sure Things, Do-Not-Disturbs
# ────────────────────────────────────────────────────────────────────

def segment_users(df: pd.DataFrame, uplift: np.ndarray, p_control: np.ndarray):
    """
    Percentile-based segmentation — produces realistic proportions across
    any uplift distribution. A real targeting workflow would tune these
    thresholds against an operational budget constraint.

    Persuadable      = top 20% by predicted uplift
    Do-Not-Disturb   = bottom 5% (negative uplift)
    Sure Thing       = middle, high control conversion (would convert anyway)
    Lost Cause       = middle, low control conversion (won't convert either way)
    """
    df = df.copy()
    df["uplift"]    = uplift
    df["p_control"] = p_control

    persuadable_cut = np.quantile(uplift, 0.80)
    dnd_cut         = np.quantile(uplift, 0.05)
    p_control_med   = np.median(p_control)

    df["segment"] = "Lost Cause"
    df.loc[df["uplift"] >= persuadable_cut, "segment"] = "Persuadable"
    df.loc[df["uplift"] <= dnd_cut,         "segment"] = "Do-Not-Disturb"
    middle = (df["uplift"] > dnd_cut) & (df["uplift"] < persuadable_cut)
    df.loc[middle & (df["p_control"] > p_control_med), "segment"] = "Sure Thing"
    return df


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic RCT dataset…")
    df = generate_rct(n_users=200_000)
    print(f"  n = {len(df):,}  |  treated = {df['treated'].mean():.0%}  |  overall convert = {df['converted'].mean():.2%}")

    # Split into train / holdout
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    train = df.iloc[idx[:160_000]].copy()
    test  = df.iloc[idx[160_000:]].copy()

    print("\nFitting T-Learner (XGBoost × 2)…")
    model_t, model_c = fit_t_learner(train)

    print("Scoring holdout…")
    test["uplift"]    = uplift_predict(model_t, model_c, test[FEATURES])
    test["p_control"] = model_c.predict_proba(test[FEATURES])[:, 1]

    # Baseline: standard propensity model — use the control model alone as
    # a propensity score. This is what a "target high P(convert)" workflow
    # would look like.
    test["propensity"] = test["p_control"]

    print("\nComputing Qini curves…")
    # Random baseline: assign each user a random score from the same distribution
    test["random_score"] = np.random.permutation(test["uplift"].values)

    q_uplift = qini_curve(test, "uplift")
    q_propen = qini_curve(test, "propensity")
    q_random = qini_curve(test, "random_score")

    auc_uplift = qini_auc(q_uplift)
    auc_propen = qini_auc(q_propen)
    auc_random = qini_auc(q_random)

    print(f"  Qini-AUUC  uplift  model : {auc_uplift:+.4f}")
    print(f"  Qini-AUUC  propensity    : {auc_propen:+.4f}")
    print(f"  Qini-AUUC  random        : {auc_random:+.4f}")

    print("\nSegmenting users by predicted uplift…")
    seg = segment_users(test, test["uplift"].values, test["p_control"].values)
    print(seg["segment"].value_counts(normalize=True).map("{:.1%}".format))

    print("\nTop decile uplift summary:")
    top10 = seg.nlargest(int(len(seg) * 0.10), "uplift")
    print(f"  Mean predicted uplift in top 10% : {top10['uplift'].mean():+.3f}")
    print(f"  Actual conversion (treated arm) : {top10[top10.treated==1]['converted'].mean():.2%}")
    print(f"  Actual conversion (control arm) : {top10[top10.treated==0]['converted'].mean():.2%}")

    # Save Qini curve plot
    print("\nSaving qini_curve.png …")
    plt.figure(figsize=(8, 5.5))
    plt.plot(np.linspace(0, 1, len(q_uplift)), q_uplift, label="Uplift model (T-Learner)", lw=2)
    plt.plot(np.linspace(0, 1, len(q_propen)), q_propen, label="Propensity model", lw=2, ls="--")
    plt.plot([0, 1], [0, q_uplift[-1]], color="gray", lw=1, ls=":", label="Random targeting")
    plt.xlabel("Share of users targeted")
    plt.ylabel("Cumulative incremental conversions")
    plt.title("Qini curve — uplift model vs propensity vs random")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("qini_curve.png", dpi=120)
    print("  Done.")


if __name__ == "__main__":
    main()
