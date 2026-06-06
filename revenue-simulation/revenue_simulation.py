"""
Revenue Simulation — Monte Carlo forecasting under uncertainty
===============================================================

Self-contained demo. Models 36-month revenue as a function of monthly user
acquisition, churn, and ARPU — each treated as a random variable with a
realistic distribution. Runs 10,000 simulations to produce a confidence band
around the forecast, then quantifies the sensitivity to each lever so the
business knows where to spend strategy attention.

Run:
    pip install -r requirements.txt
    python revenue_simulation.py

Output:
    - Console: percentile bands at month 12, 24, 36; sensitivity ranking
    - revenue_fan.png: fan chart of P10 / P50 / P90 revenue paths
    - lever_sensitivity.png: tornado chart — which lever moves the P&L most
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)


# ────────────────────────────────────────────────────────────────────
# 1. Model assumptions
#
# Each lever has a base value and a distribution shape. The point isn't to
# be precise — it's to capture that the world is uncertain and that the
# planning P&L should reflect that.
# ────────────────────────────────────────────────────────────────────

MONTHS               = 36
N_SIMS               = 10_000
STARTING_USERS       = 50_000

# Monthly new user acquisition — slowly growing with noise
ACQ_BASE             = 8_000
ACQ_GROWTH_PER_MONTH = 0.015     # 1.5% MoM
ACQ_NOISE_SD         = 0.18      # ±18% lognormal noise

# Monthly churn rate — log-normal around 4%
CHURN_BASE           = 0.040
CHURN_NOISE_SD       = 0.20      # ±20% lognormal

# ARPU (avg revenue per user per month) — Gamma-shaped
ARPU_BASE            = 220.0     # rupees
ARPU_NOISE_SD        = 0.15


# ────────────────────────────────────────────────────────────────────
# 2. Simulation engine
# ────────────────────────────────────────────────────────────────────

def simulate(n_sims: int = N_SIMS,
             months: int = MONTHS,
             acq_base: float = ACQ_BASE,
             acq_growth: float = ACQ_GROWTH_PER_MONTH,
             churn_base: float = CHURN_BASE,
             arpu_base: float = ARPU_BASE,
             starting_users: int = STARTING_USERS) -> np.ndarray:
    """
    Returns a (n_sims, months) matrix of monthly revenue.
    """
    revenue = np.zeros((n_sims, months))
    users   = np.full(n_sims, starting_users, dtype=float)

    for m in range(months):
        # Sample monthly noise (lognormal so values are positive)
        acq_mult   = rng.lognormal(mean=-(ACQ_NOISE_SD ** 2) / 2,   sigma=ACQ_NOISE_SD,   size=n_sims)
        churn_mult = rng.lognormal(mean=-(CHURN_NOISE_SD ** 2) / 2, sigma=CHURN_NOISE_SD, size=n_sims)
        arpu_mult  = rng.lognormal(mean=-(ARPU_NOISE_SD ** 2) / 2,  sigma=ARPU_NOISE_SD,  size=n_sims)

        new_users    = acq_base * ((1 + acq_growth) ** m) * acq_mult
        churned_rate = (churn_base * churn_mult).clip(0.001, 0.30)
        arpu_realised = arpu_base * arpu_mult

        users = users * (1 - churned_rate) + new_users
        revenue[:, m] = users * arpu_realised

    return revenue


# ────────────────────────────────────────────────────────────────────
# 3. Sensitivity — flex each lever ±20% and measure 36-month NPV
# ────────────────────────────────────────────────────────────────────

def sensitivity_analysis() -> pd.DataFrame:
    """
    For each lever, simulate +20% and −20% and return delta in mean
    36-month cumulative revenue. Result is a tornado chart input.
    """
    base = simulate(n_sims=2_000).sum(axis=1).mean()
    out = []
    for name, kwarg, mult in [
        ("Acquisition",        "acq_base",   1.20),
        ("Acquisition",        "acq_base",   0.80),
        ("Churn rate",         "churn_base", 1.20),
        ("Churn rate",         "churn_base", 0.80),
        ("ARPU",               "arpu_base",  1.20),
        ("ARPU",               "arpu_base",  0.80),
        ("Acquisition growth", "acq_growth", 1.20),
        ("Acquisition growth", "acq_growth", 0.80),
    ]:
        defaults = dict(acq_base=ACQ_BASE, acq_growth=ACQ_GROWTH_PER_MONTH,
                       churn_base=CHURN_BASE, arpu_base=ARPU_BASE)
        defaults[kwarg] = defaults[kwarg] * mult
        rev = simulate(n_sims=2_000, **defaults).sum(axis=1).mean()
        out.append({"lever": name, "shift": "+20%" if mult > 1 else "-20%",
                   "rev_36m": rev, "delta": rev - base})
    return pd.DataFrame(out)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    print("Running revenue simulation…")
    print(f"  n_sims  = {N_SIMS:,}")
    print(f"  months  = {MONTHS}")
    print(f"  starting users = {STARTING_USERS:,}")
    print()

    revenue = simulate()

    # Percentile bands per month
    p10 = np.percentile(revenue, 10, axis=0)
    p50 = np.percentile(revenue, 50, axis=0)
    p90 = np.percentile(revenue, 90, axis=0)

    print("Forecast (monthly revenue, ₹ Cr):")
    for m in (11, 23, 35):
        print(f"  Month {m+1:>2}  P10 = ₹{p10[m]/1e7:>6.2f} Cr  |  P50 = ₹{p50[m]/1e7:>6.2f} Cr  |  P90 = ₹{p90[m]/1e7:>6.2f} Cr")

    # Cumulative
    cum_rev_36m = revenue.sum(axis=1)
    print("\n36-month cumulative revenue (₹ Cr):")
    print(f"  P10 = ₹{np.percentile(cum_rev_36m, 10)/1e7:.1f} Cr")
    print(f"  P50 = ₹{np.percentile(cum_rev_36m, 50)/1e7:.1f} Cr")
    print(f"  P90 = ₹{np.percentile(cum_rev_36m, 90)/1e7:.1f} Cr")
    print(f"  P10–P90 spread = ₹{(np.percentile(cum_rev_36m, 90) - np.percentile(cum_rev_36m, 10))/1e7:.1f} Cr")

    print("\nRunning sensitivity analysis…")
    sens = sensitivity_analysis()
    sens["delta_cr"] = sens["delta"] / 1e7
    print("\nLever sensitivity (delta in 36m cumulative revenue, ₹ Cr):")
    grouped = sens.groupby("lever")["delta_cr"].agg(["min", "max"])
    grouped["range_cr"] = grouped["max"] - grouped["min"]
    grouped = grouped.sort_values("range_cr", ascending=False)
    for lever, row in grouped.iterrows():
        print(f"  {lever:22s}  -20% to +20% spread: ₹{row['range_cr']:.1f} Cr")

    print("\nSaving plots…")

    # Fan chart
    months_axis = np.arange(1, MONTHS + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.fill_between(months_axis, p10/1e7, p90/1e7, alpha=0.18, color="#4A6FA5", label="P10–P90")
    ax.plot(months_axis, p50/1e7, color="#4A6FA5", lw=2.2, label="P50 (median)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly revenue (₹ Cr)")
    ax.set_title("36-month revenue forecast — Monte Carlo fan chart")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("revenue_fan.png", dpi=120)

    # Tornado chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    levers = grouped.index.tolist()[::-1]
    lows  = [grouped.loc[l, "min"] for l in levers]
    highs = [grouped.loc[l, "max"] for l in levers]
    ax.barh(levers, [h - l for h, l in zip(highs, lows)], left=lows,
            color="#4A6FA5", alpha=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Delta in 36m cumulative revenue (₹ Cr)")
    ax.set_title("Lever sensitivity — which lever moves the P&L most?")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("lever_sensitivity.png", dpi=120)

    print("  revenue_fan.png and lever_sensitivity.png saved.")
    print("\nDone.")


if __name__ == "__main__":
    main()
