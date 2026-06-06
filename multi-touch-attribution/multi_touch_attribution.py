"""
Multi-Touch Attribution — Markov chain attribution
===================================================

Self-contained demo. Generates synthetic user journeys across 7 channels,
fits a Markov chain on the transitions, computes per-channel removal effect,
and compares to last-click and first-click attribution.

Run:
    pip install -r requirements.txt
    python multi_touch_attribution.py

Output:
    - Console table comparing channel weights: last-click / first-click / Markov
    - attribution_compare.png: side-by-side bars
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

CHANNELS = ["PaidSearch", "PaidSocial", "Display", "Email", "Organic", "Direct", "Referral"]
STATES   = ["Start"] + CHANNELS + ["Conversion", "Null"]
N_STATES = len(STATES)
IDX = {s: i for i, s in enumerate(STATES)}


# ────────────────────────────────────────────────────────────────────
# 1. Synthetic journeys
#
# Each user has a journey of 1–8 touches across the 7 channels, ending in
# Conversion or Null. The channel mix is biased so that:
#   - Display is a common discovery channel (high first-touch frequency)
#   - PaidSearch tends to close the journey (high last-touch frequency)
#   - Email shows up in the middle of nurture sequences
#   - Direct often appears as a returning-user last touch
#
# This mimics the "last-click overcredits closers" pattern.
# ────────────────────────────────────────────────────────────────────

# Channel transition biases (rows = from, cols = to)
# This generative model is what Markov should be able to recover.

START_PROBS = np.array([0.18, 0.12, 0.28, 0.06, 0.18, 0.10, 0.08])  # entry channel mix
# Roughly: 28% start at Display, 18% PaidSearch / Organic, etc.

# Within-journey transitions (each channel × each channel)
# Display tends to → PaidSearch (nudges intent searches)
# Email tends to → PaidSearch / Direct (converts the nurture)
TRANSITION_BIAS = np.array([
    # PS    PSoc  Disp  Email Org   Direct Ref
    [0.20, 0.10, 0.15, 0.10, 0.10, 0.25, 0.10],  # PaidSearch
    [0.20, 0.15, 0.15, 0.10, 0.10, 0.20, 0.10],  # PaidSocial
    [0.35, 0.15, 0.10, 0.10, 0.10, 0.15, 0.05],  # Display → PaidSearch
    [0.30, 0.10, 0.05, 0.10, 0.10, 0.30, 0.05],  # Email → PaidSearch / Direct
    [0.25, 0.10, 0.10, 0.05, 0.20, 0.20, 0.10],  # Organic
    [0.15, 0.10, 0.05, 0.10, 0.15, 0.40, 0.05],  # Direct (sticky)
    [0.30, 0.15, 0.10, 0.05, 0.20, 0.15, 0.05],  # Referral
])

# Per-channel conversion probability when this is the LAST touch
LAST_TOUCH_CONVERT = {
    "PaidSearch": 0.16, "PaidSocial": 0.08, "Display": 0.04, "Email": 0.12,
    "Organic":    0.10, "Direct":     0.20, "Referral":   0.07,
}

# Per-step probability of ending the journey
END_PROB = 0.32


def generate_journey() -> tuple[list[str], int]:
    """Return (path, converted)."""
    path = []
    # First touch
    cur = np.random.choice(CHANNELS, p=START_PROBS)
    path.append(cur)
    while True:
        if np.random.rand() < END_PROB:
            # End the journey — decide if it converts based on last touch
            converted = int(np.random.rand() < LAST_TOUCH_CONVERT[cur])
            return path, converted
        # Continue: sample next channel from transition bias
        nxt = np.random.choice(CHANNELS, p=TRANSITION_BIAS[CHANNELS.index(cur)])
        path.append(nxt)
        cur = nxt
        if len(path) >= 8:
            converted = int(np.random.rand() < LAST_TOUCH_CONVERT[cur])
            return path, converted


def generate_journeys(n: int = 250_000) -> list[tuple[list[str], int]]:
    return [generate_journey() for _ in range(n)]


# ────────────────────────────────────────────────────────────────────
# 2. Heuristics — last-click and first-click attribution
# ────────────────────────────────────────────────────────────────────

def last_click_attribution(journeys) -> dict[str, float]:
    counts = {ch: 0 for ch in CHANNELS}
    for path, converted in journeys:
        if converted:
            counts[path[-1]] += 1
    total = sum(counts.values())
    return {ch: c / total for ch, c in counts.items()}


def first_click_attribution(journeys) -> dict[str, float]:
    counts = {ch: 0 for ch in CHANNELS}
    for path, converted in journeys:
        if converted:
            counts[path[0]] += 1
    total = sum(counts.values())
    return {ch: c / total for ch, c in counts.items()}


# ────────────────────────────────────────────────────────────────────
# 3. Markov chain attribution
#
# Build absorbing Markov chain. States: Start → channels → Conversion / Null.
# Compute baseline P(Conversion | Start). For each channel, "remove" it
# (route all its outgoing edges to Null) and recompute. The drop in
# conversion probability is that channel's removal effect.
# ────────────────────────────────────────────────────────────────────

def build_transition_matrix(journeys) -> np.ndarray:
    P = np.zeros((N_STATES, N_STATES))
    for path, converted in journeys:
        seq = ["Start"] + path + (["Conversion"] if converted else ["Null"])
        for a, b in zip(seq[:-1], seq[1:]):
            P[IDX[a], IDX[b]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(row_sums > 0, P / row_sums, 0)
    # Conversion and Null are absorbing — self-loop
    P[IDX["Conversion"], IDX["Conversion"]] = 1.0
    P[IDX["Null"],       IDX["Null"]]       = 1.0
    return P


def conversion_probability(P: np.ndarray) -> float:
    """For absorbing Markov chain — return P(reach Conversion | start at Start)."""
    transient = [i for i, s in enumerate(STATES) if s not in ("Conversion", "Null")]
    absorbing = [IDX["Conversion"], IDX["Null"]]
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    try:
        N = np.linalg.inv(np.eye(len(transient)) - Q)
    except np.linalg.LinAlgError:
        return 0.0
    B = N @ R   # rows = starting state, cols = [Conversion, Null]
    return float(B[transient.index(IDX["Start"]), 0])


def markov_attribution(journeys) -> dict[str, float]:
    """
    For each channel: 'remove' it by routing every visit to that channel
    directly to Null (the user gives up). The drop in P(Conversion | Start)
    is the channel's removal effect — its contribution to the system.
    Normalising across channels gives attribution weights.
    """
    P = build_transition_matrix(journeys)
    baseline = conversion_probability(P)

    removal_effects = {}
    for ch in CHANNELS:
        P_minus = P.copy()
        idx = IDX[ch]
        # Any time the walk reaches `ch`, it immediately goes to Null.
        # We model this by making `ch` an absorbing state that absorbs to Null.
        P_minus[idx, :]           = 0.0
        P_minus[idx, IDX["Null"]] = 1.0
        new_p = conversion_probability(P_minus)
        removal_effects[ch] = max(baseline - new_p, 0.0)

    total = sum(removal_effects.values())
    return {ch: re / total for ch, re in removal_effects.items()} if total > 0 else removal_effects


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic journeys…")
    journeys = generate_journeys(n=250_000)
    n_conv = sum(c for _, c in journeys)
    avg_touches = float(np.mean([len(p) for p, _ in journeys]))
    print(f"  n journeys = {len(journeys):,}  |  conversions = {n_conv:,} ({n_conv/len(journeys):.1%})")
    print(f"  median touches/journey = {int(np.median([len(p) for p, _ in journeys]))}")
    print(f"  mean   touches/journey = {avg_touches:.2f}")

    print("\nComputing attributions…")
    lc  = last_click_attribution(journeys)
    fc  = first_click_attribution(journeys)
    mc  = markov_attribution(journeys)

    df = pd.DataFrame({"Last-click": lc, "First-click": fc, "Markov": mc}).reindex(CHANNELS)
    df["Markov ÷ Last-click"] = (df["Markov"] / df["Last-click"]).round(2)
    formatted = df.copy()
    for col in ["Last-click", "First-click", "Markov"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.2%}")
    print("\n" + formatted.to_string())

    print("\nInterpretation:")
    over  = df[df["Markov ÷ Last-click"] < 0.85]
    under = df[df["Markov ÷ Last-click"] > 1.15]
    for ch, row in over.iterrows():
        print(f"  {ch:12s} OVERCREDITED by last-click  ({row['Last-click']:.0%} → Markov says {row['Markov']:.0%})")
    for ch, row in under.iterrows():
        print(f"  {ch:12s} UNDERCREDITED by last-click ({row['Last-click']:.0%} → Markov says {row['Markov']:.0%})")

    print("\nSaving attribution_compare.png …")
    ax = df[["Last-click", "Markov"]].plot.barh(figsize=(9, 5.5), color=["#9aa6b8", "#4A6FA5"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of attributed conversions")
    ax.set_title("Channel attribution — Last-click vs Markov chain")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("attribution_compare.png", dpi=120)
    print("  Done.")


if __name__ == "__main__":
    main()
