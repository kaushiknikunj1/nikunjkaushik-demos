# Multi-Touch Attribution — Markov Chain Demo

Self-contained, runnable demo of data-driven attribution. Generates synthetic user journeys with realistic channel-mix biases (Display is a discovery channel, Paid Search closes, Direct is sticky), then attributes credit using three methods and shows where they disagree.

## What it does

1. Generates 250K synthetic journeys across 7 channels with a generative process where Display drives intent into Paid Search.
2. Computes attribution using three methods:
   - **Last-click** (status quo most teams ship with)
   - **First-click** (the discovery-biased alternative)
   - **Markov chain** with removal-effect calculation (data-driven)
3. Side-by-side comparison shows which channels are over- or under-credited by last-click.

## Run it

```bash
pip install -r requirements.txt
python multi_touch_attribution.py
```

Output: console table + `attribution_compare.png` horizontal bar chart.

## What you'll see

The synthetic data is rigged so that Display is a load-bearing discovery channel that rarely closes the click. Last-click systematically undercredits it. Markov recovers the truth.

This is the same pattern any real growth team will find in their data — the question is just whether they've ever looked.

## Related

- [Full case study](https://nikunjkaushik.com/projects/multi-touch-attribution.html)
- [Blog post](https://nikunjkaushik.com/blog/multi-touch-attribution.html)
