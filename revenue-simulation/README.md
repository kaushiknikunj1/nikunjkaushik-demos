# Revenue Simulation — Monte Carlo Forecast

Self-contained, runnable demo of decision-grade revenue forecasting. Models 36-month revenue as a function of three uncertain levers — acquisition, churn, and ARPU — and runs 10,000 simulations to produce a P10/P50/P90 confidence band instead of a single deterministic line.

## What it does

1. Simulates 36 months × 10K paths of monthly revenue.
2. Each lever is sampled from a realistic distribution per month (log-normal noise around the base value), so the forecast naturally widens over time.
3. Produces percentile bands at months 12, 24, 36.
4. Runs sensitivity analysis: flex each lever ±20% and measure the change in 36-month NPV to identify which lever most moves the P&L.

## Run it

```bash
pip install -r requirements.txt
python revenue_simulation.py
```

Output: console summary + two plots:
- `revenue_fan.png` — P10/P50/P90 fan chart
- `lever_sensitivity.png` — tornado chart ranking levers by P&L impact

## Why simulate instead of forecast a single line

A point forecast is a confidence statement you can't defend. "We'll do ₹50 Cr in year 3" hides every uncertainty in the assumption. A fan chart says: "P50 ₹50 Cr, but the P10-P90 spread is ₹35 Cr — that range is what we should be planning against."

The tornado chart is the more important output. It tells leadership *which lever matters most*. In most consumer businesses, churn turns out to be roughly 2× the lever of acquisition over a 36-month horizon — which usually reshapes where the team spends its attention.

## Related

- [Full case study](https://nikunjkaushik.com/projects/revenue-simulation.html)
