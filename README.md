# Analytics & ML Demos — Nikunj Kaushik

Runnable, self-contained demonstrations of the analytical methods discussed in my [project case studies](https://nikunjkaushik.com/projects.html). Each demo uses synthetic data — no proprietary information from any employer — but reproduces the structure, methodology, and outputs of real work.

Each subfolder is independent. Pick one, install its `requirements.txt`, and run.

| Folder | What it demonstrates |
|---|---|
| [`uplift-modeling/`](./uplift-modeling) | T-Learner with XGBoost · Qini curve validation · segment identification (Persuadables, Sure Things, Do-Not-Disturbs) |
| [`multi-touch-attribution/`](./multi-touch-attribution) | Markov chain attribution · removal-effect calculation · comparison to last-click & first-click |
| [`churn-prediction/`](./churn-prediction) | XGBoost churn classifier · SHAP-driven per-user interpretability · CRM-actionable outputs |
| [`revenue-simulation/`](./revenue-simulation) | Monte Carlo revenue forecasting · P10/P50/P90 fan chart · lever sensitivity analysis |

## Usage

```bash
cd uplift-modeling
pip install -r requirements.txt
python uplift_modeling.py
```

Each script prints a summary to the console and saves one or more diagnostic plots to the same folder.

## About

I'm an Analytics & Data Science leader with 9+ years of experience at Razorpay, MPL, Ola, and Mu Sigma. These demos are companion code for the full case studies on my [website](https://nikunjkaushik.com).

- 📝 [Blog](https://nikunjkaushik.com/blog.html)
- 💼 [LinkedIn](https://www.linkedin.com/in/nikunjkaushik/)
