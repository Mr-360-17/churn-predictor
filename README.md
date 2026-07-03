# Customer Churn Predictor 🔮

An end-to-end churn prediction tool for B2B SaaS — from messy data to a non-technical dashboard with plain-English explanations.

## What It Does

- Predicts which customers are likely to churn in the next **90 days**
- Explains **why** each account is at risk (not just a score)
- Gives the CS team **actionable talking points** for each call
- Works with messy real-world data (string booleans, mixed NPS values, missing fields)

## Live Demo

👉 [Open Dashboard](https://your-app.streamlit.app)

## Components

| File | Purpose |
|---|---|
| `churn_data.csv` | Synthetic dataset (1,500 accounts, 15 features, deliberately messy) |
| `churn_predictor.ipynb` | EDA, cleaning decisions, model training, SHAP analysis |
| `app.py` | Streamlit dashboard for non-technical users |
| `model_artifacts.pkl` | Trained XGBoost model + metadata |
| `REPORT.md` | 600-word memo for product managers |

## Model

- **Algorithm:** XGBoost (gradient boosting)
- **Target:** Churn within 90 days
- **ROC-AUC:** 0.66 on held-out test set
- **Top features:** login_count_30d, last_login_days_ago, feature_adoption_pct

## How to Run Locally

```bash
git clone https://github.com/Mr-360-17/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt
# Run notebook first to generate model_artifacts.pkl
jupyter notebook churn_predictor.ipynb
# Then launch the app
streamlit run app.py
```

## Data Cleaning Decisions

| Issue | Fix | Reasoning |
|---|---|---|
| `onboarding_complete` has 6 string variants | Map all to 0/1 | "True"/"true"/"1" → 1, rest → 0 |
| `nps_score` has "N/A" and "MISSING" | Median imputation + missingness flag | Missingness itself is a signal |
| `company_size` has "MISSING" | Replace with "Unknown" category | Preserve the observation |
| `cs_health_score` has 15% nulls | Median imputation + missingness flag | Same reasoning as NPS |

## Reflection

The hardest part was not the model — it was the problem framing. Choosing 90 days as the prediction horizon, defining what counts as churn, and deciding which signals to include required more thought than the XGBoost hyperparameters. The memo in REPORT.md captures these decisions for stakeholders who will never look at the notebook.
