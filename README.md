# Telecom Subscriber Churn Prediction

A machine learning project that predicts which telecom subscribers are likely to cancel their service — and more importantly, *why*. Built with XGBoost and SHAP, deployed as an interactive what-if dashboard.

**Live demo:** [telecom-churn-prediction.streamlit.app](https://telecom-churn-prediction-9ikaanuxdzednmpip56v3t.streamlit.app/)

---

## The Problem

Telecom companies lose thousands of subscribers every month. Acquiring a new customer costs 5–7x more than retaining one, so knowing *who* is about to leave — before they do — is worth a lot.

This project builds a churn classifier on real subscriber data (call records, contract type, monthly charges, service usage) and wraps it in a dashboard where retention teams can test specific customer profiles and see exactly what's driving the prediction.

---

## What's Inside

```
telecom-churn-prediction/
├── data/                        # Telco Customer Churn dataset (Kaggle)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory analysis
│   ├── 02_modeling.ipynb        # Model training and comparison
│   └── 03_shap.ipynb            # SHAP explainability
├── src/
│   ├── preprocess.py            # Cleaning, encoding, feature engineering
│   ├── train.py                 # Model training logic
│   └── evaluate.py              # Metrics and plots
├── models/
│   └── xgb_churn.pkl            # Saved XGBoost model
├── app/
│   └── streamlit_app.py         # What-if dashboard
└── requirements.txt
```

---

## Key Findings from EDA

Three features do most of the work:

- **Contract type** is the strongest signal by far. Month-to-month customers churn at nearly a 1:1 ratio. Two-year contract holders almost never leave.
- **Tenure** matters a lot in the first year. Customers who make it past 24 months are much less likely to churn.
- **Monthly charges** — churned customers pay ~$80/month on average vs ~$65 for those who stay. Higher bills with no long-term commitment is a bad combination.

---

## Models Compared

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.535 | 0.592 | 0.846 |
| Random Forest | 0.786 | 0.481 | 0.544 | 0.820 |
| **XGBoost** | **0.754** | **0.802** | **0.634** | **0.844** |

XGBoost wins on recall — it catches 80% of actual churners. For a retention team, missing a churner is more costly than a false alarm, so recall matters more than raw accuracy here.

---

## SHAP Explainability

Global feature importance shows Contract, tenure, and MonthlyCharges are the top three predictors — which matches what EDA showed.

The dashboard also shows a waterfall chart for any individual customer: which features pushed the prediction toward churn, and which ones pulled it back. A retention manager can look at this and know exactly what offer to make.

---

## Run It Locally

```bash
git clone git@github.com:Sai-manohar695/telecom-churn-prediction.git
cd telecom-churn-prediction

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), rename it to `telco_churn.csv`, and place it in the `data/` folder.

Run the notebooks in order (`01` → `02` → `03`), then launch the app:

```bash
cd app
streamlit run streamlit_app.py
```

---

## Tech Stack

- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Dashboard:** Streamlit
- **Data:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)

---

## Who This Is For

Built as a portfolio project targeting DS roles at **Airtel, Vodafone Idea, and Reliance Jio** — companies that explicitly list churn prediction and subscriber analytics in their job descriptions.
