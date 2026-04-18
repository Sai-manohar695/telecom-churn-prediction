import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import load_and_preprocess

# Load model
model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_churn.pkl'))

# Load data to get feature names and scaler context
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_churn.csv')
)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Page config
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

st.title("📡 Telecom Subscriber Churn Predictor")
st.markdown("Adjust customer details below to predict churn probability and understand why.")

# Sidebar inputs
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges (₹)", 20, 120, 65)
total_charges = st.sidebar.slider("Total Charges (₹)", 0, 9000, 1000)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Map inputs to encoded values
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
security_map = {"No": 0, "No internet service": 1, "Yes": 2}
support_map = {"No": 0, "No internet service": 1, "Yes": 2}
payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}
binary_map = {"No": 0, "Yes": 1}

tenure_group = pd.cut([tenure], bins=[0,12,24,48,60,72], labels=[0,1,2,3,4])[0]
tenure_group = int(tenure_group) if pd.notna(tenure_group) else 0
charges_ratio = total_charges / (tenure + 1)

input_data = pd.DataFrame([{
    'gender': 0,
    'SeniorCitizen': binary_map[senior_citizen],
    'Partner': binary_map[partner],
    'Dependents': binary_map[dependents],
    'tenure': tenure,
    'PhoneService': 1,
    'MultipleLines': 0,
    'InternetService': internet_map[internet_service],
    'OnlineSecurity': security_map[online_security],
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': support_map[tech_support],
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'Contract': contract_map[contract],
    'PaperlessBilling': binary_map[paperless_billing],
    'PaymentMethod': payment_map[payment_method],
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'tenure_group': tenure_group,
    'charges_per_month_ratio': charges_ratio
}])

# Scale input using training data stats
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)

# Prediction
prob = model.predict_proba(input_scaled)[0][1]
prediction = "Will Churn" if prob >= 0.5 else "Will Stay"

# Display prediction
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if prediction == "Will Churn":
        st.error(f"⚠️ {prediction}")
    else:
        st.success(f"✅ {prediction}")
    st.metric("Churn Probability", f"{prob:.1%}")

with col2:
    st.subheader("Risk Level")
    if prob >= 0.7:
        st.error("🔴 High Risk")
    elif prob >= 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

# SHAP explanation
st.subheader("Why is the model predicting this?")
shap_vals = explainer.shap_values(input_scaled)

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=input_scaled.values[0],
        feature_names=feature_names
    ),
    show=False
)
st.pyplot(fig)
plt.close()

st.markdown("---")
st.caption("Built with XGBoost + SHAP | Telecom Churn Prediction Project")