import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Load saved artifacts
model = joblib.load("models/burnout_hgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

st.title("🔥 Burnout Risk Predictor")
st.caption("Predicts a burnout risk score (0–1). Not a medical diagnosis.")

st.write("Enter employee/work conditions below, then click **Predict**.")

# Inputs (match your training features)
designation = st.slider("Designation (role level)", 0.0, 5.0, 2.0, 0.1)
resource_allocation = st.slider("Resource Allocation (workload)", 0.0, 10.0, 5.0, 0.1)
mental_fatigue = st.slider("Mental Fatigue Score", 0.0, 10.0, 5.0, 0.1)
tenure_years = st.slider("Tenure (years)", 0.0, 1.0, 0.50, 0.01)

gender = st.selectbox("Gender", ["Female", "Male"])
company_type = st.selectbox("Company Type", ["Product", "Service"])
wfh = st.selectbox("WFH Setup Available", ["No", "Yes"])

# Build input dict in the same format as training
input_dict = {
    "Designation": designation,
    "Resource Allocation": resource_allocation,
    "Mental Fatigue Score": mental_fatigue,
    "Tenure_Years": tenure_years,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Company Type_Service": 1 if company_type == "Service" else 0,
    "WFH Setup Available_Yes": 1 if wfh == "Yes" else 0,
}

# Create feature vector in the exact training column order
x = np.array([input_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)

# Scale using the same scaler
x_scaled = scaler.transform(x)

if st.button("Predict Burnout Risk"):
    pred = float(model.predict(x_scaled)[0])

    # clamp for display
    pred = max(0.0, min(1.0, pred))

    st.metric("Burnout Risk Score (0–1)", f"{pred:.3f}")

    if pred < 0.33:
        st.success("Low risk")
    elif pred < 0.66:
        st.warning("Moderate risk")
    else:
        st.error("High risk")

st.divider()
st.subheader("📊 Model Insights")

try:
    preds = pd.read_csv("reports/test_predictions.csv")
    imps = pd.read_csv("reports/permutation_importance.csv")

    # Chart 1: Distribution of predicted burnout risk
    st.write("**Predicted Burnout Risk Distribution (Test Set)**")
    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(preds["y_pred"], bins=40)
    plt.xlabel("Predicted Burnout Risk Score")
    plt.ylabel("Count")
    plt.title("Burnout Risk Distribution")
    st.pyplot(fig1)

    # Chart 2: Feature importance (Permutation Importance)
    st.write("**Top Drivers of Burnout Risk (Permutation Importance)**")
    top_k = 7
    top = imps.head(top_k).iloc[::-1]  # reverse for nice horizontal bars

    fig2 = plt.figure(figsize=(7, 4))
    plt.barh(top["Feature"], top["ImportanceMean"])
    plt.xlabel("Importance (mean drop in R² when shuffled)")
    plt.title("Top Feature Importances")
    st.pyplot(fig2)

except Exception as e:
    st.info("Charts will appear after report files exist in the /reports folder.")


