import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load saved artifacts
# ----------------------------
model = joblib.load("models/burnout_hgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Early Burnout Risk Detector", layout="centered")

st.title(" Early Burnout Risk Detector")
st.caption("Predicts a burnout risk score (0–1). Educational project — not a diagnosis tool.")

st.write(
    "Enter workplace and fatigue indicators to get an estimated **Burnout Risk Score**. "
    "This model was trained using multiple algorithms and the best performer was selected."
)

st.divider()

# ----------------------------
# Sidebar inputs (cleaner UI)
# ----------------------------
st.sidebar.header("Inputs")

designation = st.sidebar.slider("Designation (role level)", 0.0, 5.0, 2.0, 0.1)
resource_allocation = st.sidebar.slider("Resource Allocation (workload)", 0.0, 10.0, 5.0, 0.1)
mental_fatigue = st.sidebar.slider("Mental Fatigue Score", 0.0, 10.0, 5.0, 0.1)
tenure_years = st.sidebar.slider("Tenure (years)", 0.0, 1.0, 0.50, 0.01)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
company_type = st.sidebar.selectbox("Company Type", ["Product", "Service"])
wfh = st.sidebar.selectbox("WFH Setup Available", ["No", "Yes"])

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

# Create feature vector in exact training column order
x = np.array([input_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)

# Scale using the same scaler
x_scaled = scaler.transform(x)

# ----------------------------
# Predict section
# ----------------------------
st.subheader(" Prediction")

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Burnout Risk"):
        pred = float(model.predict(x_scaled)[0])
        pred = max(0.0, min(1.0, pred))  # clamp for display

        st.metric("Burnout Risk Score (0–1)", f"{pred:.3f}")

        if pred < 0.33:
            st.success("Low risk — maintain healthy workload and recovery habits.")
        elif pred < 0.66:
            st.warning("Moderate risk — monitor fatigue and adjust workload if possible.")
        else:
            st.error("High risk — consider workload reduction and support interventions.")

with col2:
    st.markdown("**What usually drives burnout most (from this project):**")
    st.markdown("- Mental Fatigue Score (strongest)")
    st.markdown("- Resource Allocation (workload)")
    st.markdown("- WFH availability (protective factor)")

st.divider()

# ----------------------------
# Charts section
# ----------------------------
st.subheader("Model Insights")

try:
    preds = pd.read_csv("reports/test_predictions.csv")
    imps = pd.read_csv("reports/permutation_importance.csv")

    st.write("**Predicted Burnout Risk Distribution (Test Set)**")
    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(preds["y_pred"], bins=40)
    plt.xlabel("Predicted Burnout Risk Score")
    plt.ylabel("Count")
    plt.title("Burnout Risk Distribution")
    st.pyplot(fig1)

    st.write("**Top Drivers of Burnout Risk (Permutation Importance)**")
    top_k = 7
    top = imps.head(top_k).iloc[::-1]

    fig2 = plt.figure(figsize=(7, 4))
    plt.barh(top["Feature"], top["ImportanceMean"])
    plt.xlabel("Importance (mean drop in R² when shuffled)")
    plt.title("Top Feature Importances")
    st.pyplot(fig2)

except Exception:
    st.info("Charts will appear after report files exist in the /reports folder.")
