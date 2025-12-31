# Early Burnout Risk Detector 🔥

An end-to-end Machine Learning + Streamlit application that predicts employee burnout risk (0–1) using workplace and mental fatigue indicators.

##  Live Demo
https://early-burnout-risk-detector-d6jghwfamevdhyvcgfhcfa.streamlit.app/

##  Project Overview
This project analyzes workplace and psychological factors to estimate burnout risk early, helping organizations and individuals take preventive action.

### Inputs include:
- Mental Fatigue Score
- Resource Allocation
- Work From Home availability
- Designation level
- Gender
- Company Type
- Employee Tenure

### Output:
- Burnout Risk Score (0 = Low, 1 = High)
- Risk band interpretation

## Modeling
- Compared Linear Regression, Ridge, Random Forest, and HistGradientBoosting
- Selected **HistGradientBoosting** for best performance
- Achieved:
  - **R² ≈ 0.90**
  - **RMSE ≈ 0.06**

##  Features & Insights
- Mental fatigue is the strongest predictor of burnout
- Higher workload increases burnout risk
- WFH availability reduces burnout risk
- High-risk employee profiling added for interpretability

##  Repository Structure
- `app.py` → Streamlit application
- `models/` → trained model, scaler, feature columns
- `reports/` → prediction outputs & feature importance
- `requirements.txt` → dependencies

##  Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py



⚠️ Disclaimer

This project is for educational and analytical purposes only and should not be used as a medical or clinical diagnostic tool.



