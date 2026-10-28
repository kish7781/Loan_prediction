import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="Automated Loan Underwriter", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B101E; color: #E2E8F0; }
[data-testid="stSidebar"] { background-color: #111827; }
[data-testid="stSidebar"] * { color: #E2E8F0; }
.main-title { font-size: 38px; font-weight: 800; color: #38BDF8; margin-bottom: 0px; }
.subtitle { color: #9CA3AF; font-size: 20px; margin-bottom: 20px; }
.section-title { color: #F8FAFC; font-size: 30px; font-weight: 700; margin-top: 16px; margin-bottom: 15px; border-bottom: 1px solid #1F2937; padding-bottom: 10px;}
div.stButton > button { background-color: #38BDF8; color: #0B101E; border-radius: 8px; border: none; padding: 12px 24px; font-weight: 800; width: 100%; transition: 0.3s; }
div.stButton > button:hover { background-color: #0284C7; color: white; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
try:
    model = joblib.load(BASE_DIR / "xgboost_underwriting_model.joblib")
    preprocessor = joblib.load(BASE_DIR / "underwriting_preprocessor.joblib")
except Exception as e:
    st.error(f"Error loading models. Make sure joblib files exist. {e}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">🏦 LOAN APPROVAL PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-driven credit risk assessment combining CIBIL behavior with institutional collateral valuation.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("underwriting_form"):
    
    st.markdown('<div class="section-title">Applicant Profile & Loan Request</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with c2:
        self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    with c3:
        no_of_dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)
    with c4:
        loan_term = st.slider("Loan Term (Years)", 1, 30, 10)

    st.markdown('<div class="section-title">Financials & Credit Score</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        income_annum = st.number_input("Annual Income (INR)", min_value=0, value=1500000, step=100000)
    with f2:
        loan_amount = st.number_input("Requested Loan Amount (INR)", min_value=0, value=5000000, step=100000)
    with f3:
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

    st.markdown('<div class="section-title">Collateral & Assets Valuation</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        residential_assets_value = st.number_input("Residential Assets", min_value=0, value=2000000, step=100000)
    with a2:
        commercial_assets_value = st.number_input("Commercial Assets", min_value=0, value=0, step=100000)
    with a3:
        luxury_assets_value = st.number_input("Luxury Assets", min_value=0, value=500000, step=100000)
    with a4:
        bank_asset_value = st.number_input("Bank Assets (Liquid)", min_value=0, value=1000000, step=100000)

    st.write("")
    submitted = st.form_submit_button("⚖️ Execute Credit Assessment")

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if submitted:
    input_data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })
    
    # Transform and Predict
    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)[0]
    probabilities = model.predict_proba(transformed_data)[0]
    
    st.markdown("---")
    st.markdown('<div class="section-title">Underwriting Verdict</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        st.success("✅ **LOAN APPROVED**")
        st.write("The applicant's credit profile and collateral backing meet institutional risk parameters. Cleared for Straight-Through Processing (STP).")
    else:
        st.error("🚫 **LOAN REJECTED**")
        st.write("The applicant represents a high default risk. Ensure CIBIL score is sufficient and that requested loan amount does not heavily outweigh collateral and income.")

    # Probability Chart
    prob_df = pd.DataFrame({
        "Status": ["Rejected (0)", "Approved (1)"],
        "Probability (%)": [round(probabilities[0] * 100, 2), round(probabilities[1] * 100, 2)]
    })

    fig = go.Figure(go.Bar(
        x=prob_df["Status"], y=prob_df["Probability (%)"],
        text=prob_df["Probability (%)"].astype(str) + "%", textposition="auto",
        marker_color=["#EF4444", "#10B981"] 
    ))

    fig.update_layout(
        title="Approval Confidence Metric",
        yaxis=dict(range=[0, 100]), plot_bgcolor="#111827", paper_bgcolor="#0B101E", font=dict(color="#E2E8F0")
    )
    st.plotly_chart(fig, use_container_width=True)