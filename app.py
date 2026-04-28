import streamlit as st
import joblib
import pandas as pd

# =========================
# INLINE CSS (NO FILE NEEDED)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}

h1 {
    color: #00c3ff;
    text-align: center;
}

.stButton>button {
    background-color: #00c3ff;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}

.stNumberInput input {
    background-color: #1c1f26;
    color: white;
}

.success-box {
    background-color: #0f5132;
    padding: 15px;
    border-radius: 10px;
    color: white;
}

.error-box {
    background-color: #842029;
    padding: 15px;
    border-radius: 10px;
    color: white;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

model = load_model()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Loan Risk Engine", layout="centered")

st.title("💳 Loan Default Risk Engine")
st.markdown("### Predict customer default risk")

# =========================
# INPUT SECTION
# =========================
st.subheader("📊 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=0, value=100000)
    credit = st.number_input("Credit Amount", min_value=0, value=500000)
    goods_price = st.number_input("Goods Price", min_value=0, value=450000)

with col2:
    annuity = st.number_input("Loan Annuity", min_value=0, value=20000)
    employment_years = st.number_input("Years Employed", min_value=0, value=5)
    age_years = st.number_input("Age (years)", min_value=18, value=30)

# =========================
# CONVERT TO MODEL FORMAT
# =========================
employment_days = -employment_years * 365
age_days = -age_years * 365

# =========================
# CREATE INPUT DATA
# =========================
def create_input():
    data = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT_x": credit,
        "AMT_GOODS_PRICE": goods_price,
        "AMT_ANNUITY": annuity,
        "DAYS_EMPLOYED": employment_days,
        "DAYS_BIRTH": age_days
    }])

    # match model features
    expected = model.feature_names_in_

    for col in expected:
        if col not in data.columns:
            data[col] = 0

    return data[expected]

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Risk"):

    input_data = create_input()

    prob = model.predict_proba(input_data)[:, 1][0]

    st.markdown("---")

    # Risk display
    st.subheader("📈 Risk Score")
    st.progress(float(prob))

    st.write(f"**Probability of Default:** {prob:.2%}")

    # Decision
    if prob > 0.4:
        st.markdown(
            """
            <div class="error-box">
            ⚠️ High Risk of Default<br>
            Recommendation: Reject or manual review
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success-box">
            ✅ Low Risk Applicant<br>
            Recommendation: Approve
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Model: XGBoost | Features: Behavioral + Financial | Explainable via SHAP 🚀")