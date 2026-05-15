import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ · Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 740px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,.12);
    border: 1px solid rgba(99,179,237,.3);
    color: #63b3ed;
    font-family: 'DM Sans', sans-serif;
    font-size: .72rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    padding: .35rem .9rem;
    border-radius: 2rem;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -.02em;
    line-height: 1.1;
    margin: 0 0 .7rem;
    background: linear-gradient(135deg, #e8eaf0 30%, #63b3ed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* ── Card ── */
.card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 1.2rem;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: .85rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #4a90d9;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.card-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #4a90d9;
    border-radius: 50%;
}

/* ── Streamlit widget overrides ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #aab4c4 !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    letter-spacing: .02em !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: .6rem !important;
    color: #e8eaf0 !important;
}
.stSlider [data-baseweb="slider"] {
    padding-top: .3rem;
}
div[data-baseweb="select"] > div {
    background: #0d1526 !important;
    border-color: #1e2d42 !important;
    border-radius: .6rem !important;
    color: #e8eaf0 !important;
}

/* ── Submit button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: .04em;
    border: none;
    border-radius: .8rem;
    padding: .95rem 0;
    cursor: pointer;
    transition: opacity .2s, transform .15s;
    margin-top: .5rem;
}
.stButton > button:hover {
    opacity: .88;
    transform: translateY(-1px);
}

/* ── Result banners ── */
.result-approved {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #065f46;
    border-radius: 1.2rem;
    padding: 2rem;
    text-align: center;
    animation: fadeUp .4s ease;
}
.result-rejected {
    background: linear-gradient(135deg, #1c0a0a, #450a0a);
    border: 1px solid #7f1d1d;
    border-radius: 1.2rem;
    padding: 2rem;
    text-align: center;
    animation: fadeUp .4s ease;
}
.result-icon { font-size: 3rem; margin-bottom: .5rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
}
.result-approved .result-label { color: #34d399; }
.result-rejected .result-label { color: #f87171; }
.result-sub { color: #9ca3af; font-size: .9rem; margin-top: .4rem; }

/* ── Confidence bar ── */
.prob-row { display: flex; align-items: center; gap: .8rem; margin-top: 1.2rem; justify-content: center; }
.prob-label { font-size: .8rem; color: #6b7280; min-width: 60px; text-align: right; }
.prob-track {
    flex: 1; max-width: 260px;
    height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden;
}
.prob-fill-green { height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 4px; transition: width .6s ease; }
.prob-fill-red   { height: 100%; background: linear-gradient(90deg, #ef4444, #f87171); border-radius: 4px; transition: width .6s ease; }
.prob-pct { font-size: .82rem; font-family: 'Syne', sans-serif; font-weight: 700; min-width: 44px; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e2a3a; margin: 1.5rem 0; }

@keyframes fadeUp {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="loan_model.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_input_df(data: dict, export_data: dict) -> pd.DataFrame:
    """Replicate the notebook's preprocessing on a single row."""
    encoders   = export_data["encoders"]
    scaler     = export_data["scaler"]
    scale_cols = export_data["scale_cols"]
    cate_cols  = export_data["cate_cols"]

    row = pd.DataFrame([data])

    # Encode categoricals
    for col in cate_cols:
        le = encoders[col]
        row[col] = le.transform(row[col])

    # Scale numeric cols
    row[scale_cols] = scaler.transform(row[scale_cols])
    return row


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI · Risk Assessment</div>
  <h1>Loan Approval<br>Predictor</h1>
  <p>Fill in the applicant details below to get an instant decision.</p>
</div>
""", unsafe_allow_html=True)

export_data = load_model()

if export_data is None:
    st.warning(
        "**Model file not found.**  "
        "Run the notebook to generate `loan_model.pkl`, then place it in the "
        "same directory as this app.",
        icon="⚠️",
    )
    st.stop()

# ── Section 1 · Personal Info ──────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=28)
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_education = st.selectbox(
        "Education Level",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    )
with col2:
    person_income = st.number_input("Annual Income ($)", min_value=0, value=55000, step=1000)
    person_emp_exp = st.number_input("Employment Experience (yrs)", min_value=0, max_value=60, value=3)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2 · Loan Details ───────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Loan Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000, step=500)
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
with col4:
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
    loan_percent_income = round(loan_amnt / max(person_income, 1), 4)
    st.metric("Loan-to-Income Ratio", f"{loan_percent_income:.2%}")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3 · Credit Profile ─────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Credit Profile</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    cb_person_cred_hist_length = st.number_input("Credit History (yrs)", min_value=0, max_value=50, value=5)
with col6:
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Analyse & Predict"):
    applicant = {
        "person_age": float(person_age),
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": float(person_income),
        "person_emp_exp": int(person_emp_exp),
        "person_home_ownership": person_home_ownership,
        "loan_amnt": float(loan_amnt),
        "loan_intent": loan_intent,
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(loan_percent_income),
        "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
        "credit_score": int(credit_score),
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }

    try:
        X_input = build_input_df(applicant, export_data)
        model = export_data["model"]
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]   # [prob_reject, prob_approve]

        approved   = int(prediction) == 1
        p_approve  = float(proba[1]) * 100
        p_reject   = float(proba[0]) * 100

        if approved:
            st.markdown(f"""
            <div class="result-approved">
              <div class="result-icon">✅</div>
              <div class="result-label">Loan Approved</div>
              <div class="result-sub">The model recommends approving this application.</div>
              <div class="prob-row">
                <span class="prob-label">Approve</span>
                <div class="prob-track"><div class="prob-fill-green" style="width:{p_approve:.0f}%"></div></div>
                <span class="prob-pct" style="color:#34d399">{p_approve:.1f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-label">Reject</span>
                <div class="prob-track"><div class="prob-fill-red" style="width:{p_reject:.0f}%"></div></div>
                <span class="prob-pct" style="color:#f87171">{p_reject:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
              <div class="result-icon">❌</div>
              <div class="result-label">Loan Rejected</div>
              <div class="result-sub">The model does not recommend approving this application.</div>
              <div class="prob-row">
                <span class="prob-label">Approve</span>
                <div class="prob-track"><div class="prob-fill-green" style="width:{p_approve:.0f}%"></div></div>
                <span class="prob-pct" style="color:#34d399">{p_approve:.1f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-label">Reject</span>
                <div class="prob-track"><div class="prob-fill-red" style="width:{p_reject:.0f}%"></div></div>
                <span class="prob-pct" style="color:#f87171">{p_reject:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#3d4a5c;font-size:.78rem;'>Powered by Random Forest · For informational purposes only</p>",
    unsafe_allow_html=True,
)
