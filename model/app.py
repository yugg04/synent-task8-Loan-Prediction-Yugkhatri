import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- App Configuration ---
APP_TITLE = "Loan Underwriting Portal"
BRAND_COLOR = "#2c3e50"  # Professional Navy/Slate
MODEL_PATH = "loan_model.pkl"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🏦")

def apply_custom_css():
    st.markdown(f"""
        <style>
        .main {{ background-color: #ffffff; }}
        .stButton>button {{
            width: 100%; border-radius: 4px; height: 3rem;
            background-color: {BRAND_COLOR}; color: white;
            font-weight: 500; border: none;
        }}
        .report-header {{ 
            font-size: 20px; font-weight: 600; color: #1a1a1a; 
            padding-bottom: 10px; border-bottom: 1px solid #eee;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 24px; }}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets(path):
    """Loads backend model components."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None

# --- Main Application ---
apply_custom_css()
assets = load_assets(MODEL_PATH)

if not assets:
    st.warning("📋 System Setup: Please ensure the 'loan_model.pkl' file is in the directory to begin.")
else:
    # Asset unpacking
    clf = assets["model"]
    label_map = assets["encoders"]
    feature_scaler = assets["scaler"]
    numeric_features = assets["scale_cols"]

    st.title(APP_TITLE)
    st.markdown("Internal Credit Review Tool | v2.4")
    st.divider()

    # --- Input Sections ---
    with st.sidebar:
        st.subheader("Applicant Details")
        u_age = st.number_input("Age", 18, 95, 28)
        u_gender = st.selectbox("Gender", ["male", "female"])
        u_edu = st.selectbox("Education Level", ["High School", "Bachelor", "Associate", "Master", "Doctorate"])
        u_home = st.selectbox("Housing Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financials")
        u_income = st.number_input("Annual Income ($)", min_value=0, value=55000)
        u_exp = st.slider("Employment Length (Years)", 0, 45, 5)
        u_loan = st.number_input("Requested Loan Amount ($)", min_value=0, value=10000)

    with col2:
        st.subheader("Credit Metrics")
        u_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        u_score = st.number_input("Credit Score", 300, 850, 680)
        u_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 10.5)
        u_history = st.slider("Credit History (Years)", 0, 30, 4)
        u_default = st.selectbox("Prior Default", ["No", "Yes"])

    loan_inc_ratio = round(u_loan / u_income, 4) if u_income > 0 else 0.0

    st.write("---")
    if st.button("RUN CREDIT REVIEW"):
        # Construct Input
        raw_input = pd.DataFrame([{
            'person_age': u_age, 'person_gender': u_gender, 'person_education': u_edu,
            'person_income': u_income, 'person_emp_exp': u_exp, 'person_home_ownership': u_home,
            'loan_amnt': u_loan, 'loan_intent': u_intent, 'loan_int_rate': u_rate,
            'loan_percent_income': loan_inc_ratio, 'cb_person_cred_hist_length': u_history,
            'credit_score': u_score, 'previous_loan_defaults_on_file': u_default
        }])

        try:
            # Transformation
            processed_input = raw_input.copy()
            for col, encoder in label_map.items():
                processed_input[col] = encoder.transform(processed_input[col])
            processed_input[numeric_features] = feature_scaler.transform(processed_input[numeric_features])

            # Prediction
            decision = clf.predict(processed_input)[0]
            risk_prob = clf.predict_proba(processed_input)[0][1]

            # Results Rendering
            st.markdown('<p class="report-header">Review Summary</p>', unsafe_allow_html=True)
            
            res1, res2, res3 = st.columns(3)
            
            if decision == 0:
                res1.success("Result: APPROVED")
            else:
                res1.error("Result: DECLINED")

            res2.metric("Estimated Risk", f"{risk_prob:.2%}")
            res3.metric("Debt-to-Income", f"{loan_inc_ratio:.1%}")

            with st.expander("Reviewer Notes"):
                if risk_prob < 0.15:
                    st.info("Low risk profile. Suggest standard processing.")
                elif risk_prob < 0.45:
                    st.warning("Borderline profile. Recommend manual secondary review.")
                else:
                    st.error("High risk detected. Does not meet current lending criteria.")

        except Exception as e:
            st.error("Error: Could not process application. Verify input data.")