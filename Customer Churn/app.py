import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Overall page */
        .main {
            padding-top: 1.5rem;
        }

        /* Hero header */
        .hero {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            padding: 2rem 2rem 1.75rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.75rem;
            color: white;
            text-align: center;
        }
        .hero h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: white;
        }
        .hero p {
            font-size: 1rem;
            opacity: 0.92;
            margin: 0;
        }

        /* Section labels */
        .section-label {
            font-size: 0.95rem;
            font-weight: 600;
            color: #4F46E5;
            margin: 0.25rem 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        /* Result cards */
        .result-card {
            padding: 1.75rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1rem;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .result-card.churn {
            background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
            border-left: 6px solid #EF4444;
        }
        .result-card.stay {
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            border-left: 6px solid #22C55E;
        }
        .result-card h2 {
            margin: 0 0 0.25rem 0;
            font-size: 1.6rem;
        }
        .result-card.churn h2 { color: #B91C1C; }
        .result-card.stay h2 { color: #15803D; }
        .result-card p {
            margin: 0;
            font-size: 0.95rem;
            color: #444;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 0.6rem 0;
            font-weight: 600;
            font-size: 1rem;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            transition: transform 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            opacity: 0.95;
            color: white;
        }

        /* Footer */
        .footer-note {
            text-align: center;
            color: #888;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

try:
    scaler, model = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `scaler.pkl` and `model.pkl` are in the "
        "same folder as this app."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this app")
    st.write(
        "This tool estimates the likelihood that a customer will **churn** "
        "(cancel their service) based on a few key account details."
    )
    st.markdown("---")
    st.markdown("**Model:** Support Vector Machine (linear kernel)")
    st.markdown("**Test accuracy:** ~88%")
    st.markdown("---")
    st.caption(
        "Predictions are estimates based on historical patterns and should be "
        "used as one input among several when making retention decisions."
    )

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Predictor</h1>
        <p>Enter a customer's details below to estimate their churn risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Customer Details</div>', unsafe_allow_html=True)

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        tenure = st.number_input(
            "Tenure (months with company)", min_value=0, max_value=130, value=10, step=1
        )

    with col2:
        monthly_charge = st.number_input(
            "Monthly Charges ($)", min_value=30.0, max_value=150.0, value=60.0, step=1.0
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.write("")
    submitted = st.form_submit_button("🔮 Predict Churn")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    gender_encoded = 1 if gender == "Female" else 0
    X = np.array([[age, gender_encoded, tenure, monthly_charge]])
    X_scaled = scaler.transform(X)

    try:
        proba = model.predict_proba(X_scaled)[0]
        churn_prob = proba[1]
        prediction = 1 if churn_prob >= 0.5 else 0
        confidence_text = f"Churn probability: {churn_prob * 100:.1f}%"
    except AttributeError:
        prediction = model.predict(X_scaled)[0]
    try:
        score = model.decision_function(X_scaled)[0]
        confidence_text = f"Decision score: {score:.2f}"
    except AttributeError:
        confidence_text = ""
    except AttributeError:
            pass

    if prediction == 0:
        st.markdown(
            f"""
            <div class="result-card churn">
                <h2>⚠️ Likely to Churn</h2>
                <p>This customer shows patterns associated with cancelling their service.</p>
                <p>{confidence_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card stay">
                <h2>✅ Likely to Stay</h2>
                <p>This customer shows patterns associated with remaining subscribed.</p>
                <p>{confidence_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View input summary"):
        st.write(
            {
                "Age": age,
                "Gender": gender,
                "Tenure (months)": tenure,
                "Monthly Charges ($)": monthly_charge,
            }
        )
else:
    st.info("👆 Fill in the details above and click **Predict Churn** to see the result.")

st.markdown(
    '<div class="footer-note">Built with Streamlit • Model trained with scikit-learn</div>',
    unsafe_allow_html=True,
)
