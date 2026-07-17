import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Symptom Assistant",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0f1b2d 0%, #1a2e4a 100%); }

    /* Hide default header */
    header { visibility: hidden; }

    /* Title block */
    .hero {
        text-align: center;
        padding: 2rem 0 1.5rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #8faec7;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Card wrapper */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(6px);
    }
    .card h3 {
        color: #60b4ff;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1.2rem;
    }

    /* Labels */
    .stRadio > label, .stSelectbox > label, .stSlider > label, .stNumberInput > label {
        color: #c8dff0 !important;
        font-weight: 500;
    }

    /* Radio options */
    .stRadio div[role="radiogroup"] label { color: #a0c4e0 !important; }

    /* Analyse button */
    .stButton > button {
        width: 100%;
        padding: 0.85rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #1e90ff, #00c6ff);
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        border: none;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Result banners */
    .result-positive {
        background: linear-gradient(135deg, #1a3a5c, #0d2741);
        border-left: 5px solid #00c6ff;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        color: #dff0ff;
    }
    .result-negative {
        background: linear-gradient(135deg, #0d3320, #0a2318);
        border-left: 5px solid #00e68a;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        color: #d0ffe8;
    }
    .result-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; }
    .result-sub   { font-size: 0.92rem; opacity: 0.8; }

    /* Probability badge */
    .prob-badge {
        display: inline-block;
        background: rgba(0,198,255,0.15);
        border: 1px solid #00c6ff55;
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        color: #00c6ff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,200,0,0.07);
        border: 1px solid rgba(255,200,0,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #ffd060;
        font-size: 0.82rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Synthetic training data ───────────────────────────────────────────────────
@st.cache_resource
def build_model():
    """
    Build a RandomForestClassifier trained on a synthetic dataset that mirrors
    the structure of Disease_symptom_and_patient_profile_dataset.csv.
    Columns: Fever, Cough, Fatigue, Difficulty Breathing, Age,
             Cholesterol Level, Gender, Blood Pressure, Outcome Variable
    """
    np.random.seed(42)
    n = 2000

    fever        = np.random.choice(['yes', 'no'], n, p=[0.6, 0.4])
    cough        = np.random.choice(['yes', 'no'], n, p=[0.55, 0.45])
    fatigue      = np.random.choice(['yes', 'no'], n, p=[0.5, 0.5])
    diff_breath  = np.random.choice(['yes', 'no'], n, p=[0.35, 0.65])
    age          = np.random.randint(1, 90, n)
    cholesterol  = np.random.randint(100, 300, n)
    gender       = np.random.choice(['Male', 'Female'], n)
    blood_press  = np.random.choice(['Low', 'Normal', 'High'], n)

    # Outcome: positive if ≥2 symptoms + (age>50 or high cholesterol)
    symptom_score = (
        (fever == 'yes').astype(int) +
        (cough == 'yes').astype(int) +
        (fatigue == 'yes').astype(int) +
        (diff_breath == 'yes').astype(int)
    )
    risk = ((symptom_score >= 2) & ((age > 50) | (cholesterol > 200))).astype(int)
    # add noise
    flip = np.random.choice([0, 1], n, p=[0.85, 0.15])
    outcome = np.where(flip, 1 - risk, risk)

    df = pd.DataFrame({
        'Fever': fever,
        'Cough': cough,
        'Fatigue': fatigue,
        'Difficulty Breathing': diff_breath,
        'Age': age,
        'Cholesterol Level': cholesterol,
        'Gender': gender,
        'Blood Pressure': blood_press,
        'Outcome Variable': outcome
    })

    X = df.drop('Outcome Variable', axis=1)
    y = df['Outcome Variable']
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    return clf, X.columns.tolist()


model, feature_cols = build_model()


def predict(fever, cough, fatigue, diff_breath, age, cholesterol, gender, blood_pressure):
    row = {
        'Age': age,
        'Cholesterol Level': cholesterol,
        'Fever_yes': 1 if fever == 'Yes' else 0,
        'Cough_yes': 1 if cough == 'Yes' else 0,
        'Fatigue_yes': 1 if fatigue == 'Yes' else 0,
        'Difficulty Breathing_yes': 1 if diff_breath == 'Yes' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Blood Pressure_Low': 1 if blood_pressure == 'Low' else 0,
        'Blood Pressure_Normal': 1 if blood_pressure == 'Normal' else 0,
    }
    df_row = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_cols]

    pred  = model.predict(df_row)[0]
    proba = model.predict_proba(df_row)[0][1]
    return pred, proba


# ── Disease mapping ───────────────────────────────────────────────────────────
DISEASE_MAP = {
    # (fever, cough, fatigue, diff_breath) → likely condition
    (True,  True,  True,  True):  ("Severe Respiratory Infection / Pneumonia",    "🫁"),
    (True,  True,  True,  False): ("Influenza (Flu)",                              "🤧"),
    (True,  True,  False, True):  ("Bronchitis / COVID-19",                        "😷"),
    (True,  False, True,  True):  ("Anemia / Heart Failure",                       "❤️"),
    (True,  False, True,  False): ("Viral Fever / Dengue",                         "🦠"),
    (True,  False, False, True):  ("Pulmonary Embolism (seek urgent care)",        "⚠️"),
    (False, True,  True,  True):  ("Chronic Obstructive Pulmonary Disease (COPD)", "🫀"),
    (False, True,  True,  False): ("Common Cold / Mild Infection",                 "🤒"),
    (False, True,  False, True):  ("Asthma / Allergic Reaction",                   "💨"),
    (False, False, True,  True):  ("Cardiovascular Issue / Thyroid Disorder",      "💓"),
    (False, False, True,  False): ("Fatigue Syndrome / Anemia",                    "😴"),
    (False, False, False, True):  ("Anxiety / Panic Disorder",                     "🧠"),
    (False, True,  False, False): ("Upper Respiratory Infection",                  "😮‍💨"),
    (True,  False, False, False): ("Bacterial Infection / UTI",                    "🌡️"),
    (False, False, False, False): ("Low Risk — No major symptoms detected",        "✅"),
}

def get_disease(fever, cough, fatigue, diff_breath):
    key = (fever == 'Yes', cough == 'Yes', fatigue == 'Yes', diff_breath == 'Yes')
    return DISEASE_MAP.get(key, ("Unknown — please consult a doctor", "🏥"))


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 AI Symptom Assistant</h1>
    <p>Enter your symptoms and health profile to receive an AI-powered medical analysis.</p>
</div>
""", unsafe_allow_html=True)

# — Symptoms card —
st.markdown('<div class="card"><h3>🔍 Symptoms</h3>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    fever       = st.radio("Fever",               ["No", "Yes"], horizontal=True)
    fatigue     = st.radio("Fatigue",             ["No", "Yes"], horizontal=True)
with col2:
    cough       = st.radio("Cough",               ["No", "Yes"], horizontal=True)
    diff_breath = st.radio("Difficulty Breathing",["No", "Yes"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# — Patient profile card —
st.markdown('<div class="card"><h3>👤 Patient Profile</h3>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    age           = st.slider("Age", 1, 100, 30)
    gender        = st.selectbox("Gender", ["Male", "Female"])
with col4:
    cholesterol   = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=180, step=5)
    blood_pressure = st.selectbox("Blood Pressure", ["Normal", "Low", "High"])
st.markdown('</div>', unsafe_allow_html=True)

# — Analyse button —
if st.button("🔬 Analyse My Symptoms"):
    with st.spinner("Analysing your symptoms…"):
        pred, proba = predict(fever, cough, fatigue, diff_breath, age, cholesterol, gender, blood_pressure)
        disease, icon = get_disease(fever, cough, fatigue, diff_breath)

    st.markdown("---")
    if pred == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title">{icon} Possible Condition Detected</div>
            <div style="font-size:1.15rem; font-weight:600; margin:0.4rem 0;">{disease}</div>
            <div class="result-sub">Based on your symptoms and health profile, the model has flagged a potential medical concern. Please consult a healthcare professional for a proper diagnosis.</div>
            <div class="prob-badge">Confidence: {proba*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title">✅ Low Risk — {disease.split('—')[-1].strip() if '—' in disease else disease}</div>
            <div class="result-sub">Your symptom profile suggests a low risk of serious illness. Continue monitoring and maintain a healthy lifestyle. Seek medical attention if symptoms worsen.</div>
            <div class="prob-badge" style="color:#00e68a; border-color:#00e68a55; background:rgba(0,230,138,0.1);">Risk Score: {proba*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Symptom summary
    st.markdown("<br>", unsafe_allow_html=True)
    active = [s for s, v in [("Fever", fever), ("Cough", cough), ("Fatigue", fatigue), ("Difficulty Breathing", diff_breath)] if v == "Yes"]
    if active:
        st.info(f"**Reported symptoms:** {', '.join(active)}  |  Age: {age}  |  Cholesterol: {cholesterol} mg/dL  |  BP: {blood_pressure}")

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and does not constitute medical advice.
        Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><p style='text-align:center;color:#4a6a88;font-size:0.8rem;'>Built with ❤️ · SMIT Final Project · Powered by RandomForestClassifier</p>", unsafe_allow_html=True)
