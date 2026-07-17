import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Only fill if there actually are missing values, and do it per-column cheaply
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, random_state: int = 2):
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]

    n = min(len(legit), 492) if len(fraud) == 0 else len(fraud)
    legit_sample = legit.sample(n=min(len(legit), max(n, 1)), random_state=42)

    if len(fraud) == 0:
        raise ValueError("Dataset has no fraud (Class=1) examples to train on.")

    fraud_upsampled = resample(
        fraud, replace=True, n_samples=len(legit_sample), random_state=42
    )
    balanced = pd.concat([legit_sample, fraud_upsampled])

    X = balanced.drop(columns="Class", axis=1)
    y = balanced["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    report = classification_report(y_test, test_pred, target_names=["Legit", "Fraud"])

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cm": cm,
        "report": report,
        "feature_names": list(X.columns),
    }


def predict_transaction(model, input_values, feature_names):
    row = pd.DataFrame([input_values], columns=feature_names)
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    return pred, proba


# ---------------------------------------------------------------------------
# Sidebar — data loading
# ---------------------------------------------------------------------------

st.sidebar.title("💳 Fraud Detector Setup")
st.sidebar.markdown(
    "Upload the Kaggle **creditcard.csv** dataset "
    "(`Time, V1-V28, Amount, Class` columns) to train the model."
)
uploaded = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])

use_demo = st.sidebar.checkbox(
    "No file? Use a small synthetic demo dataset instead", value=False
)

df = None
if uploaded is not None:
    with st.spinner("Loading dataset..."):
        df = load_data(uploaded)
elif use_demo:
    with st.spinner("Generating synthetic demo data..."):
        rng = np.random.default_rng(42)
        n_legit, n_fraud = 2000, 60
        legit_rows = rng.normal(0, 1, size=(n_legit, 28))
        fraud_rows = rng.normal(2.5, 1.5, size=(n_fraud, 28))
        data = np.vstack([legit_rows, fraud_rows])
        amounts = np.concatenate(
            [rng.exponential(50, n_legit), rng.exponential(300, n_fraud)]
        )
        times = rng.integers(0, 172792, n_legit + n_fraud)
        classes = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
        demo = pd.DataFrame(data, columns=[f"V{i}" for i in range(1, 29)])
        demo.insert(0, "Time", times)
        demo["Amount"] = amounts
        demo["Class"] = classes.astype(int)
        df = demo
        st.sidebar.info("Using synthetic demo data — not real transactions.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("💳 Credit Card Fraud Detection")
st.caption("Logistic Regression model trained on a class-balanced sample, as in the notebook.")

if df is None:
    st.info(
        "👈 Upload the `creditcard.csv` dataset in the sidebar (or check the demo-data "
        "box) to train the model and start predicting."
    )
    st.stop()

missing = [c for c in FEATURE_COLS + ["Class"] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing expected columns: {missing}")
    st.stop()

with st.spinner("Training model..."):
    result = train_model(df)

model = result["model"]
feature_names = result["feature_names"]

tab_predict, tab_batch, tab_insights = st.tabs(
    ["🔍 Predict a Transaction", "📄 Batch Prediction (CSV)", "📊 Dataset & Model Insights"]
)

# ---------------------------------------------------------------------------
# Tab 1: single transaction prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Check a single transaction")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("Fill in the transaction fields, or grab a random real row from the test data.")
    with col_b:
        if st.button("🎲 Fill with random row from dataset"):
            sample_row = df.sample(1).iloc[0]
            for col in feature_names:
                st.session_state[f"in_{col}"] = float(sample_row[col])

    with st.expander("Time & Amount", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (seconds since first transaction)", value=st.session_state.get("in_Time", 0.0), key="in_Time")
        with c2:
            amount_val = st.number_input("Amount", value=st.session_state.get("in_Amount", 0.0), key="in_Amount")

    with st.expander("PCA Features (V1 - V28)"):
        v_values = {}
        cols = st.columns(4)
        for i in range(1, 29):
            col_name = f"V{i}"
            with cols[(i - 1) % 4]:
                v_values[col_name] = st.number_input(
                    col_name, value=st.session_state.get(f"in_{col_name}", 0.0),
                    key=f"in_{col_name}", format="%.4f"
                )

    if st.button("Predict", type="primary"):
        input_values = [time_val] + [v_values[f"V{i}"] for i in range(1, 29)] + [amount_val]
        pred, proba = predict_transaction(model, input_values, feature_names)

        if pred == 0:
            st.success(f"✅ Legit Transaction  (fraud probability: {proba[1]:.2%})")
        else:
            st.error(f"🚨 Fraud Transaction  (fraud probability: {proba[1]:.2%})")

# ---------------------------------------------------------------------------
# Tab 2: batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a batch of transactions")
    st.markdown(
        "Upload a CSV with the same columns as the training data "
        "(`Time, V1-V28, Amount` — a `Class` column is optional)."
    )
    batch_file = st.file_uploader("Upload transactions CSV", type=["csv"], key="batch_upload")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing_batch = [c for c in feature_names if c not in batch_df.columns]
        if missing_batch:
            st.error(f"Missing columns: {missing_batch}")
        else:
            preds = model.predict(batch_df[feature_names])
            probas = model.predict_proba(batch_df[feature_names])[:, 1]
            out = batch_df.copy()
            out["Prediction"] = np.where(preds == 1, "Fraud", "Legit")
            out["Fraud Probability"] = probas
            st.dataframe(out, use_container_width=True)

            n_fraud = int((preds == 1).sum())
            st.metric("Flagged as Fraud", f"{n_fraud} / {len(out)}")

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV", csv_bytes,
                file_name="fraud_predictions.csv", mime="text/csv"
            )

# ---------------------------------------------------------------------------
# Tab 3: insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", f"{len(df):,}")
    c2.metric("Legit", f"{(df.Class == 0).sum():,}")
    c3.metric("Fraud", f"{(df.Class == 1).sum():,}")

    st.bar_chart(df["Class"].value_counts().rename({0: "Legit", 1: "Fraud"}))

    st.markdown("**Amount statistics by class**")
    st.dataframe(df.groupby("Class")["Amount"].describe(), use_container_width=True)

    st.subheader("Model performance (on balanced held-out test set)")
    m1, m2 = st.columns(2)
    m1.metric("Training accuracy", f"{result['train_acc']:.2%}")
    m2.metric("Test accuracy", f"{result['test_acc']:.2%}")

    st.markdown("**Confusion matrix** (rows = actual, cols = predicted)")
    cm_df = pd.DataFrame(result["cm"], index=["Actual Legit", "Actual Fraud"], columns=["Pred Legit", "Pred Fraud"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Classification report**")
    st.code(result["report"])
