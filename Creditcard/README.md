# Credit Card Fraud Detection — Streamlit App

A front end for the Logistic Regression fraud-detection model from the notebook.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

1. In the sidebar, upload the Kaggle `creditcard.csv` dataset (columns: `Time, V1-V28, Amount, Class`).
   - No dataset handy? Check "Use a small synthetic demo dataset instead" to try the app.
2. **Predict a Transaction** tab — manually enter `Time`, `Amount`, and the 28 `V` features (or click
   "Fill with random row from dataset") and click Predict.
3. **Batch Prediction** tab — upload a CSV of transactions (same columns, `Class` optional) to score
   many rows at once and download the results.
4. **Dataset & Model Insights** tab — class balance, amount statistics, training/test accuracy,
   confusion matrix, and classification report.

## Notes

- The model is retrained each time a new dataset is uploaded (cached via `st.cache_resource`), using
  the same upsampling-to-balance approach as the notebook (fraud class resampled with replacement to
  match the legit sample size, then an 80/20 train/test split).
