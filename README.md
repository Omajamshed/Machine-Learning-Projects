# 🌾 Crop Yield Intelligence Platform

An interactive Streamlit app built from your `Crop_Yeilds.ipynb` notebook — turns the notebook's
data cleaning, EDA, and model-training steps into a full, polished dashboard.

## What's inside
- **Explore Data** — filter by crop, country, and year; view yield trends, top crops/countries,
  and yield vs. climate scatter plots (all interactive via Plotly).
- **Model Comparison** — trains Linear Regression, Lasso, Ridge, KNN, Decision Tree, and Random
  Forest live, then ranks them by R² / RMSE / MAE, plus an actual-vs-predicted chart for the best model.
- **Predict Yield** — pick a country, crop, year, rainfall, pesticide use, and temperature to get
  a live yield prediction from any trained model, with historical context for that country/crop.
- **Raw Data** — browse and download the cleaned dataset.

## Run it locally

1. Make sure `yield_df.csv` is in the same folder as `app.py`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```
4. It will open automatically at `http://localhost:8501`.

## Notes
- Models are trained live on startup (per your preference) and cached, so changing the sidebar's
  test-size or random-seed will retrain automatically; otherwise cached results are reused instantly.
- The data-cleaning logic (dropping the index column, removing duplicates, filtering malformed
  rainfall rows) mirrors exactly what was done in the original notebook.
