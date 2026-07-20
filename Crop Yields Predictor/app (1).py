import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f7f9f7;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(27, 67, 50, 0.25);
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin: 0;
    }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e8ede8;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7d6e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1b4332;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b4332;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #40916c;
        padding-left: 0.6rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #eef3ee;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        color: #2d6a4f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d6a4f !important;
        color: white !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(45, 106, 79, 0.35);
    }

    .prediction-box {
        background: linear-gradient(135deg, #f1f8f4, #e4f2e8);
        border: 1px solid #b7dfc4;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .prediction-box h2 {
        color: #1b4332;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .prediction-box p {
        color: #52796f;
        font-weight: 600;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("yield_df.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df = df.drop_duplicates()

    def is_str(obj):
        try:
            float(obj)
            return False
        except (ValueError, TypeError):
            return True

    to_drop = df[df["average_rain_fall_mm_per_year"].apply(is_str)].index
    df = df.drop(to_drop)
    df["average_rain_fall_mm_per_year"] = df["average_rain_fall_mm_per_year"].astype(float)

    cols = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes",
            "avg_temp", "Area", "Item", "hg/ha_yield"]
    df = df[cols]
    return df

df = load_data()

# ----------------------------------------------------------------------------
# MODEL TRAINING (cached so it only reruns when inputs change)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_models(data, test_size, random_state):
    X = data.drop("hg/ha_yield", axis=1)
    y = data["hg/ha_yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHotEncoder", ohe, ["Area", "Item"]),
            ("Standardization", scaler,
             ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]),
        ],
        remainder="passthrough",
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        results[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
        fitted[name] = model

    return preprocessor, results, fitted, (X_train.shape[0], X_test.shape[0])

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌾 Crop Yield Intelligence Platform</h1>
    <p>Explore global agricultural data, compare machine learning models, and predict crop yield (hg/ha) from climate & farming inputs.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR — GLOBAL CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05,
                           help="Fraction of data held out for testing")
    random_state = st.number_input("Random seed", value=42, step=1)
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.metric("Total records", f"{len(df):,}")
    st.metric("Countries", df["Area"].nunique())
    st.metric("Crops", df["Item"].nunique())
    st.markdown("---")
    st.caption("Data source: `yield_df.csv` — FAO / World Bank agricultural yield dataset.")

with st.spinner("Training models on the current data & settings..."):
    preprocessor, results, fitted_models, split_sizes = train_models(df, test_size, int(random_state))

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
best_model_name = results_df.index[0]

# ----------------------------------------------------------------------------
# TOP-LEVEL METRICS
# ----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Records</div>
    <div class="metric-value">{len(df):,}</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Yield (hg/ha)</div>
    <div class="metric-value">{df['hg/ha_yield'].mean():,.0f}</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Best Model</div>
    <div class="metric-value" style="font-size:1.25rem;">{best_model_name}</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Best R² Score</div>
    <div class="metric-value">{results_df.loc[best_model_name, 'R2']:.3f}</div></div>""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Explore Data", "🧮 Model Comparison", "🔮 Predict Yield", "🗂️ Raw Data"])

# ============================== TAB 1: EDA ==================================
with tab1:
    st.markdown('<div class="section-title">Filter the Dataset</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_items = st.multiselect("Crop(s)", sorted(df["Item"].unique()),
                                    default=sorted(df["Item"].unique())[:5])
    with fc2:
        sel_areas = st.multiselect("Country / Area", sorted(df["Area"].unique()),
                                    default=[])
    with fc3:
        yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
        sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    fdf = df.copy()
    if sel_items:
        fdf = fdf[fdf["Item"].isin(sel_items)]
    if sel_areas:
        fdf = fdf[fdf["Area"].isin(sel_areas)]
    fdf = fdf[(fdf["Year"] >= sel_years[0]) & (fdf["Year"] <= sel_years[1])]

    st.caption(f"Showing **{len(fdf):,}** records after filtering.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Yield Trend Over Time</div>', unsafe_allow_html=True)
        trend = fdf.groupby(["Year", "Item"])["hg/ha_yield"].mean().reset_index()
        fig = px.line(trend, x="Year", y="hg/ha_yield", color="Item", markers=True,
                      color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(plot_bgcolor="white", legend_title="Crop",
                           yaxis_title="Avg Yield (hg/ha)", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Total Yield by Crop</div>', unsafe_allow_html=True)
        crop_totals = fdf.groupby("Item")["hg/ha_yield"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(crop_totals, x="hg/ha_yield", y="Item", orientation="h",
                     color="hg/ha_yield", color_continuous_scale="Greens")
        fig.update_layout(plot_bgcolor="white", xaxis_title="Total Yield (hg/ha)",
                           yaxis_title="", height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Top Countries by Total Yield</div>', unsafe_allow_html=True)
        top_countries = (fdf.groupby("Area")["hg/ha_yield"].sum()
                          .sort_values(ascending=False).head(15).sort_values())
        fig = px.bar(top_countries, x=top_countries.values, y=top_countries.index, orientation="h",
                     color=top_countries.values, color_continuous_scale="Teal")
        fig.update_layout(plot_bgcolor="white", xaxis_title="Total Yield (hg/ha)",
                           yaxis_title="", height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Yield vs. Climate Factors</div>', unsafe_allow_html=True)
        axis_choice = st.selectbox("Compare yield against:",
                                    ["avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes"],
                                    format_func=lambda x: {
                                        "avg_temp": "Average Temperature (°C)",
                                        "average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                                        "pesticides_tonnes": "Pesticides (tonnes)"
                                    }[x])
        sample = fdf.sample(min(3000, len(fdf)), random_state=1) if len(fdf) > 0 else fdf
        fig = px.scatter(sample, x=axis_choice, y="hg/ha_yield", color="Item",
                          opacity=0.6, color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(plot_bgcolor="white", height=420)
        st.plotly_chart(fig, use_container_width=True)

# ======================= TAB 2: MODEL COMPARISON ============================
with tab2:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.caption(f"Trained on {split_sizes[0]:,} records, tested on {split_sizes[1]:,} records "
               f"({int(test_size*100)}% held out).")

    display_df = results_df.copy()
    display_df["R2"] = display_df["R2"].round(4)
    display_df["RMSE"] = display_df["RMSE"].round(1)
    display_df["MAE"] = display_df["MAE"].round(1)
    display_df["MSE"] = display_df["MSE"].round(1)
    display_df = display_df[["R2", "RMSE", "MAE", "MSE"]]

    cc1, cc2 = st.columns([1.1, 1])
    with cc1:
        st.dataframe(
            display_df.style.background_gradient(subset=["R2"], cmap="Greens")
                             .format({"R2": "{:.4f}", "RMSE": "{:,.1f}", "MAE": "{:,.1f}", "MSE": "{:,.1f}"}),
            use_container_width=True, height=260
        )
        st.success(f"🏆 **{best_model_name}** performs best with an R² of "
                   f"**{results_df.loc[best_model_name, 'R2']:.3f}**, meaning it explains that "
                   f"share of variance in crop yield.")

    with cc2:
        fig = px.bar(results_df.reset_index(), x="index", y="R2",
                     color="R2", color_continuous_scale="Greens",
                     labels={"index": "Model", "R2": "R² Score"})
        fig.update_layout(plot_bgcolor="white", height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Predicted vs. Actual (Best Model)</div>', unsafe_allow_html=True)
    X = df.drop("hg/ha_yield", axis=1)
    y = df["hg/ha_yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    X_test_t = preprocessor.transform(X_test)
    best_model = fitted_models[best_model_name]
    y_pred = best_model.predict(X_test_t)

    plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).sample(
        min(2000, len(y_test)), random_state=1)
    fig = px.scatter(plot_df, x="Actual", y="Predicted", opacity=0.5,
                      color_discrete_sequence=["#2d6a4f"])
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                              line=dict(color="firebrick", dash="dash"), name="Perfect Prediction"))
    fig.update_layout(plot_bgcolor="white", height=450,
                       xaxis_title="Actual Yield (hg/ha)", yaxis_title="Predicted Yield (hg/ha)")
    st.plotly_chart(fig, use_container_width=True)

# ========================== TAB 3: PREDICTION ===============================
with tab3:
    st.markdown('<div class="section-title">Predict Crop Yield</div>', unsafe_allow_html=True)
    st.caption("Enter growing conditions below to estimate yield using the trained models.")

    p1, p2 = st.columns(2)
    with p1:
        pred_area = st.selectbox("Country / Area", sorted(df["Area"].unique()))
        pred_item = st.selectbox("Crop", sorted(df["Item"].unique()))
        pred_year = st.number_input("Year", min_value=1960, max_value=2050,
                                     value=int(df["Year"].max()))
    with p2:
        pred_rain = st.number_input("Average rainfall (mm/year)", min_value=0.0,
                                     value=float(df["average_rain_fall_mm_per_year"].median()))
        pred_pest = st.number_input("Pesticides used (tonnes)", min_value=0.0,
                                     value=float(df["pesticides_tonnes"].median()))
        pred_temp = st.number_input("Average temperature (°C)",
                                     value=float(df["avg_temp"].median()))

    model_choice = st.selectbox("Model to use for prediction", results_df.index.tolist(),
                                 index=0)

    if st.button("🔍 Predict Yield", use_container_width=True):
        input_df = pd.DataFrame([{
            "Year": pred_year,
            "average_rain_fall_mm_per_year": pred_rain,
            "pesticides_tonnes": pred_pest,
            "avg_temp": pred_temp,
            "Area": pred_area,
            "Item": pred_item,
        }])
        input_t = preprocessor.transform(input_df)
        model = fitted_models[model_choice]
        prediction = model.predict(input_t)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <p>Estimated Yield ({model_choice})</p>
            <h2>{prediction:,.0f} hg/ha</h2>
            <p>≈ {prediction/10000:,.2f} tonnes per hectare</p>
        </div>
        """, unsafe_allow_html=True)

        hist = df[(df["Area"] == pred_area) & (df["Item"] == pred_item)]
        if len(hist) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Historical Context</div>', unsafe_allow_html=True)
            fig = px.line(hist.sort_values("Year"), x="Year", y="hg/ha_yield", markers=True,
                          title=f"Historical yield: {pred_item} in {pred_area}")
            fig.add_hline(y=prediction, line_dash="dash", line_color="firebrick",
                          annotation_text="Prediction")
            fig.update_layout(plot_bgcolor="white", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available for this exact Area/Crop combination for comparison.")

# =========================== TAB 4: RAW DATA ================================
with tab4:
    st.markdown('<div class="section-title">Cleaned Dataset</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        raw_area = st.multiselect("Filter by Area", sorted(df["Area"].unique()), key="raw_area")
    with r2:
        raw_item = st.multiselect("Filter by Crop", sorted(df["Item"].unique()), key="raw_item")

    rdf = df.copy()
    if raw_area:
        rdf = rdf[rdf["Area"].isin(raw_area)]
    if raw_item:
        rdf = rdf[rdf["Item"].isin(raw_item)]

    st.dataframe(rdf, use_container_width=True, height=450)
    st.download_button("⬇️ Download filtered data as CSV",
                        rdf.to_csv(index=False).encode("utf-8"),
                        "crop_yield_filtered.csv", "text/csv", use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · Plotly — Crop Yield Intelligence Platform")
