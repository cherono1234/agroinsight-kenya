"""
AgroInsight Kenya - Dashboard
--------------------------------
Interactive web dashboard for crop yield
prediction built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys
import os
import streamlit.components.v1 as components

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)
from startup import ensure_model_exists
ensure_model_exists()


from prediction_engine import PredictionEngine

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AgroInsight Kenya",
    page_icon  = "🌱",
    layout     = "wide",
)

# ── Load model once and cache it ──────────────────────────────────────
@st.cache_resource
def load_engine():
    return PredictionEngine()

engine = load_engine()

# ── Load clean data for charts ────────────────────────────────────────
@st.cache_data
def load_clean_data():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean', 'master_clean.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_clean_data()

# ── Colour helpers ─────────────────────────────────────────────────────
RATING_COLORS = {
    "Excellent": "#1A6B3A",
    "Good"     : "#4CAF50",
    "Fair"     : "#FF9800",
    "Poor"     : "#F44336",
}

CONF_COLORS = {
    "High"  : "#1A6B3A",
    "Medium": "#FF9800",
    "Low"   : "#F44336",
}

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Kenya_coat_of_arms.svg", width=80)
    st.title("🌱 AgroInsight Kenya")
    st.caption("Data-Driven Crop Yield Prediction")
    st.divider()

    options = PredictionEngine.get_options()

    st.subheader("📍 Select Parameters")

    county  = st.selectbox("County",  options['counties'])
    crop    = st.selectbox("Crop",    options['crops'])
    season  = st.selectbox("Season",  options['seasons'])
    year    = st.selectbox("Year",    options['years'], index=5)

    st.divider()
    st.subheader("🌦️ Weather & Soil Inputs")

    rainfall    = st.slider("Avg Rainfall (mm)",    100, 1500, 700, step=10)
    temperature = st.slider("Avg Temperature (°C)",  10,   35,  22, step=1)
    rain_dev    = st.slider("Rainfall Deviation (mm)", -200, 200, 0, step=10)
    ph_level    = st.slider("Soil pH Level",          4.0,  8.0, 6.2, step=0.1)
    fertility   = st.slider("Soil Fertility Index",   0.1,  1.0, 0.7, step=0.05)
    area        = st.slider("Area Planted (ha)",      100, 5000, 1000, step=100)

    st.divider()
    predict_btn = st.button("🔍 Predict Yield", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════
st.title("🌱 AgroInsight Kenya")
st.caption("A Machine Learning System for Crop Yield Prediction | Kenya 🌱")
# ── Tabs ───────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home","🔮 Prediction","📊 Data Explorer","📈 Trends","ℹ️ About"])
# ══════════════════════════════════════════════════════════════════════
with tab0:
    components.html(open(os.path.join(BASE_DIR, "website", "index.html")).read(), height=4000, scrolling=True)
with tab0:
    st.markdown("""
    <div style="text-align:center; padding:40px 0">
        <h1 style="font-size:48px; color:#1A6B3A">🌱 AgroInsight Kenya</h1>
        <p style="font-size:20px; color:#555; margin:16px 0 32px">
            Machine Learning Crop Yield Prediction for Kenyan Farmers
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Counties Covered", "10")
    c2.metric("Crops Supported", "5")
    c3.metric("Years of Data", "10")

    st.divider()
    st.subheader("What is AgroInsight Kenya?")
    st.write("""
    AgroInsight Kenya is a free, data-driven crop yield prediction system 
    built specifically for Kenyan smallholder farmers and agricultural 
    extension officers. Using machine learning trained on real Kenyan 
    agricultural, weather, and soil data, it predicts how much your crop 
    will yield before you plant — helping you make smarter decisions.
    """)
    st.divider()
    st.subheader("How to use it")
    st.write("1. Click the **Prediction** tab above")
    st.write("2. Select your county, crop, and season from the sidebar")
    st.write("3. Adjust the weather and soil sliders")
    st.write("4. Click **Predict Yield** to get your forecast")
    st.divider()
    st.info("👈 Click **Prediction** tab to get started!")
    # ══════════════════════════════════════════════════════════════════════

#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        with st.spinner("Running prediction model..."):
            result = engine.predict(
                county             = county,
                crop               = crop,
                season             = season,
                year               = year,
                rainfall           = rainfall,
                temperature        = temperature,
                rainfall_deviation = rain_dev,
                ph_level           = ph_level,
                fertility          = fertility,
                area               = area,
            )

        if 'error' in result:
            st.error(f"❌ {result['error']}")
        else:
            # ── Top metrics ───────────────────────────────────────────
            st.subheader(f"Results for {crop} in {county} — {season} {year}")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                label = "Predicted Yield",
                value = f"{result['predicted_yield']:,} kg/ha",
            )
            col2.metric(
                label = "Performance Rating",
                value = result['rating'],
            )
            col3.metric(
                label = "Confidence Level",
                value = result['confidence'],
            )
            col4.metric(
                label = "Model R² Score",
                value = f"{result['model_r2']}",
            )

            st.divider()

            # ── Recommendation box ────────────────────────────────────
            rating_color = RATING_COLORS.get(result['rating'], "#555")
            st.markdown(f"""
            <div style="background:{rating_color}18;border-left:4px solid {rating_color};
                        padding:16px;border-radius:8px;margin-bottom:16px;">
                <b style="color:{rating_color}">💡 Recommendation</b><br>
                <span>{result['recommendation']}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Yield gauge chart ─────────────────────────────────────
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Yield Gauge")
                fig, ax = plt.subplots(figsize=(5, 3))
                categories = ['Poor\n(<1500)', 'Fair\n(1500–2500)',
                              'Good\n(2500–3500)', 'Excellent\n(>3500)']
                values     = [1500, 1000, 1000, 1000]
                colors     = ['#F44336', '#FF9800', '#4CAF50', '#1A6B3A']
                bars = ax.barh(categories, values, color=colors, alpha=0.6, height=0.5)

                # Mark predicted yield
                pred = result['predicted_yield']
                ax.axvline(x=min(pred, 5000), color='navy', linewidth=2.5,
                           linestyle='--', label=f"Predicted: {pred:,} kg/ha")
                ax.set_xlabel("Yield (kg/ha)")
                ax.set_xlim(0, 5000)
                ax.legend(fontsize=9)
                ax.set_facecolor('#FAFAFA')
                fig.patch.set_facecolor('white')
                st.pyplot(fig)
                plt.close()

            with col_b:
                st.subheader("Input Summary")
                summary_data = {
                    "Parameter"  : ["County","Crop","Season","Year","Rainfall","Temperature","Soil pH","Fertility","Area"],
                    "Value"      : [county, crop, season, str(year),
                                    f"{rainfall} mm", f"{temperature} °C",
                                    str(ph_level), str(fertility),
                                    f"{area:,} ha"],
                }
                st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    else:
        # Default state before prediction
        st.info("👈 Set your parameters in the sidebar and click **Predict Yield** to get started.")

        if df is not None:
            st.subheader("📌 Quick Stats from Training Data")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records",  f"{len(df):,}")
            c2.metric("Counties",       df['county'].nunique())
            c3.metric("Crops",          df['crop'].nunique())
            c4.metric("Avg Yield",      f"{df['yield_kg_per_ha'].mean():.0f} kg/ha")

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Explorer")

    if df is None:
        st.warning("Clean dataset not found. Run the training pipeline first.")
    else:
        col1, col2 = st.columns(2)
        filter_county = col1.multiselect("Filter by County", sorted(df['county'].unique()), default=sorted(df['county'].unique())[:3])
        filter_crop   = col2.multiselect("Filter by Crop",   sorted(df['crop'].unique()),   default=sorted(df['crop'].unique())[:2])

        filtered = df[
            (df['county'].isin(filter_county)) &
            (df['crop'].isin(filter_crop))
        ]

        st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
        st.dataframe(
            filtered[['county','crop','season','year','yield_kg_per_ha',
                       'avg_rainfall_mm','avg_temp_celsius','ph_level','fertility_index']]
            .sort_values('yield_kg_per_ha', ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        st.divider()
        st.subheader("Yield Distribution by Crop")
        fig, ax = plt.subplots(figsize=(10, 4))
        crops_available = sorted(filtered['crop'].unique())
        data_to_plot    = [filtered[filtered['crop'] == c]['yield_kg_per_ha'].values for c in crops_available]
        bp = ax.boxplot(data_to_plot, labels=crops_available, patch_artist=True)
        palette = ['#1A6B3A','#4CAF50','#81C784','#A5D6A7','#C8E6C9']
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Yield (kg/ha)")
        ax.set_xlabel("Crop")
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — TRENDS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Yield Trends Over Time")

    if df is None:
        st.warning("Clean dataset not found. Run the training pipeline first.")
    else:
        sel_crop   = st.selectbox("Select Crop for Trend", sorted(df['crop'].unique()), key="trend_crop")
        sel_county = st.multiselect("Select Counties", sorted(df['county'].unique()),
                                     default=sorted(df['county'].unique())[:4], key="trend_county")

        trend_df = df[(df['crop'] == sel_crop) & (df['county'].isin(sel_county))]
        trend_grouped = (trend_df.groupby(['year','county'])['yield_kg_per_ha']
                         .mean().reset_index())

        fig, ax = plt.subplots(figsize=(11, 5))
        colors  = plt.cm.Set2(np.linspace(0, 1, len(sel_county)))
        for county_name, color in zip(sel_county, colors):
            county_data = trend_grouped[trend_grouped['county'] == county_name]
            ax.plot(county_data['year'], county_data['yield_kg_per_ha'],
                    marker='o', linewidth=2, label=county_name, color=color)

        ax.set_xlabel("Year")
        ax.set_ylabel("Average Yield (kg/ha)")
        ax.set_title(f"{sel_crop} Yield Trends by County")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Average Yield by County (All Years)")
        county_avg = (df[df['crop'] == sel_crop]
                      .groupby('county')['yield_kg_per_ha']
                      .mean()
                      .sort_values(ascending=True))

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.barh(county_avg.index, county_avg.values,
                        color='#1A6B3A', alpha=0.75)
        ax2.bar_label(bars, fmt='%.0f kg/ha', padding=4, fontsize=9)
        ax2.set_xlabel("Average Yield (kg/ha)")
        ax2.set_title(f"Average {sel_crop} Yield by County")
        ax2.set_facecolor('#FAFAFA')
        fig2.patch.set_facecolor('white')
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("ℹ️ About AgroInsight Kenya")
    st.markdown("""
    **AgroInsight Kenya** is an end-year Computer Science project that applies
    machine learning to predict crop yields for Kenyan smallholder farmers.

    ---

    ### 🎯 Objectives
    - Predict county-level crop yields using historical data
    - Help farmers and extension officers make data-driven planting decisions
    - Contribute to Kenya's food security goals (Big Four Agenda, SDG 2)

    ### 🛠️ Technologies Used
    | Tool | Purpose |
    |---|---|
    | Python 3.13 | Core programming language |
    | Pandas & NumPy | Data processing |
    | Scikit-learn | Model training & evaluation |
    | XGBoost | Primary prediction algorithm |
    | Matplotlib & Seaborn | Data visualisation |
    | Streamlit | This dashboard |

    ### 📊 Model Performance
    """)

    if engine.model_name:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", engine.model_name)
        col2.metric("R² Score",   engine.model_r2)
        col3.metric("RMSE",       f"{engine.model_rmse} kg/ha")

    st.markdown("""
    ### 📁 Data Sources
    - Kenya National Bureau of Statistics (KNBS)
    - Kenya Meteorological Department
    - FAO GAEZ Portal
    - Ministry of Agriculture, Livestock and Fisheries

    ---
     | AgroInsight Kenya | 2026*
    """)