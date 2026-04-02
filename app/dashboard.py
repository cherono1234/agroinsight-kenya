import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from startup import ensure_model_exists
ensure_model_exists()

from prediction_engine import PredictionEngine
from weather_api import get_weather, get_counties
from crop_advisory import get_advisory

st.set_page_config(page_title="AgroInsight Kenya", page_icon="plant", layout="wide")

@st.cache_resource
def load_engine():
    return PredictionEngine()
engine = load_engine()

@st.cache_data
def load_clean_data():
    path = os.path.join(BASE_DIR, "data", "clean", "master_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
df = load_clean_data()

RATING_COLORS = {"Excellent":"#0B3E85","Good":"#4CAF50","Fair":"#FF9800","Poor":"#F44336"}

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AgroInsight Kenya")
    st.caption("Data-Driven Crop Yield Prediction")
    st.divider()
    st.subheader("Select Parameters")

    all_counties = get_counties()
    county  = st.selectbox("County",  all_counties)
    crop    = st.selectbox("Crop",    PredictionEngine.get_options()["crops"])
    season  = st.selectbox("Season",  PredictionEngine.get_options()["seasons"])
    year    = st.selectbox("Year",    PredictionEngine.get_options()["years"], index=5)

    st.divider()
    st.subheader("Weather & Soil Inputs")

    if st.button("Auto-fetch live weather", use_container_width=True):
        with st.spinner(f"Fetching weather for {county}..."):
            wx = get_weather(county)
            if wx:
                st.session_state["rainfall"]    = float(wx["avg_rainfall_mm"])
                st.session_state["temperature"] = float(wx["avg_temp_celsius"])
                st.success(f"Weather loaded for {county}")
            else:
                st.warning("Could not fetch weather. Use manual sliders.")

    rainfall    = st.slider("Avg Rainfall (mm)",      100, 1500, int(st.session_state.get("rainfall",    700)), step=10)
    temperature = st.slider("Avg Temperature (C)",     10,   35, int(st.session_state.get("temperature",  22)), step=1)
    rain_dev    = st.slider("Rainfall Deviation (mm)",-200, 200, 0,   step=10)
    ph_level    = st.slider("Soil pH Level",           4.0,  8.0, 6.2, step=0.1)
    fertility   = st.slider("Soil Fertility Index",    0.1,  1.0, 0.7, step=0.05)
    area        = st.slider("Area Planted (ha)",       1,  100,   1,  step=1)

    st.divider()
    predict_btn = st.button("Predict Yield", use_container_width=True, type="primary")

# ── TABS ─────────────────────────────────────────────────────────────
st.title("AgroInsight Kenya")
st.caption("A Machine Learning System for Crop Yield Prediction | Kenya")
st.divider()

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home", "Prediction", "Farm Advisory",
    "Data Explorer", "Trends", "About"
])

# ── HOME ─────────────────────────────────────────────────────────────
with tab0:
    col_l, col_r = st.columns([1,1], gap="large")
    with col_l:
        st.markdown("""
        <div style="padding:40px 20px">
            <div style="background:#2ecc71;color:#0a2e1a;padding:6px 16px;border-radius:100px;
                        display:inline-block;font-size:12px;font-weight:600;letter-spacing:1px;
                        margin-bottom:24px">LIVE · POWERED BY MACHINE LEARNING</div>
            <h1 style="font-size:48px;font-weight:900;color:#0a2e1a;line-height:1.1;margin-bottom:20px">
                Predict Kenya's<br><span style="color:#1A6B3A">Crop Yields</span><br>Before You Plant
            </h1>
            <p style="font-size:17px;color:#3d5a47;line-height:1.7;margin-bottom:32px">
                AgroInsight Kenya uses machine learning trained on real Kenyan 
                agricultural data to help farmers make smarter planting decisions.
            </p>
            <div style="display:flex;gap:16px;flex-wrap:wrap">
                <div style="background:#1A6B3A;color:white;padding:14px 28px;border-radius:8px;
                            font-size:15px;font-weight:500">Click Prediction tab to get started →</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div style="background:#0a2e1a;border-radius:16px;padding:32px;margin:20px 0">
            <div style="font-size:11px;letter-spacing:1.5px;color:#a8e6c1;margin-bottom:20px;opacity:0.7">
                LIVE PREDICTION SAMPLE
            </div>
            <div style="margin-bottom:16px">
                <div style="font-size:12px;color:rgba(255,255,255,0.45);margin-bottom:6px">Rainfall suitability</div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px">
                    <div style="width:82%;height:100%;background:#2ecc71;border-radius:4px"></div></div>
            </div>
            <div style="margin-bottom:16px">
                <div style="font-size:12px;color:rgba(255,255,255,0.45);margin-bottom:6px">Soil fertility index</div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px">
                    <div style="width:74%;height:100%;background:#c9a84c;border-radius:4px"></div></div>
            </div>
            <div style="margin-bottom:16px">
                <div style="font-size:12px;color:rgba(255,255,255,0.45);margin-bottom:6px">Model confidence</div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px">
                    <div style="width:95%;height:100%;background:#2ecc71;border-radius:4px"></div></div>
            </div>
            <div style="background:rgba(46,204,113,0.1);border:1px solid rgba(46,204,113,0.2);
                        border-radius:10px;padding:16px;margin-top:24px">
                <div style="font-size:11px;color:#2ecc71;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
                    Predicted Yield · Nakuru · Maize</div>
                <div style="font-size:36px;color:white;font-weight:700">3,247 kg/ha</div>
                <div style="font-size:12px;color:rgba(255,255,255,0.4);margin-top:4px">Long Rains · 2025</div>
                <div style="display:inline-block;background:rgba(46,204,113,0.2);color:#2ecc71;
                            padding:3px 10px;border-radius:20px;font-size:11px;margin-top:8px">
                    Good · High Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<h2 style='text-align:center;color:#0a2e1a;margin-bottom:32px'>What AgroInsight Kenya offers</h2>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Counties Covered", "47", "All of Kenya")
    c2.metric("Crops Supported", "5", "Key staples")
    c3.metric("Years of Data", "10", "2015-2025")
    c4.metric("Model Accuracy", "98%+", "R2 score")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#E8F5E9;border-radius:12px;padding:24px;border-left:4px solid #1A6B3A">
            <div style="font-size:28px;margin-bottom:12px">🔮</div>
            <div style="font-size:17px;font-weight:600;color:#0a2e1a;margin-bottom:8px">Yield Prediction</div>
            <div style="font-size:14px;color:#3d5a47;line-height:1.6">
                Get county-level crop yield forecasts before you plant — 
                with confidence ratings and performance scores.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#E8F5E9;border-radius:12px;padding:24px;border-left:4px solid #1A6B3A">
            <div style="font-size:28px;margin-bottom:12px">🌱</div>
            <div style="font-size:17px;font-weight:600;color:#0a2e1a;margin-bottom:8px">Farm Advisory</div>
            <div style="font-size:14px;color:#3d5a47;line-height:1.6">
                Crop varieties, fertiliser programmes, and herbicide 
                recommendations tailored to your county and crop.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#E8F5E9;border-radius:12px;padding:24px;border-left:4px solid #1A6B3A">
            <div style="font-size:28px;margin-bottom:12px">📈</div>
            <div style="font-size:17px;font-weight:600;color:#0a2e1a;margin-bottom:8px">Historical Trends</div>
            <div style="font-size:14px;color:#3d5a47;line-height:1.6">
                Explore 10 years of yield data across all 47 Kenyan counties 
                with interactive charts and comparisons.
            </div>
        </div>""", unsafe_allow_html=True)

# ── PREDICTION ───────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        with st.spinner("Running prediction model..."):
            result = engine.predict(
                county=county, crop=crop, season=season, year=year,
                rainfall=rainfall, temperature=temperature,
                rainfall_deviation=rain_dev, ph_level=ph_level,
                fertility=fertility, area=area,
            )
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Results for {crop} in {county} — {season} {year}")
            col1, col2, col3, col4 = st.columns(4)
            total_yield = round(result["predicted_yield"] * area, 2)
            col1.metric("Yield per Hectare",  f"{result['predicted_yield']:,} kg/ha")
            col2.metric(f"Total Yield ({area} ha)", f"{total_yield:,} kg")
            col3.metric("Performance Rating", result["rating"])
            col4.metric("Model R2 Score",     str(result["model_r2"]))
            st.divider()
            rc = RATING_COLORS.get(result["rating"], "#555")
            st.markdown(f"""
            <div style="background:{rc}18;border-left:4px solid {rc};padding:16px;border-radius:8px">
                <b style="color:{rc}">Recommendation</b><br>{result["recommendation"]}
            </div>""", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Yield Gauge")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(["Poor","Fair","Good","Excellent"],[1500,1000,1000,1000],
                        color=["#F44336","#FF9800","#4CAF50","#1A6B3A"], alpha=0.6, height=0.5)
                ax.axvline(x=min(result["predicted_yield"],5000), color="navy",
                           linewidth=2.5, linestyle="--",
                           label=f"Predicted: {result['predicted_yield']:,} kg/ha")
                ax.set_xlabel("Yield (kg/ha)"); ax.set_xlim(0,5000)
                ax.legend(fontsize=9); ax.set_facecolor("#FAFAFA")
                fig.patch.set_facecolor("white")
                st.pyplot(fig); plt.close()
            with col_b:
                st.subheader("Input Summary")
                st.dataframe(pd.DataFrame({
                    "Parameter":["County","Crop","Season","Year","Rainfall","Temperature","Soil pH","Fertility","Area"],
                    "Value":[county,crop,season,str(year),f"{rainfall} mm",f"{temperature} C",str(ph_level),str(fertility),f"{area} ha"]
                }), hide_index=True, use_container_width=True)
    else:
        st.info("Set your parameters in the sidebar and click Predict Yield to get started.")
        if df is not None:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records",  f"{len(df):,}")
            c2.metric("Counties",       df["county"].nunique())
            c3.metric("Crops",          df["crop"].nunique())
            c4.metric("Avg Yield",      f"{df['yield_kg_per_ha'].mean():.0f} kg/ha")

# ── FARM ADVISORY ────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Farm Advisory — {crop} in {county}")
    advisory = get_advisory(crop, county)
    if advisory:
        st.markdown(f"""
        <div style="background:#0a2e1a;border-left:4px solid #2ecc71;padding:20px;border-radius:8px;margin-bottom:16px">
            <b style="color:#2ecc71;font-size:14px;letter-spacing:1px">RECOMMENDED VARIETIES FOR {county}</b><br>
            <span style="font-size:20px;font-weight:700;color:#ffffff">{advisory["varieties"]}</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Fertiliser Programme")
            f = advisory["fertiliser"]
            st.markdown(f"""
            | Stage | Product & Rate |
            |---|---|
            | **Basal (at planting)** | {f["basal"]} |
            | **Top Dressing** | {f["top_dressing"]} |
            """)
            st.info(f["note"])

        with col2:
            st.markdown("### Herbicide Programme")
            h = advisory["herbicides"]
            st.markdown(f"""
            | Stage | Product & Rate |
            |---|---|
            | **Pre-emergence** | {h["pre_emergence"]} |
            | **Post-emergence** | {h["post_emergence"]} |
            """)
            st.warning(h["note"])

        st.divider()
        col3, col4 = st.columns(2)
        col3.metric("Recommended Spacing",  advisory["spacing"])
        col4.metric("Planting Depth",       advisory["planting_depth"])

        st.divider()
        st.markdown("### Cost Estimate per Hectare")
        cost_data = {
            "Maize":   {"seed":800,"basal":3500,"topdress":3000,"herbicide":1200},
            "Beans":   {"seed":1200,"basal":3000,"topdress":0,"herbicide":800},
            "Wheat":   {"seed":2000,"basal":7000,"topdress":7000,"herbicide":2000},
            "Sorghum": {"seed":600,"basal":3500,"topdress":3000,"herbicide":900},
            "Potatoes":{"seed":15000,"basal":14000,"topdress":10500,"herbicide":1500},
        }
        costs = cost_data.get(crop, {})
        if costs:
            total = sum(costs.values()) * area
            st.markdown(f"""
            | Item | Cost per ha (KES) | Total for {area} ha (KES) |
            |---|---|---|
            | Seed / Planting Material | {costs["seed"]:,} | {costs["seed"]*area:,} |
            | Basal Fertiliser | {costs["basal"]:,} | {costs["basal"]*area:,} |
            | Top Dressing Fertiliser | {costs["topdress"]:,} | {costs["topdress"]*area:,} |
            | Herbicides | {costs["herbicide"]:,} | {costs["herbicide"]*area:,} |
            | **TOTAL** | **{sum(costs.values()):,}** | **{total:,}** |
            """)
    else:
        st.warning("Advisory not available for this crop.")

# ── DATA EXPLORER ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Dataset Explorer")
    if df is None:
        st.warning("Clean dataset not found. Run the training pipeline first.")
    else:
        col1, col2 = st.columns(2)
        fc = col1.multiselect("Filter by County", sorted(df["county"].unique()), default=sorted(df["county"].unique())[:3])
        fk = col2.multiselect("Filter by Crop",   sorted(df["crop"].unique()),   default=sorted(df["crop"].unique())[:2])
        filtered = df[(df["county"].isin(fc)) & (df["crop"].isin(fk))]
        st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
        st.dataframe(
            filtered[["county","crop","season","year","yield_kg_per_ha","avg_rainfall_mm","avg_temp_celsius","ph_level","fertility_index"]]
            .sort_values("yield_kg_per_ha", ascending=False).reset_index(drop=True),
            use_container_width=True, height=400)
        if len(filtered) > 0:
            st.divider()
            st.subheader("Yield Distribution by Crop")
            fig, ax = plt.subplots(figsize=(10, 4))
            ca = sorted(filtered["crop"].unique())
            bp = ax.boxplot([filtered[filtered["crop"]==c]["yield_kg_per_ha"].values for c in ca],
                           labels=ca, patch_artist=True)
            for patch, color in zip(bp["boxes"], ["#1A6B3A","#4CAF50","#81C784","#A5D6A7","#C8E6C9"]):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_ylabel("Yield (kg/ha)"); ax.set_facecolor("#FAFAFA")
            fig.patch.set_facecolor("white"); st.pyplot(fig); plt.close()

# ── TRENDS ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Yield Trends Over Time")
    if df is None:
        st.warning("Clean dataset not found.")
    else:
        sc = st.selectbox("Select Crop", sorted(df["crop"].unique()), key="tc")
        sk = st.multiselect("Select Counties", sorted(df["county"].unique()),
                            default=sorted(df["county"].unique())[:4], key="tk")
        tdf = df[(df["crop"]==sc) & (df["county"].isin(sk))]
        tg  = tdf.groupby(["year","county"])["yield_kg_per_ha"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(11, 5))
        for cn, color in zip(sk, plt.cm.Set2(np.linspace(0,1,max(len(sk),1)))):
            cd = tg[tg["county"]==cn]
            ax.plot(cd["year"], cd["yield_kg_per_ha"], marker="o", linewidth=2, label=cn, color=color)
        ax.set_xlabel("Year"); ax.set_ylabel("Average Yield (kg/ha)")
        ax.set_title(f"{sc} Yield Trends"); ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.set_facecolor("#FAFAFA"); fig.patch.set_facecolor("white")
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ── ABOUT ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("About AgroInsight Kenya")
    st.markdown("""
    **AgroInsight Kenya** applies machine learning to predict crop yields for Kenyan smallholder farmers.
    ---
    ### Technologies Used
    | Tool | Purpose |
    |---|---|
    | Python 3.12 | Core programming language |
    | Pandas & NumPy | Data processing |
    | Scikit-learn | Model training & evaluation |
    | XGBoost | Primary prediction algorithm |
    | Streamlit | Dashboard |
    | Open-Meteo API | Live weather data |

    ### Data Sources
    - Kenya National Bureau of Statistics (KNBS)
    - Kenya Meteorological Department
    - FAO GAEZ Portal
    - Ministry of Agriculture
    ---
    *AgroInsight Kenya | 2026 | Open Source*
    """)
    if engine.model_name:
        c1,c2,c3 = st.columns(3)
        c1.metric("Best Model", engine.model_name)
        c2.metric("R2 Score",   engine.model_r2)
        c3.metric("RMSE",       f"{engine.model_rmse} kg/ha")