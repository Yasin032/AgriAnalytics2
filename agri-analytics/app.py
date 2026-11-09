"""
AgriSight — Smart Agricultural Analytics (Refactored)
======================================================
A complete, production-ready rewrite of the legacy agricultural dashboard.

Refactor highlights
-------------------
1. Layout: fixed the broken/cut-off drawer headers by replacing fragile
   raw-HTML card wrappers around Streamlit widgets with Streamlit-native
   containers and a single, deduplicated CSS block. All 6 interactive
   drawers render cleanly with no clipping.
2. Visuals: the legacy Inter / forest-green theme is retained across the
   refreshed layout, charts, and interactive controls.
3. Charts: removed cluttered multi-series visuals; now uses intuitive pie /
   donut charts and clean metric cards.
4. Quality: eliminated the repeated ``st.columns(4)[0]`` bug, consolidated
   duplicated logic, hardened error handling, and added input validation.

Run with:  streamlit run agrianalytics.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Agri Analytics"
APP_TAGLINE = "Smart Agricultural Analytics & Yield Intelligence"
DATA_PATH = "crop_data.csv"
RANDOM_SEED = 42

CROP_TYPES: list[str] = ["Corn", "Wheat", "Rice", "Soybeans", "Cotton"]

# Optimal growing parameters per crop (used for data generation, scoring,
# and the recommendation guide).
CROP_PROFILES: dict[str, dict] = {
    "Corn":     {"rain": (750, 150),  "temp": (24, 4), "ph": (6.2, 0.5), "base_yield": 8.5},
    "Wheat":    {"rain": (550, 120),  "temp": (18, 4), "ph": (6.8, 0.4), "base_yield": 4.5},
    "Rice":     {"rain": (1400, 250), "temp": (28, 3), "ph": (5.8, 0.5), "base_yield": 6.0},
    "Soybeans": {"rain": (650, 130),  "temp": (23, 3), "ph": (6.5, 0.4), "base_yield": 3.0},
    "Cotton":   {"rain": (850, 180),  "temp": (27, 3), "ph": (6.0, 0.5), "base_yield": 2.5},
}

DATA_COLUMNS: list[str] = [
    "Crop_Type", "Rainfall_mm", "Temperature_C", "Soil_pH",
    "Nitrogen_N", "Phosphorus_P", "Potassium_K",
    "Fertilizer_kg_ha", "Yield_tons_ha",
]

CROP_EMOJI = {"Corn": "🌽", "Wheat": "🌾", "Rice": "🍚", "Soybeans": "🫘", "Cotton": "☁️"}

# Legacy forest-green palette
C_BG        = "#f8faf7"
C_SIDEBAR   = "#1a3c1d"
C_SURFACE   = "#ffffff"
C_BORDER    = "#e8ede6"
C_TEXT      = "#263626"
C_TEXT_MUT  = "#6b7f6b"
C_ACCENT    = "#1a3c1d"
C_ACCENT_2  = "#2e7d32"
C_WARN      = "#ef6c00"
C_DANGER    = "#c62828"

PIE_PALETTE = px.colors.sequential.Greens[-5:][::-1]

# ---------------------------------------------------------------------------
# Styling — legacy green theme adapted to the refactored native layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important; color: __TEXT__;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: __BG__ !important;
    }
    .block-container {
        width: 100%; max-width: 1400px;
        padding: 1.5rem clamp(1rem, 3vw, 2rem) 2rem;
    }
    /* ---- HEAVY-DUTY FORCED FULL-WIDTH HORIZONTAL NAVIGATION OVERRIDE ---- */
    div[data-testid="stTabs"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    div[data-baseweb="tab-list"] {
        display: flex !important; 
        width: 100% !important; 
        max-width: 100% !important;
        gap: 10px !important;
        background: #E2E8F0 !important; /* Forces uniform slate row background */
        padding: 6px !important; 
        border-radius: 8px !important;
    }
    
    /* Target the buttons directly and force each to expand equally to fill the row */
    div[data-baseweb="tab-list"] > button {
        flex: 1 1 0% !important; 
        max-width: 100% !important;
        min-width: 0 !important;
        text-align: center !important;
        justify-content: center !important;
        border-radius: 6px !important; 
        font-weight: 600 !important; 
        font-size: 14px !important; 
        padding: 12px 0px !important;
        background: transparent !important;
    }
    
    /* Active selected tab state formatting */
    div[data-baseweb="tab-list"] > button[aria-selected="true"] {
        background: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: __SIDEBAR__ !important;
        border-right: 1px solid __SIDEBAR__;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
        color: #e8f5e9 !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #c8e6c9 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {
        color: __TEXT__ !important;
    }

    /* ---- Header ---- */
    .app-header { text-align: center; padding: 20px 0 30px; }
    .app-header h1 { font-size: 32px !important; font-weight: 700 !important; color: __ACCENT__ !important; margin: 0 0 6px !important; }
    .app-header h1 span { color: __ACCENT__ !important; }
    .app-header p { font-size: 14px; color: __TEXT_MUT__; margin: 0; }

    /* ---- Section labels ---- */
    .section-label {
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.2px; color: __TEXT_MUT__; margin: 22px 0 10px 0;
    }
    [data-testid="stSidebar"] .section-label { color: #c8e6c9; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 16px;
        padding: 24px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center; transition: box-shadow 0.2s ease;
    }
    .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .metric-card .icon { font-size: 32px; margin-bottom: 8px; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: __ACCENT__; margin: 4px 0; overflow-wrap: anywhere; }
    .metric-card .label { font-size: 13px; font-weight: 500; color: __TEXT_MUT__; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .delta { font-size: 12px; font-weight: 600; margin-top: 6px; }
    .delta.up, .delta.down, .delta.flat { color: __ACCENT_2__; }

    /* ---- Interactive drawers ---- */
    [data-testid="stExpander"] {
        background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 8px;
        margin-bottom: 10px; overflow: visible;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stExpander"] summary {
        font-size: 14px; font-weight: 600; color: __ACCENT__ !important;
        background: __SURFACE__ !important; padding: 14px 18px;
    }
    [data-testid="stExpander"] summary * { color: __ACCENT__ !important; }
    [data-testid="stExpander"] summary:hover { background: #e8f5e9 !important; }
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        padding: 4px 18px 16px 18px; gap: 0.6rem;
    }

    /* ---- Prediction banner ---- */
    .prediction-banner {
        border-radius: 16px; padding: 28px 24px; text-align: center;
        border: 1px solid __BORDER__; background: __SURFACE__;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .prediction-banner.high { border: 2px solid #4caf50; background: #e8f5e9; color: #1b5e20; }
    .prediction-banner.mid  { border: 2px solid #ffb300; background: #fff8e1; color: #e65100; }
    .prediction-banner.low  { border: 2px solid #ef5350; background: #ffebee; color: #b71c1c; }
    .prediction-banner .big { font-size: 44px; font-weight: 800; color: inherit; overflow-wrap: anywhere; }
    .prediction-banner .tag { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: inherit; }
    .prediction-banner .advice { font-size: 13px; color: inherit; margin-top: 10px; line-height: 1.6; }

    /* ---- Condition pills ---- */
    .pill { display: inline-block; padding: 3px 11px; border-radius: 999px;
            font-size: 11px; font-weight: 700; margin: 2px 4px 2px 0; }
    .pill.ok { background: #e8f5e9; color: __ACCENT_2__; }
    .pill.warn { background: #fff3e0; color: #ef6c00; }
    .pill.bad { background: #ffebee; color: __DANGER__; }

    /* ---- Widget polish ---- */
    .stApp label,
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stSlider"] p,
    .stApp [data-testid="stThumbValue"] {
        color: __TEXT__ !important;
    }
    .stApp [data-baseweb="tag"] {
        background: #e8f5e9 !important;
        color: __ACCENT__ !important;
        border: 1px solid __BORDER__;
    }
    .stApp [data-baseweb="tag"] * { color: __ACCENT__ !important; }
    .stApp [data-testid="stSlider"] [role="slider"] {
        background: __ACCENT__ !important;
    }
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp input:not([type="range"]), .stApp textarea {
        background-color: __SURFACE__ !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }
    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="input"] * { color: __TEXT__ !important; }
    .stApp [data-baseweb="select"]:focus-within,
    .stApp [data-baseweb="input"]:focus-within {
        border-color: __ACCENT__ !important;
        box-shadow: 0 0 0 2px rgba(26, 60, 29, 0.16);
    }
    .stApp [data-testid="stNumberInputStepDown"],
    .stApp [data-testid="stNumberInputStepUp"] {
        background: #e8f5e9 !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }
    .stApp [data-testid="stNumberInputStepDown"] svg,
    .stApp [data-testid="stNumberInputStepUp"] svg {
        fill: currentColor !important;
    }
    .stApp [data-testid="stSidebar"] label,
    .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stSidebar"] [data-testid="stSlider"] p {
        color: #e8f5e9 !important;
    }
    .stApp [data-testid="stSidebar"] .stMarkdown,
    .stApp [data-testid="stSidebar"] .stMarkdown p,
    .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #c8e6c9 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: __SURFACE__ !important;
        border: 1px dashed __BORDER__ !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: __TEXT__ !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #e8f5e9 !important;
        border: 1px solid __BORDER__ !important;
    }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        background: __ACCENT__ !important; color: #fff !important;
        border: none !important; border-radius: 10px;
        padding: 10px 24px; font-weight: 600; font-size: 13px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover { background: __ACCENT_2__ !important; color: #fff !important; box-shadow: 0 4px 12px rgba(26,60,29,0.25); }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    .stFormSubmitButton > button:focus-visible { outline: 2px solid __ACCENT__; outline-offset: 2px; }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        background: #e8f5e9 !important; color: __ACCENT__ !important;
        border: 1px solid #c8e6c9 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        background: __ACCENT_2__ !important; color: #fff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex; width: 100%; gap: 8px;
        background: __BG__; padding: 4px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0; min-width: 0; justify-content: center;
        padding: 10px 12px !important; border-radius: 10px !important;
        font-weight: 500 !important; font-size: 13px !important; color: #5a7a5a !important;
        background: transparent !important; border: none !important;
        transition: background 0.15s ease, color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #e8f5e9 !important; color: __ACCENT__ !important; }
    .stTabs [aria-selected="true"] { background: __ACCENT__ !important; color: #fff !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 40%; min-width: max-content;
            padding: 8px 6px !important; font-size: 12px !important;
        }
        .metric-card { padding: 18px 12px; }
        .metric-card .value { font-size: 23px; }
    }

    hr.soft { border: none; border-top: 1px solid __BORDER__; margin: 18px 0; }

    header[data-testid="stHeader"] { background: transparent; }
    header [data-testid="stExpandSidebarButton"] {
        color: __ACCENT__ !important; background: #e8f5e9;
        border: 1px solid __BORDER__; border-radius: 8px;
    }
    </style>
""".replace("__BG__", C_BG).replace("__SIDEBAR__", C_SIDEBAR) \
    .replace("__SURFACE__", C_SURFACE) \
     .replace("__BORDER__", C_BORDER).replace("__TEXT__", C_TEXT) \
     .replace("__TEXT_MUT__", C_TEXT_MUT).replace("__ACCENT__", C_ACCENT) \
     .replace("__ACCENT_2__", C_ACCENT_2).replace("__DANGER__", C_DANGER)


# ---------------------------------------------------------------------------
# Data engine
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_rows: int = 550, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic records whose yield correlates with environment."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    per_crop = n_rows // len(CROP_TYPES)
    remainder = n_rows % len(CROP_TYPES)

    for idx, crop in enumerate(CROP_TYPES):
        prof = CROP_PROFILES[crop]
        count = per_crop + (1 if idx < remainder else 0)
        rain = rng.uniform(200, 1800, count)
        temp = rng.uniform(12, 42, count)
        ph = rng.uniform(4.5, 8.5, count)
        n = rng.uniform(10, 250, count)
        p = rng.uniform(10, 120, count)
        k = rng.uniform(20, 250, count)
        fert = rng.uniform(0, 300, count)

        for i in range(count):
            rain_s = 1 - abs(rain[i] - prof["rain"][0]) / 800
            temp_s = 1 - abs(temp[i] - prof["temp"][0]) / 15
            ph_s = 1 - abs(ph[i] - prof["ph"][0]) / 2
            score = (0.35 * np.clip(rain_s, 0, 1) + 0.30 * np.clip(temp_s, 0, 1)
                     + 0.10 * np.clip(ph_s, 0, 1) + 0.05 * min(1, n[i] / 210)
                     + 0.05 * min(1, p[i] / 82.5) + 0.05 * min(1, k[i] / 210)
                     + 0.10 * min(1, fert[i] / 270))
            y = max(0.3, round(prof["base_yield"] * score * rng.normal(1, 0.05), 2))
            rows.append({
                "Crop_Type": crop, "Rainfall_mm": round(rain[i], 1),
                "Temperature_C": round(temp[i], 1), "Soil_pH": round(ph[i], 2),
                "Nitrogen_N": round(n[i], 1), "Phosphorus_P": round(p[i], 1),
                "Potassium_K": round(k[i], 1), "Fertilizer_kg_ha": round(fert[i], 1),
                "Yield_tons_ha": y,
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_or_generate_data() -> pd.DataFrame:
    """Load the persisted CSV when valid; otherwise generate and persist."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if list(df.columns) == DATA_COLUMNS and len(df) >= 10:
                return df
            st.warning("Existing CSV has unexpected columns — regenerating dataset.")
        except Exception as exc:  # noqa: BLE001 — surface any read failure
            st.warning(f"Could not read CSV ({exc}) — regenerating dataset.")
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# ML engine
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Container for the trained model and its evaluation metadata."""
    model: RandomForestRegressor
    encoder: LabelEncoder
    r2: float
    rmse: float
    importances: pd.DataFrame = field(default_factory=pd.DataFrame)


FEATURE_NAMES = ["Crop", "Rainfall", "Temperature", "Soil pH",
                 "Nitrogen", "Phosphorus", "Potassium", "Fertilizer"]


def train_model(df: pd.DataFrame) -> ModelBundle | None:
    """Train a Random Forest regressor; return None with an error on failure."""
    try:
        data = df.copy()
        le = LabelEncoder()
        data["Crop_Encoded"] = le.fit_transform(data["Crop_Type"])
        X = data[["Crop_Encoded", "Rainfall_mm", "Temperature_C", "Soil_pH",
                  "Nitrogen_N", "Phosphorus_P", "Potassium_K", "Fertilizer_kg_ha"]]
        y = data["Yield_tons_ha"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED)
        model = RandomForestRegressor(
            n_estimators=150, max_depth=15, min_samples_split=5,
            random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        importances = (pd.DataFrame({"Feature": FEATURE_NAMES,
                                     "Importance": model.feature_importances_})
                       .sort_values("Importance", ascending=False)
                       .reset_index(drop=True))
        return ModelBundle(
            model=model, encoder=le,
            r2=round(r2_score(y_test, pred), 4),
            rmse=round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            importances=importances)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model training failed: {exc}")
        return None


def predict_yield(bundle: ModelBundle, crop: str, rain: float, temp: float,
                  ph: float, n: float, p: float, k: float, fert: float) -> float:
    """Predict yield (t/ha); returns 0 on unexpected failure."""
    try:
        row = pd.DataFrame([{
            "Crop_Encoded": bundle.encoder.transform([crop])[0],
            "Rainfall_mm": rain, "Temperature_C": temp, "Soil_pH": ph,
            "Nitrogen_N": n, "Phosphorus_P": p, "Potassium_K": k,
            "Fertilizer_kg_ha": fert}])
        return max(0.0, round(float(bundle.model.predict(row)[0]), 2))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULT_INPUTS = {"crop": "Corn", "rainfall": 750.0, "temperature": 24.0,
                  "ph": 6.5, "nitrogen": 130.0, "phosphorus": 50.0,
                  "potassium": 120.0, "fertilizer": 150.0}


def init_session() -> None:
    if "df" not in st.session_state:
        st.session_state.df = load_or_generate_data()
    if "bundle" not in st.session_state:
        st.session_state.bundle = train_model(st.session_state.df)
    if "inputs" not in st.session_state:
        st.session_state.inputs = dict(DEFAULT_INPUTS)


def persist_and_retrain(df: pd.DataFrame) -> bool:
    """Validate, persist, and retrain. Returns True on success."""
    df = df.dropna(subset=["Crop_Type"])
    if len(df) < 10:
        st.error("At least 10 valid rows are required to retrain the model.")
        return False
    try:
        df.to_csv(DATA_PATH, index=False)
        bundle = train_model(df)
        if bundle is None:
            return False
        st.session_state.df = df
        st.session_state.bundle = bundle
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to save data: {exc}")
        return False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="app-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True)


def metric_card(col, icon: str, value: str, label: str, delta: str = "",
                delta_cls: str = "flat") -> None:
    """Render one metric card inside a pre-created column (fixed: single columns() call)."""
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="icon">{icon}</div>'
            f'<div class="value">{value}</div><div class="label">{label}</div>'
            f'<div class="delta {delta_cls}">{delta}</div></div>',
            unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title_font=dict(size=14, color=C_ACCENT),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11)))
    axis_style = dict(
        title_font=dict(color=C_TEXT, size=12),
        tickfont=dict(color=C_TEXT, size=11),
        gridcolor="#d8e2d6", linecolor="#758875",
        zerolinecolor="#a5b4a3", showline=True, ticks="outside")
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def condition_pill(value: float, lo: float, hi: float, name: str) -> str:
    """Classify a value against an optimal range with 25% / 50% tolerance bands."""
    tol = (hi - lo) * 0.25
    if lo - tol <= value <= hi + tol:
        return f'<span class="pill ok">✓ {name}</span>'
    if lo - 2 * tol <= value <= hi + 2 * tol:
        return f'<span class="pill warn">~ {name}</span>'
    return f'<span class="pill bad">✗ {name}</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    with st.sidebar:
        st.markdown(
            f"### {APP_TITLE} 🌱\n"
            f"<span style='color:#c8e6c9;font-size:12px;'>"
            f"{APP_TAGLINE}</span>",
            unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
        selected = st.multiselect("Crop types", CROP_TYPES, default=CROP_TYPES)
        rain_rng = st.slider("Rainfall (mm)", 200, 2000, (200, 2000))
        temp_rng = st.slider("Temperature (°C)", 10, 45, (10, 45))

        st.divider()
        st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
        if st.button("🔄 Regenerate demo data", use_container_width=True):
            df = generate_synthetic_data()
            if persist_and_retrain(df):
                st.success("Dataset regenerated & model retrained.")
                st.rerun()

        st.download_button("📥 Download CSV", st.session_state.df.to_csv(index=False),
                           DATA_PATH, "text/csv", use_container_width=True)

        st.divider()
        st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
        bundle = st.session_state.get("bundle")
        rows = len(st.session_state.df)
        st.caption(f"{rows:,} records · {st.session_state.df['Crop_Type'].nunique()} crops")
        if bundle:
            st.caption(f"Model R² **{bundle.r2}** · RMSE **{bundle.rmse}** t/ha")
    return selected, rain_rng, temp_rng


# ---------------------------------------------------------------------------
# Tab 1 — Analytics dashboard (pie charts + metric cards + drawers)
# ---------------------------------------------------------------------------

def render_dashboard(selected: list[str], rain_rng, temp_rng) -> None:
    page_header(f"{APP_TITLE} <span>Dashboard</span>",
                "Clean, glanceable insights into crop performance")
    df = st.session_state.df
    f = df[df["Crop_Type"].isin(selected)
           & df["Rainfall_mm"].between(*rain_rng)
           & df["Temperature_C"].between(*temp_rng)].copy()
    if f.empty:
        st.warning("No records match the current filters — adjust the sidebar.")
        return

    # ---- Metric cards: created ONCE, then filled (fixes old columns bug) ----
    avg_yield = f["Yield_tons_ha"].mean()
    best = f.groupby("Crop_Type")["Yield_tons_ha"].mean().sort_values(ascending=False)
    best_crop, best_yield = best.index[0], best.iloc[0]
    rain_at_peak = f.loc[f["Yield_tons_ha"].idxmax(), "Rainfall_mm"]

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "🌱", f"{len(f):,}", "Records", "in current view")
    metric_card(c2, "📈", f"{avg_yield:.2f}", "Avg yield (t/ha)", "global mean")
    metric_card(c3, "🌧️", f"{rain_at_peak:.0f} mm", "Rainfall at peak", "best record")
    metric_card(c4, CROP_EMOJI.get(best_crop, "🏆"), best_crop, "Top crop", f"{best_yield:.2f} t/ha", "up")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- Easy-to-read charts: pie / donut focus ----
    left, right = st.columns(2)
    with left:
        share = f.groupby("Crop_Type")["Yield_tons_ha"].sum().reset_index()
        fig = px.pie(share, names="Crop_Type", values="Yield_tons_ha", hole=0.55,
                     color="Crop_Type", color_discrete_sequence=PIE_PALETTE,
                     title="Total yield share by crop")
        fig.update_traces(textposition="inside", textinfo="percent+label",
                  textfont_size=11, textfont_color=C_TEXT)
        st.plotly_chart(style_fig(fig), use_container_width=True,
                        config={"displayModeBar": False})
    with right:
        counts = f["Crop_Type"].value_counts().reset_index()
        counts.columns = ["Crop_Type", "Records"]
        fig = px.pie(counts, names="Crop_Type", values="Records", hole=0.55,
                     color="Crop_Type", color_discrete_sequence=PIE_PALETTE,
                     title="Dataset composition by crop")
        fig.update_traces(textposition="inside", textinfo="percent+label",
                  textfont_size=11, textfont_color=C_TEXT)
        st.plotly_chart(style_fig(fig), use_container_width=True,
                        config={"displayModeBar": False})

    # Simple per-crop average bar (single series, sorted — instantly readable)
    fig = px.bar(best.reset_index().rename(columns={"index": "Crop"}),
                 x="Crop_Type", y="Yield_tons_ha",
                 color="Yield_tons_ha", color_continuous_scale="Greens",
                 labels={"Yield_tons_ha": "Avg yield (t/ha)", "Crop_Type": ""},
                 title="Average yield per crop")
    fig.update_layout(coloraxis_showscale=False, showlegend=False)
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside",
                      textfont=dict(size=11, color=C_TEXT))
    st.plotly_chart(style_fig(fig, 320), use_container_width=True,
                    config={"displayModeBar": False})

    # ---- Six interactive drawers (fully visible, no clipping) ----
    st.markdown('<div class="section-label">Explore the data</div>',
                unsafe_allow_html=True)

    with st.expander("📋 Summary statistics", expanded=True):
        summary = (f.groupby("Crop_Type")
                   .agg(Records=("Yield_tons_ha", "size"),
                        Yield_Mean=("Yield_tons_ha", "mean"),
                        Yield_Min=("Yield_tons_ha", "min"),
                        Yield_Max=("Yield_tons_ha", "max"),
                        Avg_Rain=("Rainfall_mm", "mean"),
                        Avg_Temp=("Temperature_C", "mean"),
                        Avg_pH=("Soil_pH", "mean"))
                   .round(2).reset_index())
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("🔬 Soil pH distribution"):
        fig = px.histogram(f, x="Soil_pH", nbins=24, color_discrete_sequence=[C_ACCENT_2],
                           labels={"Soil_pH": "Soil pH", "count": "Records"},
                           title="Soil pH distribution")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                        config={"displayModeBar": False})

    with st.expander("🌧️ Rainfall vs yield"):
        fig = px.scatter(f, x="Rainfall_mm", y="Yield_tons_ha", color="Crop_Type",
                         color_discrete_sequence=PIE_PALETTE, opacity=0.65,
                         labels={"Rainfall_mm": "Rainfall (mm)", "Yield_tons_ha": "Yield (t/ha)"},
                         title="Rainfall vs yield by crop")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True,
                        config={"displayModeBar": False})

    with st.expander("🧾 Raw records"):
        st.dataframe(f, use_container_width=True, hide_index=True, height=320)

    with st.expander("ℹ️ Column definitions"):
        defs = pd.DataFrame({
            "Column": DATA_COLUMNS,
            "Meaning": [
                "Crop variety", "Seasonal rainfall in mm", "Mean temperature in °C",
                "Soil acidity (0–14)", "Nitrogen level", "Phosphorus level",
                "Potassium level", "Fertilizer applied (kg/ha)",
                "Observed yield (tons/hectare) — target column"],
        })
        st.dataframe(defs, use_container_width=True, hide_index=True)

    with st.expander("🤖 Model performance"):
        bundle = st.session_state.get("bundle")
        if bundle:
            m1, m2 = st.columns(2)
            m1.metric("R² score", bundle.r2)
            m2.metric("RMSE (t/ha)", bundle.rmse)
            imp = bundle.importances.sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[C_ACCENT],
                         title="What drives yield (feature importance)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("Model is not trained yet.")


# ---------------------------------------------------------------------------
# Tab 2 — Data management
# ---------------------------------------------------------------------------

def render_data_manager() -> None:
    page_header("Data <span>Manager</span>",
                "Upload, edit, and commit records — the model retrains automatically")
    df = st.session_state.df

    with st.expander("📁 Upload a CSV dataset", expanded=True):
        st.caption("Required columns: " + ", ".join(DATA_COLUMNS))
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
                missing = [c for c in DATA_COLUMNS if c not in new_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    new_df = new_df[DATA_COLUMNS]
                    if persist_and_retrain(new_df):
                        st.success(f"Loaded {len(new_df):,} records and retrained the model.")
                        st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")

    with st.expander("➕ Add a single record"):
        with st.form("add_record", clear_on_submit=True):
            a1, a2, a3 = st.columns(3)
            crop = a1.selectbox("Crop", CROP_TYPES)
            rain = a2.number_input("Rainfall (mm)", 0.0, 3000.0, 750.0, 10.0)
            temp = a3.number_input("Temperature (°C)", -10.0, 55.0, 24.0, 0.5)
            b1, b2, b3 = st.columns(3)
            ph = b1.number_input("Soil pH", 3.0, 10.0, 6.5, 0.1)
            n = b2.number_input("Nitrogen", 0.0, 300.0, 130.0, 5.0)
            p = b3.number_input("Phosphorus", 0.0, 150.0, 50.0, 5.0)
            c1, c2 = st.columns(2)
            k = c1.number_input("Potassium", 0.0, 300.0, 120.0, 5.0)
            y = c2.number_input("Observed yield (t/ha)", 0.0, 50.0, 5.0, 0.1)
            if st.form_submit_button("Add record", use_container_width=True):
                row = pd.DataFrame([{
                    "Crop_Type": crop, "Rainfall_mm": rain, "Temperature_C": temp,
                    "Soil_pH": ph, "Nitrogen_N": n, "Phosphorus_P": p,
                    "Potassium_K": k, "Fertilizer_kg_ha": 150.0, "Yield_tons_ha": y}])
                if persist_and_retrain(pd.concat([df, row], ignore_index=True)):
                    st.success("Record added and model retrained.")
                    st.rerun()

    with st.expander("✏️ Edit the ledger & commit"):
        st.caption("Edit any cell, add or delete rows, then commit. "
                   "The CSV is overwritten and the model is retrained.")
        col_cfg = {
            "Crop_Type": st.column_config.SelectboxColumn("Crop", options=CROP_TYPES, required=True),
            "Rainfall_mm": st.column_config.NumberColumn("Rainfall (mm)", min_value=0.0),
            "Temperature_C": st.column_config.NumberColumn("Temp (°C)"),
            "Soil_pH": st.column_config.NumberColumn("Soil pH", min_value=0.0, max_value=14.0),
            "Nitrogen_N": st.column_config.NumberColumn("Nitrogen", min_value=0.0),
            "Phosphorus_P": st.column_config.NumberColumn("Phosphorus", min_value=0.0),
            "Potassium_K": st.column_config.NumberColumn("Potassium", min_value=0.0),
            "Fertilizer_kg_ha": st.column_config.NumberColumn("Fertilizer (kg/ha)", min_value=0.0),
            "Yield_tons_ha": st.column_config.NumberColumn("Yield (t/ha)", min_value=0.0),
        }
        edited = st.data_editor(df.copy(), column_config=col_cfg,
                                num_rows="dynamic", use_container_width=True,
                                height=420, key="ledger_editor")
        if st.button("💾 Commit changes & retrain", type="primary"):
            if persist_and_retrain(edited):
                st.success(f"Saved {len(st.session_state.df):,} rows. Model retrained.")
                st.rerun()

    with st.expander("🗑️ Reset to demo data"):
        st.caption("Discard all changes and restore the built-in synthetic dataset.")
        if st.button("Reset dataset"):
            df = generate_synthetic_data()
            if persist_and_retrain(df):
                st.success("Demo dataset restored.")
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Yield predictor
# ---------------------------------------------------------------------------

def render_predictor() -> None:
    page_header("Yield <span>Predictor</span>",
                "Adjust conditions to get an instant ML-powered yield estimate")
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.error("The model could not be trained. Check your data in the Data Manager tab.")
        return

    inp = st.session_state.inputs
    ctrl, out = st.columns([1.1, 1])

    with ctrl:
        crop = st.selectbox("Crop type", CROP_TYPES, index=CROP_TYPES.index(inp["crop"]))
        rain = st.slider("🌧️ Rainfall (mm)", 200, 2000, int(inp["rainfall"]), 10)
        temp = st.slider("🌡️ Temperature (°C)", 10, 45, int(inp["temperature"]), 1)
        ph = st.slider("⚗️ Soil pH", 4.5, 8.5, float(inp["ph"]), 0.1)
        n = st.slider("🔬 Nitrogen (kg/ha)", 5, 250, int(inp["nitrogen"]), 5)
        p = st.slider("🔬 Phosphorus (kg/ha)", 5, 150, int(inp["phosphorus"]), 5)
        k = st.slider("🔬 Potassium (kg/ha)", 10, 250, int(inp["potassium"]), 5)
        fert = st.slider("🧪 Fertilizer (kg/ha)", 0, 300, int(inp["fertilizer"]), 5)

    prediction = predict_yield(bundle, crop, rain, temp, ph, n, p, k, fert)
    base = CROP_PROFILES[crop]["base_yield"]
    if prediction >= base * 1.1:
        cls, tag, advice = "high", "🟢 Excellent yield", \
            "Conditions are highly favorable — expect a strong harvest."
    elif prediction >= base * 0.85:
        cls, tag, advice = "mid", "🟡 Moderate yield", \
            "Yield is acceptable. Fine-tune irrigation and fertilizer to improve."
    else:
        cls, tag, advice = "low", "🔴 Suboptimal yield", \
            "Conditions are challenging — review rainfall, temperature, and soil pH."

    with out:
        st.markdown(
            f'<div class="prediction-banner {cls}"><div class="tag">{tag}</div>'
            f'<div class="big">{prediction:.2f}</div>'
            f'<div class="tag">tons per hectare</div>'
            f'<div class="advice">{advice}</div></div>',
            unsafe_allow_html=True)
        st.caption(f"Model: R² = {bundle.r2} · RMSE = {bundle.rmse} t/ha")

        # Simple, readable comparison vs. history
        hist = st.session_state.df[st.session_state.df["Crop_Type"] == crop]
        if not hist.empty:
            hist_mean = float(hist["Yield_tons_ha"].mean())
            fig = go.Figure(go.Bar(
                x=["Historical avg", "Your prediction"],
                y=[hist_mean, prediction],
                marker_color=[C_ACCENT_2, C_ACCENT],
                text=[f"{hist_mean:.2f}", f"{prediction:.2f}"],
                textposition="outside", width=0.45))
            fig.update_layout(showlegend=False, yaxis_title="t/ha",
                              title=f"{CROP_EMOJI.get(crop, '')} {crop}: history vs. prediction")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                            config={"displayModeBar": False})
            diff = prediction - hist_mean
            if diff > 0:
                st.success(f"🌟 {diff:.2f} t/ha above the historical average ({hist_mean:.2f}).")
            elif diff > -0.5:
                st.info(f"👍 Roughly in line with the historical average ({hist_mean:.2f} t/ha).")
            else:
                st.warning(f"⚠️ {abs(diff):.2f} t/ha below the historical average ({hist_mean:.2f}).")

    st.session_state.inputs = {
        "crop": crop, "rainfall": rain, "temperature": temp, "ph": ph,
        "nitrogen": n, "phosphorus": p, "potassium": k, "fertilizer": fert}


# ---------------------------------------------------------------------------
# Tab 4 — Recommendation guide
# ---------------------------------------------------------------------------

CROP_ADVICE = {
    "Corn": "Warm-season crop and a heavy nitrogen feeder. Plant when soil exceeds 10°C, keep rows spaced for pollination, and watch for borers and aphids.",
    "Wheat": "Prefers cool seasons and loamy, well-drained soil. Split nitrogen between sowing and stem elongation; monitor for rust and mildew.",
    "Rice": "Water-intensive; thrives in flooded paddies with high humidity. Transplant seedlings at 3–4 weeks and control weeds early.",
    "Soybeans": "A nitrogen-fixing legume that needs little fertilizer. Inoculate seeds with rhizobium and avoid waterlogging during pod fill.",
    "Cotton": "Needs a long, warm season with full sun. Irrigate during flowering and boll set; harvest promptly once bolls open.",
}

TIPS = [
    ("🧪 Soil testing", "Test soil annually to guide precise fertilizer application."),
    ("💧 Water management", "Use drip irrigation and water early morning or late evening."),
    ("🌱 Crop rotation", "Rotate crops to prevent nutrient depletion and break pest cycles."),
    ("🐛 IPM", "Combine biological, cultural, and chemical pest control sustainably."),
    ("📊 Record keeping", "Log inputs, weather, and yields to optimize future seasons."),
    ("🌿 Cover crops", "Plant clover or rye off-season to fix nitrogen and prevent erosion."),
    ("☀️ Weather monitoring", "Track rainfall, temperature, and frost warnings with local stations."),
    ("🔄 Precision agriculture", "Apply inputs only where needed with GPS-guided equipment."),
]


def render_guide() -> None:
    page_header("Crop <span>Guide</span>",
                "Per-crop recommendations evaluated against your simulated conditions")
    inp = st.session_state.inputs
    cur_crop, cur_rain = inp["crop"], inp["rainfall"]
    cur_temp, cur_ph = inp["temperature"], inp["ph"]

    with st.expander("🎯 Your current simulated conditions", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Crop", cur_crop)
        c2.metric("Rainfall", f"{cur_rain:.0f} mm")
        c3.metric("Temperature", f"{cur_temp:.0f} °C")
        c4.metric("Soil pH", f"{cur_ph:.1f}")

    for crop in CROP_TYPES:
        prof = CROP_PROFILES[crop]
        r_lo, r_hi = prof["rain"][0] - prof["rain"][1], prof["rain"][0] + prof["rain"][1]
        t_lo, t_hi = prof["temp"][0] - prof["temp"][1], prof["temp"][0] + prof["temp"][1]
        ph_lo, ph_hi = prof["ph"][0] - prof["ph"][1], prof["ph"][0] + prof["ph"][1]
        pills = (condition_pill(cur_rain, r_lo, r_hi, "Rainfall")
                 + condition_pill(cur_temp, t_lo, t_hi, "Temp")
                 + condition_pill(cur_ph, ph_lo, ph_hi, "pH"))
        active = crop == cur_crop
        with st.expander(
                f"{CROP_EMOJI.get(crop, '🌱')} {crop}"
                + ("  ·  current selection" if active else ""),
                expanded=active):
            st.markdown(pills, unsafe_allow_html=True)
            st.write(CROP_ADVICE[crop])
            st.caption(f"Optimal ranges — Rainfall {r_lo:.0f}–{r_hi:.0f} mm · "
                       f"Temp {t_lo:.0f}–{t_hi:.0f} °C · pH {ph_lo:.1f}–{ph_hi:.1f}")

    with st.expander("💡 Universal best practices"):
        for title, body in TIPS:
            st.markdown(f"**{title}** — {body}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌱",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session()
    selected, rain_rng, temp_rng = render_sidebar()

    tab_dash, tab_data, tab_pred, tab_guide = st.tabs(
        ["📊 Dashboard", "🚜 Data Manager", "🔮 Predictor", "🌿 Guide"])
    with tab_dash:
        render_dashboard(selected, rain_rng, temp_rng)
    with tab_data:
        render_data_manager()
    with tab_pred:
        render_predictor()
    with tab_guide:
        render_guide()

    st.caption(f"{APP_TITLE} v3.0 — data is simulated for demonstration. "
               "Consult agronomists for real farming decisions.")


if __name__ == "__main__":
    main()
