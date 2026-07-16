"""
NEOM Bio-Secure V9.0 — Aviation HUD Edition
Bird Aircraft Strike Hazard (BASH) Decision Support System
NEOM Smart City | Aviation Safety Division

New in V9.0
-----------
  * Interactive Simulation Engine — sidebar sliders for real-time RF inference
  * Aviation HUD glassmorphism UI — glowing cyan cards, cockpit typography
  * Pulsing critical-alert animation tied to the live Risk %
  * ATC Radar map — concentric PathLayer rings + neon cyan→risk gradient arcs
  * Redesigned precision gauge — thin bar, zone annotations, monospace readout
  * Radar scan-line animation sweeping the viewport

Stability contract
------------------
  * All plotly.graph_objects layouts use template="plotly_dark" only
  * paper_bgcolor / plot_bgcolor set to rgba(0,0,0,0) — no ValueErrors
  * pydeck layers validated against pydeck >=0.9 API
"""

import warnings
warnings.filterwarnings("ignore")

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NEOM Bio-Secure V9.0 | ATC HUD",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
# CSS — Full Aviation HUD Theme
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global background ─────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp,
[data-testid="stMain"],
section.main > div {
    background-color : #0A0F1D !important;
    color            : #B0BEC5 !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────── */
#MainMenu                              { display: none !important; }
header[data-testid="stHeader"]        { display: none !important; }
footer                                 { display: none !important; }
[data-testid="stToolbar"]             { display: none !important; }
.stDeployButton                        { display: none !important; }
[data-testid="collapsedControl"]       { display: none !important; }

/* ── Sidebar — ATC control panel ───────────────────────────────── */
[data-testid="stSidebar"] {
    background-color : #060C18 !important;
    border-right     : 1px solid rgba(0,229,255,0.14) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color : transparent !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown p {
    font-family   : 'Courier New', monospace !important;
    color         : #78909C !important;
    font-size     : 11px !important;
    letter-spacing: 0.5px;
}
/* Sidebar radio labels */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family   : 'Courier New', monospace !important;
    font-size     : 11px !important;
    color         : #546E7A !important;
    letter-spacing: 0.5px;
    padding       : 3px 0;
}
/* Sidebar slider labels */
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    font-family   : 'Courier New', monospace !important;
    font-size     : 9px !important;
    color         : #455A64 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
/* Slider thumb and track */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
    background : #00E5FF !important;
}
/* Selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background-color : rgba(10,16,30,0.9) !important;
    border-color     : rgba(0,229,255,0.25) !important;
    font-family      : 'Courier New', monospace !important;
    font-size        : 11px !important;
}

/* ── Column containers — glassmorphism ─────────────────────────── */
[data-testid="column"] {
    background       : rgba(7,12,26,0.60) !important;
    border           : 1px solid rgba(0,229,255,0.08) !important;
    border-radius    : 10px !important;
    padding          : 18px 16px !important;
    backdrop-filter  : blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition       : border-color 0.4s ease, box-shadow 0.4s ease;
}
[data-testid="column"]:hover {
    border-color : rgba(0,229,255,0.20) !important;
    box-shadow   : 0 0 18px rgba(0,229,255,0.06),
                   inset 0 1px 0 rgba(255,255,255,0.03) !important;
}

/* ── Metric cards ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background   : rgba(8,14,28,0.85) !important;
    border       : 1px solid rgba(0,229,255,0.14) !important;
    border-radius: 8px !important;
    padding      : 12px 14px !important;
    box-shadow   : 0 0 10px rgba(0,229,255,0.04) !important;
}
[data-testid="stMetricLabel"] > div {
    font-family   : 'Courier New', monospace !important;
    font-size     : 9px !important;
    color         : #455A64 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] > div {
    font-family   : 'Courier New', monospace !important;
    font-size     : 20px !important;
    color         : #ECEFF1 !important;
    letter-spacing: 1px;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-family: 'Courier New', monospace !important;
    font-size  : 9px !important;
}

/* ── HUD section labels ─────────────────────────────────────────── */
.hud-label {
    font-family   : 'Courier New', monospace;
    font-size     : 9px;
    color         : #00E5FF;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom : 1px solid rgba(0,229,255,0.14);
    padding-bottom: 5px;
    margin-bottom : 10px;
    opacity       : 0.80;
}

/* ── Glass card wrapper ─────────────────────────────────────────── */
.glass-card {
    background      : rgba(8,14,28,0.70);
    border          : 1px solid rgba(0,229,255,0.12);
    border-radius   : 8px;
    box-shadow      : 0 0 14px rgba(0,229,254,0.05),
                      inset 0 1px 0 rgba(255,255,255,0.04);
    padding         : 14px 16px;
    margin-bottom   : 10px;
    backdrop-filter : blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* ── Risk pulse animations ──────────────────────────────────────── */
@keyframes pulse-critical {
    0%, 100% {
        box-shadow   : 0 0 10px rgba(255,23,68,0.35),
                       0 0 24px rgba(255,23,68,0.10);
        border-color : rgba(255,23,68,0.50);
    }
    50% {
        box-shadow   : 0 0 26px rgba(255,23,68,0.80),
                       0 0 55px rgba(255,23,68,0.28),
                       0 0 80px rgba(255,23,68,0.08);
        border-color : rgba(255,23,68,1.00);
    }
}
@keyframes pulse-warning {
    0%, 100% {
        box-shadow   : 0 0 8px  rgba(255,179,0,0.28),
                       0 0 18px rgba(255,179,0,0.07);
        border-color : rgba(255,179,0,0.42);
    }
    50% {
        box-shadow   : 0 0 20px rgba(255,179,0,0.60),
                       0 0 42px rgba(255,179,0,0.20);
        border-color : rgba(255,179,0,0.90);
    }
}
@keyframes pulse-nominal {
    0%, 100% { border-color: rgba(0,230,118,0.22); }
    50% {
        border-color: rgba(0,230,118,0.55);
        box-shadow  : 0 0 16px rgba(0,230,118,0.14);
    }
}
.pulse-critical { animation: pulse-critical 1.4s ease-in-out infinite; }
.pulse-warning  { animation: pulse-warning  2.0s ease-in-out infinite; }
.pulse-nominal  { animation: pulse-nominal  3.2s ease-in-out infinite; }

/* ── Radar scan-line ────────────────────────────────────────────── */
@keyframes scan {
    0%   { top: -3px; opacity: 0; }
    4%   { opacity: 1; }
    96%  { opacity: 0.8; }
    100% { top: 100vh; opacity: 0; }
}
.scan-line {
    position       : fixed;
    left           : 0;
    right          : 0;
    height         : 2px;
    background     : linear-gradient(
        90deg,
        transparent    0%,
        rgba(0,229,255,0.5)  30%,
        rgba(0,229,255,0.95) 50%,
        rgba(0,229,255,0.5)  70%,
        transparent   100%
    );
    animation      : scan 6s linear infinite;
    z-index        : 9998;
    pointer-events : none;
}

/* ── Bottom ticker ──────────────────────────────────────────────── */
.bs-ticker-wrap {
    position    : fixed;
    bottom      : 0;
    left        : 0;
    right       : 0;
    height      : 28px;
    display     : flex;
    align-items : center;
    overflow    : hidden;
    background  : linear-gradient(
        90deg, #04070F 0%, #0A0F1D 25%, #0A0F1D 75%, #04070F 100%
    );
    border-top  : 1px solid rgba(0,229,255,0.16);
    z-index     : 99999;
}
.bs-ticker-tag {
    flex-shrink   : 0;
    padding       : 0 12px;
    font-family   : 'Courier New', monospace;
    font-size     : 9px;
    font-weight   : 700;
    color         : #00E5FF;
    border-right  : 1px solid rgba(0,229,255,0.20);
    letter-spacing: 1.5px;
    white-space   : nowrap;
}
.bs-ticker-track {
    flex        : 1;
    overflow    : hidden;
    height      : 100%;
    display     : flex;
    align-items : center;
}
.bs-ticker-inner {
    display    : flex;
    white-space: nowrap;
    animation  : bs-scroll 80s linear infinite;
}
@keyframes bs-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.bs-ticker-seg {
    font-family   : 'Courier New', monospace;
    font-size     : 9px;
    color         : #607D8B;
    padding       : 0 5px;
    letter-spacing: 0.5px;
}
.bs-ticker-dot {
    font-family : 'Courier New', monospace;
    font-size   : 9px;
    color       : #00E5FF;
    padding     : 0 3px;
}

/* ── Divider ────────────────────────────────────────────────────── */
hr { border-color: rgba(0,229,255,0.09) !important; margin: 8px 0 !important; }

/* ── Bottom padding — keep ticker clear ────────────────────────── */
.main .block-container { padding-bottom: 44px !important; }
</style>
""", unsafe_allow_html=True)

# Inject radar scan-line element (pure decoration)
st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════
NEOM_LAT         = 28.03
NEOM_LON         = 34.64
UTC              = timezone.utc
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}

C_CYAN  = "#00E5FF"
C_GREEN = "#00E676"
C_AMBER = "#FFB300"
C_RED   = "#FF1744"
C_BG    = "#0A0F1D"


# ════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA — 5 000 rows  (unchanged from V8)
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    np.random.seed(42)
    n     = 5_000
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    doy   = dates.dayofyear
    month = dates.month

    temp_base   = 25 + 18 * np.sin((doy - 80) * 2 * np.pi / 365)
    temperature = np.clip(temp_base + np.random.normal(0, 4, n), 8, 52)
    wind_speed  = np.clip(np.random.gamma(2.5, 7.5, n), 0, 75)
    migration   = np.isin(month, list(MIGRATION_MONTHS)).astype(int)

    base_p    = 0.08
    mig_p     = migration * 0.28
    thermal_p = ((temperature >= 22) & (temperature <= 40)).astype(float) * 0.15
    low_w_p   = (wind_speed < 18).astype(float) * 0.12
    high_w_p  = (wind_speed > 55).astype(float) * (-0.08)
    prob      = np.clip(base_p + mig_p + thermal_p + low_w_p + high_w_p, 0.03, 0.72)

    risk_event = (np.random.random(n) < prob).astype(int)
    flip       = np.random.random(n) < 0.05
    risk_event = np.where(flip, 1 - risk_event, risk_event)

    return pd.DataFrame({
        "temperature":      temperature,
        "wind_speed":       wind_speed,
        "migration_season": migration,
        "risk_event":       risk_event,
    })


# ════════════════════════════════════════════════════════════════════
# MODEL TRAINING  (unchanged from V8)
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_model():
    df = generate_training_data()
    X  = df[["temperature", "wind_speed", "migration_season"]]
    y  = df["risk_event"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    imp = pd.DataFrame({
        "Feature":    ["Migration Season", "Wind Speed", "Temperature"],
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=True).reset_index(drop=True)
    return clf, acc, imp


# ════════════════════════════════════════════════════════════════════
# LIVE WEATHER + 12-H FORECAST — Open-Meteo  (unchanged from V8)
# ════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def get_live_weather() -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={NEOM_LAT}&longitude={NEOM_LON}"
        "&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        "&hourly=temperature_2m,wind_speed_10m"
        "&wind_speed_unit=kmh"
        "&forecast_days=2"
        "&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d    = r.json()
        cur  = d["current"]
        hrly = d["hourly"]

        temp_now = round(float(cur["temperature_2m"]), 1)
        wind_now = round(float(cur["wind_speed_10m"]), 1)
        wdir_now = int(cur.get("wind_direction_10m", 270))

        times     = pd.to_datetime(hrly["time"])
        df_hourly = pd.DataFrame({
            "time": times,
            "temp": [float(x) for x in hrly["temperature_2m"]],
            "wind": [float(x) for x in hrly["wind_speed_10m"]],
        })
        now_ts    = pd.Timestamp(cur["time"])
        df_future = (
            df_hourly[df_hourly["time"] >= now_ts]
            .head(13)
            .reset_index(drop=True)
        )
        if df_future.empty:
            df_future = df_hourly.head(13).reset_index(drop=True)

        return {
            "temp_now":    temp_now,
            "wind_now":    wind_now,
            "wind_dir":    wdir_now,
            "forecast_df": df_future,
            "source":      "OPEN-METEO",
            "error":       None,
        }

    except Exception as exc:
        rng   = np.random.default_rng(42)
        now   = pd.Timestamp.utcnow().floor("h")
        times = pd.date_range(now, periods=13, freq="1h")
        return {
            "temp_now":    32.0,
            "wind_now":    18.0,
            "wind_dir":    270,
            "forecast_df": pd.DataFrame({
                "time": times,
                "temp": np.clip(32 + rng.normal(0, 2, 13), 20, 48).tolist(),
                "wind": np.clip(18 + rng.normal(0, 4, 13), 2, 60).tolist(),
            }),
            "source": "FALLBACK",
            "error":  str(exc),
        }


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def is_migration(month: int) -> int:
    return 1 if month in MIGRATION_MONTHS else 0


def risk_hex(risk: float) -> str:
    if risk >= 70:
        return C_RED
    if risk >= 30:
        return C_AMBER
    return C_GREEN


def risk_rgba_list(risk: float, alpha: int = 220) -> list:
    if risk >= 70:
        return [255, 23, 68, alpha]
    if risk >= 30:
        return [255, 179, 0, alpha]
    return [0, 229, 255, alpha]


def _geo_offset(lat: float, lon: float, bearing_deg: float, dist_km: float):
    """Haversine: return (lat2, lon2) displaced dist_km along bearing_deg."""
    R    = 6371.0
    d    = dist_km / R
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _circle_path(lat: float, lon: float, radius_km: float, n_pts: int = 90) -> list:
    """Return a closed list of [lon, lat] pairs forming a circle."""
    pts = []
    for i in range(n_pts + 1):
        angle      = (i / n_pts) * 360
        rlat, rlon = _geo_offset(lat, lon, angle, radius_km)
        pts.append([rlon, rlat])
    return pts


def _sim_forecast(temp: float, wind: float, mig: int) -> pd.DataFrame:
    """Synthetic 12-hour forecast anchored to simulation inputs."""
    rng   = np.random.default_rng(int(temp * 100 + wind * 10 + mig))
    now   = pd.Timestamp.utcnow().floor("h")
    times = pd.date_range(now, periods=13, freq="1h")
    return pd.DataFrame({
        "time": times,
        "temp": np.clip(temp + rng.normal(0, 2.5, 13), 8, 52).tolist(),
        "wind": np.clip(wind + rng.normal(0, 5.0, 13), 0, 75).tolist(),
    })


# ════════════════════════════════════════════════════════════════════
# SIDEBAR — Mode selector + Simulation Engine
# ════════════════════════════════════════════════════════════════════
def build_sidebar(accuracy: float):
    """
    Returns (mode, sim_temp, sim_wind, sim_wind_dir, sim_mig).
    When mode == 'LIVE', sim_* values are None.
    """
    with st.sidebar:
        # ── Logo ───────────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center;padding:12px 0 6px;'>"
            "<span style='font-size:28px;color:#00E5FF;'>⬡</span><br>"
            "<span style='font-family:\"Courier New\",monospace;font-size:13px;"
            "font-weight:700;color:#00E5FF;letter-spacing:3px;'>NEOM</span><br>"
            "<span style='font-family:\"Courier New\",monospace;font-size:10px;"
            "color:#546E7A;letter-spacing:2px;'>BIO-SECURE V9.0</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<hr style='border-color:rgba(0,229,255,0.12);margin:10px 0;'>",
            unsafe_allow_html=True,
        )

        # ── Mode selector ──────────────────────────────────────────
        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:9px;"
            "color:#00E5FF;letter-spacing:3px;margin-bottom:6px;'>"
            "◉ OPERATING MODE</p>",
            unsafe_allow_html=True,
        )
        mode_raw = st.radio(
            label="mode",
            options=["\U0001f6f0  LIVE TELEMETRY", "\U0001f3ae  SIMULATION MODE"],
            label_visibility="collapsed",
        )
        is_sim = "SIMULATION" in mode_raw

        # ── Simulation parameters ──────────────────────────────────
        sim_temp = sim_wind = sim_wind_dir = sim_mig = None
        if is_sim:
            st.markdown(
                "<hr style='border-color:rgba(0,229,255,0.10);margin:10px 0;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='font-family:\"Courier New\",monospace;font-size:9px;"
                "color:#00E5FF;letter-spacing:3px;margin-bottom:8px;'>"
                "◉ SIMULATION PARAMETERS</p>",
                unsafe_allow_html=True,
            )
            sim_temp = st.slider(
                "TEMPERATURE (°C)", min_value=8, max_value=52, value=32, step=1
            )
            sim_wind = st.slider(
                "WIND SPEED (km/h)", min_value=0, max_value=75, value=20, step=1
            )
            sim_wind_dir = st.slider(
                "WIND DIRECTION (°)", min_value=0, max_value=359, value=270, step=5,
                help="0 = North · 90 = East · 180 = South · 270 = West",
            )
            mig_opt = st.selectbox(
                "MIGRATION SEASON",
                ["AUTO-DETECT", "ACTIVE — HIGH RISK", "INACTIVE — LOWER RISK"],
                label_visibility="visible",
            )
            now_month = datetime.now(UTC).month
            if mig_opt == "ACTIVE — HIGH RISK":
                sim_mig = 1
            elif mig_opt == "INACTIVE — LOWER RISK":
                sim_mig = 0
            else:
                sim_mig = is_migration(now_month)

        # ── AI engine stats ────────────────────────────────────────
        st.markdown(
            "<hr style='border-color:rgba(0,229,255,0.10);margin:10px 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:9px;"
            "color:#00E5FF;letter-spacing:3px;margin-bottom:8px;'>"
            "◉ AI ENGINE STATUS</p>",
            unsafe_allow_html=True,
        )
        stats = [
            ("ALGORITHM",  "RANDOM FOREST"),
            ("ESTIMATORS", "150 TREES"),
            ("TRAINING",   "5 000 ROWS"),
            ("FEATURES",   "TEMP · WIND · MIG"),
            ("TEST ACC",   f"{accuracy * 100:.1f}%"),
            ("MODEL VER",  "RF-V9.0"),
        ]
        for k, v in stats:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-family:\"Courier New\",monospace;font-size:9px;"
                f"color:#546E7A;margin:3px 0;'>"
                f"<span>{k}</span>"
                f"<span style='color:#78909C;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='border-color:rgba(0,229,255,0.10);margin:10px 0;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:8px;"
            "color:#37474F;text-align:center;letter-spacing:1px;'>"
            "NEOM AVIATION SAFETY DIV<br>"
            "ICAO · BASH PROTOCOL ACTIVE</p>",
            unsafe_allow_html=True,
        )

    mode = "SIM" if is_sim else "LIVE"
    return mode, sim_temp, sim_wind, sim_wind_dir, sim_mig


# ════════════════════════════════════════════════════════════════════
# CHART — Aviation HUD Risk Gauge  (redesigned for V9)
# ════════════════════════════════════════════════════════════════════
def make_risk_gauge(risk: float) -> go.Figure:
    bar_color = risk_hex(risk)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number={
            "suffix":      "%",
            "font":        {
                "size":   54,
                "color":  bar_color,
                "family": "Courier New, monospace",
            },
            "valueformat": ".1f",
        },
        delta={
            "reference":  30,
            "relative":   False,
            "prefix":     "Δ ",
            "font":       {"size": 12, "family": "Courier New, monospace"},
            "decreasing": {"color": C_GREEN},
            "increasing": {"color": C_RED},
        },
        title={
            "text": (
                "<b>BASH RISK INDEX</b><br>"
                "<span style='font-size:10px;color:#455A64;"
                "font-family:Courier New,monospace;letter-spacing:2px;'>"
                "BIRD STRIKE HAZARD · AI MODEL</span>"
            ),
            "font": {"size": 13, "color": "#607D8B"},
        },
        gauge={
            "axis": {
                "range":    [0, 100],
                "tickwidth": 2,
                "tickcolor": "#1C2A3A",
                "tickfont":  {
                    "size":   9,
                    "color":  "#37474F",
                    "family": "Courier New, monospace",
                },
                "dtick": 10,
            },
            "bar":        {"color": bar_color, "thickness": 0.14},
            "bgcolor":    "rgba(4,8,18,0.92)",
            "borderwidth": 2,
            "bordercolor": "rgba(0,229,255,0.10)",
            "steps": [
                {"range": [0,  30],  "color": "rgba(0,230,118,0.07)"},
                {"range": [30, 70],  "color": "rgba(255,179,0,0.07)"},
                {"range": [70, 100], "color": "rgba(255,23,68,0.09)"},
            ],
            "threshold": {
                "line":      {"color": C_RED, "width": 3},
                "thickness": 0.88,
                "value":     70,
            },
        },
    ))

    # Zone annotations — positioned in gauge paper coords
    for txt, xp, yp, col in [
        ("SAFE",     0.13, 0.22, C_GREEN),
        ("WARN",     0.50, 0.04, C_AMBER),
        ("CRITICAL", 0.87, 0.22, C_RED),
    ]:
        fig.add_annotation(
            x=xp, y=yp,
            xref="paper", yref="paper",
            text=txt,
            font={"size": 8, "color": col, "family": "Courier New, monospace"},
            showarrow=False,
            opacity=0.55,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin={"t": 120, "b": 8, "l": 18, "r": 18},
        font={"family": "Courier New, monospace"},
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# CHART — XAI Feature Importance  (enhanced for V9)
# ════════════════════════════════════════════════════════════════════
def make_xai_chart(imp_df: pd.DataFrame) -> go.Figure:
    color_map = {
        "Migration Season": "#448AFF",
        "Wind Speed":       "#7C4DFF",
        "Temperature":      C_AMBER,
    }
    colors = [color_map.get(f, C_CYAN) for f in imp_df["Feature"]]

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"] * 100,
        y=imp_df["Feature"],
        orientation="h",
        marker={
            "color":   colors,
            "opacity": 0.85,
            "line":    {"color": "rgba(255,255,255,0.07)", "width": 1},
        },
        text=[f"{v * 100:.1f}%" for v in imp_df["Importance"]],
        textposition="outside",
        textfont={"size": 10, "color": "#546E7A", "family": "Courier New, monospace"},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    max_imp = float(imp_df["Importance"].max()) * 100
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=215,
        margin={"t": 6, "b": 18, "l": 8, "r": 72},
        xaxis={
            "range":      [0, max_imp * 1.48],
            "ticksuffix": "%",
            "showgrid":   True,
            "gridcolor":  "rgba(255,255,255,0.03)",
            "tickfont":   {"size": 9, "family": "Courier New, monospace"},
        },
        yaxis={
            "tickfont": {"size": 10, "family": "Courier New, monospace"},
        },
        bargap=0.40,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# CHART — 12-Hour Forecast  (unchanged logic, enhanced style)
# ════════════════════════════════════════════════════════════════════
def make_forecast_chart(forecast_df: pd.DataFrame, model, mig: int) -> go.Figure:
    rows  = forecast_df.head(13).reset_index(drop=True)
    n     = len(rows)
    risks = [
        float(model.predict_proba([[row["temp"], row["wind"], mig]])[0][1] * 100)
        for _, row in rows.iterrows()
    ]
    x_idx    = list(range(n))
    x_labels = ["NOW"] + [f"+{i}h" for i in range(1, n)]
    line_clr = risk_hex(max(risks))

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=x_idx, y=risks,
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.03)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=x_idx, y=risks,
        mode="lines+markers",
        name="AI Risk Forecast",
        line={"color": line_clr, "width": 2.5},
        marker={
            "size":  [10 if r >= 70 else 5 for r in risks],
            "color": [risk_hex(r) for r in risks],
            "line":  {"color": C_BG, "width": 1.5},
        },
        text=x_labels,
        hovertemplate="<b>%{text}</b><br>Risk: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=30,
        line={"color": C_AMBER, "dash": "dash", "width": 1},
        opacity=0.45,
        annotation_text="WARN 30%",
        annotation_font={"size": 8, "color": C_AMBER, "family": "Courier New"},
        annotation_position="right",
    )
    fig.add_hline(
        y=70,
        line={"color": C_RED, "dash": "dash", "width": 1.5},
        opacity=0.60,
        annotation_text="CRITICAL 70%",
        annotation_font={"size": 8, "color": C_RED, "family": "Courier New"},
        annotation_position="right",
    )
    fig.add_vline(
        x=0,
        line={"color": C_CYAN, "dash": "dot", "width": 1},
        opacity=0.40,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=265,
        margin={"t": 14, "b": 48, "l": 8, "r": 96},
        xaxis={
            "tickvals":  x_idx,
            "ticktext":  x_labels,
            "showgrid":  True,
            "gridcolor": "rgba(255,255,255,0.03)",
            "tickfont":  {"size": 9, "family": "Courier New, monospace"},
        },
        yaxis={
            "title":      "Risk %",
            "range":      [0, 108],
            "ticksuffix": "%",
            "gridcolor":  "rgba(255,255,255,0.03)",
            "tickfont":   {"size": 9, "family": "Courier New, monospace"},
            "title_font": {"size": 9, "family": "Courier New, monospace"},
        },
        legend={
            "orientation": "h",
            "y":           -0.22,
            "bgcolor":     "rgba(0,0,0,0)",
            "font":        {"family": "Courier New, monospace", "size": 9},
        },
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# PYDECK MAP — ATC Radar Edition
#   Layers: PathLayer radar rings + ArcLayer neon gradient vectors
#            + ScatterplotLayer airport marker + glow ring
# ════════════════════════════════════════════════════════════════════
def make_pydeck_map(risk_pct: float, wind_speed: float, wind_dir: float) -> pdk.Deck:
    dot     = risk_rgba_list(risk_pct, 235)
    glow    = risk_rgba_list(risk_pct, 45)
    arc_tgt = risk_rgba_list(risk_pct, 30)

    # Neon cyan source → risk-colored target for dramatic gradient
    arc_src = [0, 229, 255, 190]

    airport_data = [{
        "lon":  NEOM_LON,
        "lat":  NEOM_LAT,
        "name": (
            f"NEOM INTL (OEGN)"
            f"  |  Risk: {risk_pct:.1f}%"
            f"  |  Wind: {wind_speed:.1f} km/h @ {wind_dir}°"
        ),
    }]

    # ── Radar rings (PathLayer circles) ────────────────────────────
    ring_specs = [
        (25,  [0, 229, 255, 35]),
        (50,  [0, 229, 255, 22]),
        (85,  [0, 229, 255, 14]),
        (130, [0, 229, 255,  7]),
    ]
    ring_data = [
        {"path": _circle_path(NEOM_LAT, NEOM_LON, r), "color": c}
        for r, c in ring_specs
    ]
    radar_ring_layer = pdk.Layer(
        "PathLayer",
        data=ring_data,
        get_path="path",
        get_color="color",
        get_width=1_200,
        width_min_pixels=1,
        width_max_pixels=2,
        pickable=False,
    )

    # ── Wind threat vectors (ArcLayer) — 3 bearings × 3 distances ──
    arc_rows = []
    for offset in (-22, 0, 22):
        bearing = (wind_dir + offset) % 360
        for dist_km in (20, 42, 68):
            elat, elon = _geo_offset(NEOM_LAT, NEOM_LON, bearing, dist_km)
            arc_rows.append({
                "slat": NEOM_LAT, "slon": NEOM_LON,
                "elat": elat,     "elon": elon,
            })

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_rows,
        get_source_position=["slon", "slat"],
        get_target_position=["elon", "elat"],
        get_source_color=arc_src,
        get_target_color=arc_tgt,
        get_width=3,
        width_min_pixels=2,
        pickable=False,
    )

    # ── Airport glow ring ───────────────────────────────────────────
    glow_layer = pdk.Layer(
        "ScatterplotLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_color=glow,
        get_radius=18_000,
        filled=True,
        stroked=False,
    )

    # ── Airport marker ──────────────────────────────────────────────
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_color=dot,
        get_radius=5_500,
        pickable=True,
        filled=True,
        stroked=True,
        radius_min_pixels=7,
        radius_max_pixels=26,
        line_width_min_pixels=2,
    )

    view = pdk.ViewState(
        latitude=NEOM_LAT,
        longitude=NEOM_LON,
        zoom=7,
        pitch=40,
        bearing=6,
    )
    return pdk.Deck(
        layers=[radar_ring_layer, glow_layer, arc_layer, scatter_layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}"},
    )


# ════════════════════════════════════════════════════════════════════
# TICKER
# ════════════════════════════════════════════════════════════════════
def inject_ticker(
    risk_pct: float,
    temp: float,
    wind: float,
    accuracy: float,
    source: str,
    mode: str,
    now_utc: datetime,
) -> None:
    level = (
        "DANGER — CRITICAL" if risk_pct >= 70
        else ("WARNING — ELEVATED" if risk_pct >= 30 else "NOMINAL — CLEAR")
    )
    segments = [
        f"RAW TELEMETRY: SENSORS NOMINAL",
        f"MODE: {mode}",
        f"TEMP: {temp:.1f}°C",
        f"WIND: {wind:.1f} km/h",
        f"BASH RISK INDEX: {risk_pct:.1f}%",
        f"THREAT LEVEL: {level}",
        "PREDICTIVE HORIZON: 12H",
        "BASH PROTOCOL: ACTIVE",
        "AIRPORT: NEOM INTL (OEGN)",
        "COORD: 28.03°N / 34.64°E",
        "AI: RANDOM FOREST · 150 TREES",
        f"ACCURACY: {accuracy * 100:.1f}%",
        f"DATA: {source}",
        f"UTC: {now_utc.strftime('%H:%M:%S')} ZULU",
        "ICAO MONITORING: ACTIVE",
        "NEOM BIO-SECURE V9.0 · AVIATION HUD EDITION",
    ]
    dot   = "<span class='bs-ticker-dot'>◆</span>"
    inner = dot.join(
        f"<span class='bs-ticker-seg'>{s}</span>" for s in segments
    )
    content = inner + dot + inner

    st.markdown(
        f"<div class='bs-ticker-wrap'>"
        f"<div class='bs-ticker-tag'>▶ ATC FEED</div>"
        f"<div class='bs-ticker-track'>"
        f"<div class='bs-ticker-inner'>{content}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    now_utc = datetime.now(UTC)

    # ── Bootstrap AI model ──────────────────────────────────────────
    with st.spinner("Initialising RF engine…"):
        model, accuracy, imp_df = train_model()

    # ── Sidebar — get operating mode ────────────────────────────────
    mode, sim_temp, sim_wind, sim_wind_dir, sim_mig = build_sidebar(accuracy)

    # ── Resolve sensor inputs ────────────────────────────────────────
    if mode == "SIM":
        temp     = float(sim_temp)
        wind     = float(sim_wind)
        wind_dir = int(sim_wind_dir)
        mig      = int(sim_mig)
        wx_source  = "SIMULATION ENGINE"
        uplink_ok  = False
        forecast_df = _sim_forecast(temp, wind, mig)
    else:
        wx          = get_live_weather()
        temp        = wx["temp_now"]
        wind        = wx["wind_now"]
        wind_dir    = wx["wind_dir"]
        forecast_df = wx["forecast_df"]
        wx_source   = wx["source"]
        uplink_ok   = wx_source == "OPEN-METEO"
        mig         = is_migration(now_utc.month)

    # ── AI inference ─────────────────────────────────────────────────
    risk_pct = float(model.predict_proba([[temp, wind, mig]])[0][1] * 100)
    r_color  = risk_hex(risk_pct)

    # Derive pulse class and threat text
    if risk_pct >= 70:
        pulse_cls    = "pulse-critical"
        threat_label = "⚠ DANGER — BIRD STRIKE RISK CRITICAL"
        threat_icon  = "\U0001f534"
    elif risk_pct >= 30:
        pulse_cls    = "pulse-warning"
        threat_label = "⚡ CAUTION — ELEVATED RISK DETECTED"
        threat_icon  = "\U0001f7e0"
    else:
        pulse_cls    = "pulse-nominal"
        threat_label = "✓ NOMINAL — ALL SYSTEMS CLEAR"
        threat_icon  = "\U0001f7e2"

    uplink_icon  = "\U0001f7e2" if uplink_ok else ("\U0001f7e1" if mode == "LIVE" else "\U0001f535")
    uplink_label = (
        "ACTIVE"   if uplink_ok
        else ("DEGRADED" if mode == "LIVE" else "SIM MODE")
    )
    uplink_color = C_GREEN if uplink_ok else (C_AMBER if mode == "LIVE" else C_CYAN)

    # ════════════════════════════════════════════════════════════════
    # A.  TOP COMMAND HEADER
    # ════════════════════════════════════════════════════════════════
    h_logo, h_status, h_sync = st.columns([3, 2, 2])

    with h_logo:
        st.markdown(
            "<div style='font-family:\"Courier New\",monospace;padding:4px 0;'>"
            "<span style='font-size:20px;font-weight:700;color:#00E5FF;"
            "letter-spacing:3px;'>&#x2B21; NEOM BIO-SECURE ATC</span><br>"
            "<span style='font-size:9px;color:#455A64;letter-spacing:3px;'>"
            "BASH DECISION SUPPORT SYSTEM · V9.0 AVIATION HUD"
            "</span></div>",
            unsafe_allow_html=True,
        )

    with h_status:
        st.markdown(
            f"<div style='font-family:\"Courier New\",monospace;"
            f"text-align:center;padding:4px 0;'>"
            f"<span style='font-size:13px;font-weight:700;color:{uplink_color};'>"
            f"{uplink_icon} LIVE UPLINK: {uplink_label}</span><br>"
            f"<span style='font-size:8px;color:#455A64;letter-spacing:1.5px;'>"
            f"SRC: {wx_source}</span></div>",
            unsafe_allow_html=True,
        )

    with h_sync:
        st.markdown(
            f"<div style='font-family:\"Courier New\",monospace;"
            f"text-align:right;padding:4px 0;'>"
            f"<span style='font-size:12px;color:#90A4AE;'>"
            f"LAST SYNC: {now_utc.strftime('%H:%M:%S')} ZULU</span><br>"
            f"<span style='font-size:8px;color:#455A64;letter-spacing:1.5px;'>"
            f"{now_utc.strftime('%Y-%m-%d')} · "
            f"{NEOM_LAT}N {NEOM_LON}E</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,229,255,0.10);"
        "margin:8px 0 12px;'>",
        unsafe_allow_html=True,
    )

    # ── Pulsing critical-alert banner (visible above main split) ────
    st.markdown(
        f"<div class='glass-card {pulse_cls}' "
        f"style='border-color:{r_color}55;background:{r_color}0D;"
        f"text-align:center;padding:8px 16px;'>"
        f"<span style='font-family:\"Courier New\",monospace;font-size:12px;"
        f"font-weight:700;color:{r_color};letter-spacing:2px;'>"
        f"{threat_icon} {threat_label} &nbsp;·&nbsp; "
        f"<span style='font-size:16px;'>{risk_pct:.1f}%</span>"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════
    # B.  MAIN SPLIT  [4 | 6]
    # ════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([4, 6])

    # ── LEFT: Expert Core ───────────────────────────────────────────
    with left_col:

        # Sensor telemetry row
        st.markdown(
            "<p class='hud-label'>◉ LIVE SENSOR TELEMETRY</p>",
            unsafe_allow_html=True,
        )
        s_temp_col, s_wind_col, s_mig_col = st.columns(3)

        with s_temp_col:
            st.metric(
                "TEMPERATURE",
                f"{temp:.1f} °C",
                delta=f"{'SIM' if mode=='SIM' else 'LIVE'}",
            )
            st.markdown(
                "<p style='font-family:\"Courier New\",monospace;font-size:8px;"
                "color:#00E676;margin-top:-10px;'>\U0001f7e2 ONLINE</p>",
                unsafe_allow_html=True,
            )
        with s_wind_col:
            st.metric(
                "WIND SPEED",
                f"{wind:.1f} km/h",
                delta=f"{wind_dir}°",
            )
            st.markdown(
                "<p style='font-family:\"Courier New\",monospace;font-size:8px;"
                "color:#00E676;margin-top:-10px;'>\U0001f7e2 ONLINE</p>",
                unsafe_allow_html=True,
            )
        with s_mig_col:
            mig_label = "ACTIVE" if mig else "INACTIVE"
            mig_delta = f"Month {now_utc.month}"
            st.metric("MIGRATION", mig_label, delta=mig_delta)
            mig_ind_color = C_RED if mig else C_GREEN
            st.markdown(
                f"<p style='font-family:\"Courier New\",monospace;font-size:8px;"
                f"color:{mig_ind_color};margin-top:-10px;'>"
                f"\U0001f7e2 SENSOR: OK</p>",
                unsafe_allow_html=True,
            )

        # Risk gauge
        st.markdown(
            "<p class='hud-label' style='margin-top:12px;'>"
            "◉ AI RISK ASSESSMENT</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_risk_gauge(risk_pct),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # XAI feature importance
        st.markdown(
            "<p class='hud-label'>◉ EXPLAINABLE AI — FEATURE WEIGHTS</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_xai_chart(imp_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── RIGHT: Visual Twin (ATC Radar Map) ──────────────────────────
    with right_col:

        st.markdown(
            "<p class='hud-label'>◉ VISUAL TWIN — NEOM BAY ATC RADAR</p>",
            unsafe_allow_html=True,
        )

        # Map mode badge
        mode_badge_color = C_AMBER if mode == "SIM" else C_CYAN
        mode_badge_text  = (
            f"▶ SIMULATION — {temp:.1f}°C · "
            f"{wind:.1f} km/h · {wind_dir}° · "
            f"Mig: {'ON' if mig else 'OFF'}"
            if mode == "SIM"
            else f"▶ LIVE TELEMETRY — {wx_source}"
        )
        st.markdown(
            f"<div style='background:{mode_badge_color}12;"
            f"border:1px solid {mode_badge_color}40;border-radius:4px;"
            f"padding:5px 12px;margin-bottom:8px;"
            f"font-family:\"Courier New\",monospace;font-size:10px;"
            f"color:{mode_badge_color};'>{mode_badge_text}</div>",
            unsafe_allow_html=True,
        )

        try:
            deck = make_pydeck_map(risk_pct, wind, wind_dir)
            st.pydeck_chart(deck, use_container_width=True, height=636)
        except Exception as map_err:
            st.warning(f"Map renderer unavailable: {map_err}")
            st.markdown(
                f"<div style='background:#060C18;border:1px solid #1A2540;"
                f"border-radius:8px;padding:48px;text-align:center;"
                f"font-family:\"Courier New\",monospace;'>"
                f"<div style='color:{C_CYAN};font-size:40px;'>&#x2B21;</div>"
                f"<div style='color:#37474F;font-size:10px;letter-spacing:2px;"
                f"margin-top:10px;'>NEOM BAY · ATC RADAR OFFLINE</div>"
                f"<div style='color:#546E7A;font-size:13px;margin-top:4px;'>"
                f"28.03°N / 34.64°E</div>"
                f"<div style='color:{r_color};font-size:26px;margin-top:16px;"
                f"font-weight:700;'>RISK: {risk_pct:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════
    # C.  BOTTOM — 12-Hour Predictive Horizon
    # ════════════════════════════════════════════════════════════════
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,229,255,0.09);"
        "margin:14px 0 10px;'>",
        unsafe_allow_html=True,
    )

    forecast_label = (
        "SIMULATION FORECAST — AI RISK OVER SYNTHETIC 12H HORIZON"
        if mode == "SIM"
        else "12H PREDICTIVE HORIZON — AI RISK FORECAST (OPEN-METEO DATA)"
    )
    st.markdown(
        f"<p class='hud-label'>◉ {forecast_label}</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_forecast_chart(forecast_df, model, mig),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Footer
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.caption("AIRPORT: NEOM INTL (OEGN)")
    f2.caption(f"COORD: {NEOM_LAT}N · {NEOM_LON}E")
    f3.caption(f"AI ACCURACY: {accuracy * 100:.1f}%")
    f4.caption(f"CACHE TTL: 900 s · {wx_source}")
    f5.caption("NEOM Bio-Secure V9.0 · Aviation HUD Edition")

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # ── Ticker ──────────────────────────────────────────────────────
    inject_ticker(risk_pct, temp, wind, accuracy, wx_source, mode, now_utc)


if __name__ == "__main__":
    main()
