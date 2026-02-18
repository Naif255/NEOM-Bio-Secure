"""
╔══════════════════════════════════════════════════════════════════╗
║          NEOM Bio-Secure  —  Version 5.0                        ║
║    AI-Powered Bird Strike Risk Prediction Platform               ║
║    NEOM Smart City  |  Aviation Safety Division                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
from datetime import datetime

# ════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG & BRANDING
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEOM Bio-Secure v5.0",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEOM = {
    "primary":    "#00D4AA",   # NEOM Teal
    "secondary":  "#1A1A2E",   # Dark Navy
    "accent":     "#16213E",   # Deep Blue
    "card":       "#0F3460",   # Card background
    "success":    "#00FF88",   # Bright Green
    "warning":    "#FFD700",   # Gold
    "danger":     "#FF4444",   # Red
    "text":       "#FFFFFF",   # White
    "muted":      "#A0AEC0",   # Muted text
}

# NEOM coordinates
NEOM_LAT = 28.2933
NEOM_LON = 35.0000
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}   # Mar–May, Sep–Nov

# ════════════════════════════════════════════════════════════════
# 2.  GLOBAL CSS — NEOM DARK THEME
# ════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: #0E1117;
        color: {NEOM["text"]};
        font-family: 'Segoe UI', sans-serif;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NEOM["secondary"]} 0%, {NEOM["accent"]} 100%);
        border-right: 1px solid {NEOM["primary"]}44;
    }}
    [data-testid="stSidebar"] * {{ color: {NEOM["text"]} !important; }}

    /* ── Header Banner ── */
    .neom-header {{
        background: linear-gradient(135deg, {NEOM["secondary"]} 0%, {NEOM["card"]} 60%, {NEOM["accent"]} 100%);
        border: 1px solid {NEOM["primary"]}55;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }}
    .neom-header h1 {{
        color: {NEOM["primary"]};
        font-size: 2.2rem;
        margin: 0 0 4px 0;
        letter-spacing: 1px;
    }}
    .neom-header p {{
        color: {NEOM["muted"]};
        margin: 0;
        font-size: 0.95rem;
    }}

    /* ── KPI Cards ── */
    .kpi-card {{
        background: linear-gradient(135deg, {NEOM["secondary"]}, {NEOM["card"]});
        border: 1px solid {NEOM["primary"]}44;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        height: 100%;
    }}
    .kpi-label {{
        color: {NEOM["muted"]};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {NEOM["primary"]};
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .kpi-sub {{
        color: {NEOM["muted"]};
        font-size: 0.75rem;
        margin-top: 4px;
    }}

    /* ── Alert Box ── */
    .alert-box {{
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
    }}
    .alert-title {{
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 6px;
    }}
    .alert-action {{
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
    }}
    .alert-box ul {{
        margin: 0;
        padding-left: 18px;
    }}
    .alert-box li {{
        font-size: 0.88rem;
        margin-bottom: 4px;
        color: {NEOM["text"]};
    }}

    /* ── Mode Badge ── */
    .mode-badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }}

    /* ── Status pill ── */
    .status-pill {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    /* ── Section headers ── */
    .section-header {{
        color: {NEOM["primary"]};
        font-size: 1.05rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid {NEOM["primary"]}44;
    }}

    /* ── Divider ── */
    hr {{ border-color: {NEOM["primary"]}33 !important; }}

    /* ── Plotly chart containers ── */
    .js-plotly-plot .plotly {{ background: transparent !important; }}

    /* ── Streamlit overrides ── */
    .stSlider > div > div {{ background: {NEOM["primary"]}44; }}
    .stRadio label {{ font-size: 0.9rem !important; }}
    div[data-baseweb="radio"] > label > div:first-child {{
        border-color: {NEOM["primary"]} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 3.  SYNTHETIC DATA GENERATION  (self-contained, no CSV)
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    """Generate 5 000-row synthetic NEOM bird-strike dataset."""
    rng = np.random.default_rng(42)
    n = 5000

    dates = pd.date_range(start="2022-01-01", periods=n, freq="H")
    months = dates.month

    # Temperature: seasonal sinusoidal pattern around NEOM climate
    temp_base = 28 + 14 * np.sin((months - 3) * np.pi / 6)
    temperature = np.clip(
        rng.normal(loc=temp_base, scale=4.5), 8, 52
    )

    # Wind speed: right-skewed (calm most of the time, gusts occasionally)
    wind_speed = np.clip(
        rng.gamma(shape=2, scale=7, size=n), 0, 75
    )

    # Migration season: truth derived from month
    migration_season = np.where(np.isin(months, list(MIGRATION_MONTHS)), 1, 0)
    # Add 5% label noise for realism
    flip_mask = rng.random(n) < 0.05
    migration_season = np.where(flip_mask, 1 - migration_season, migration_season)

    # Risk logic  ─  higher probability when:
    #   • migration active
    #   • calm winds  (birds fly more in calm conditions)
    #   • temperature in sweet-spot 18 – 40 °C
    base_risk_prob = (
        0.55 * migration_season
        + 0.25 * np.where(wind_speed < 20, 1, 0)
        + 0.20 * np.where((temperature >= 18) & (temperature <= 40), 1, 0)
    )
    risk_event = (rng.random(n) < base_risk_prob).astype(int)

    df = pd.DataFrame({
        "Date":             dates,
        "Month":            months,
        "Temperature":      temperature.round(1),
        "Wind_Speed":       wind_speed.round(1),
        "Migration_Season": migration_season,
        "Risk_Event":       risk_event,
    })
    return df


# ════════════════════════════════════════════════════════════════
# 4.  RANDOM FOREST MODEL
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train RandomForestClassifier; return (model, accuracy, importance_df)."""
    features = ["Temperature", "Wind_Speed", "Migration_Season"]
    X = df[features]
    y = df["Risk_Event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    importance_df = (
        pd.DataFrame({"Feature": features, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return clf, acc, importance_df


def predict_risk(model, temperature: float, wind_speed: float, migration: int) -> float:
    X = np.array([[temperature, wind_speed, migration]])
    prob = model.predict_proba(X)[0][1] * 100
    return float(np.clip(prob, 0.0, 100.0))


# ════════════════════════════════════════════════════════════════
# 5.  LIVE WEATHER — OPEN-METEO API  (no key required)
# ════════════════════════════════════════════════════════════════

def fetch_live_weather() -> dict:
    """
    Fetch current temperature & wind speed for NEOM via Open-Meteo.
    Returns dict with keys: temperature, wind_speed, source, error.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={NEOM_LAT}&longitude={NEOM_LON}"
        "&current_weather=true"
        "&wind_speed_unit=kmh"
        "&temperature_unit=celsius"
        "&timezone=Asia%2FRiyadh"
    )
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        cw = resp.json()["current_weather"]
        return {
            "temperature": round(float(cw["temperature"]), 1),
            "wind_speed":  round(float(cw["windspeed"]), 1),
            "source":      "live",
            "error":       None,
        }
    except requests.exceptions.ConnectionError:
        return {"temperature": None, "wind_speed": None,
                "source": "fallback", "error": "No internet connection"}
    except requests.exceptions.Timeout:
        return {"temperature": None, "wind_speed": None,
                "source": "fallback", "error": "API request timed out"}
    except Exception as exc:
        return {"temperature": None, "wind_speed": None,
                "source": "fallback", "error": str(exc)}


def auto_migration_season() -> int:
    """Return 1 if current month is a migration month for NEOM region."""
    return 1 if datetime.now().month in MIGRATION_MONTHS else 0


# ════════════════════════════════════════════════════════════════
# 6.  PLOTLY CHARTS
# ════════════════════════════════════════════════════════════════

def make_gauge(risk_pct: float) -> go.Figure:
    if risk_pct < 30:
        bar_color, status_label = NEOM["success"], "SAFE  /  آمن"
    elif risk_pct < 60:
        bar_color, status_label = NEOM["warning"], "CAUTION  /  حذر"
    else:
        bar_color, status_label = NEOM["danger"], "DANGER  /  خطر"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 40, "color": bar_color}},
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": NEOM["danger"]},
               "decreasing": {"color": NEOM["success"]}},
        title={
            "text": (
                f"<b>Bird Strike Risk</b><br>"
                f"<span style='font-size:0.82em;color:{bar_color}'>{status_label}</span>"
            ),
            "font": {"size": 16, "color": NEOM["text"]},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": NEOM["muted"],
                "tickfont": {"color": NEOM["muted"]},
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(0,255,136,0.08)"},
                {"range": [30, 60], "color": "rgba(255,215,0,0.08)"},
                {"range": [60,100], "color": "rgba(255,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": NEOM["primary"], "width": 3},
                "thickness": 0.75,
                "value": risk_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": NEOM["text"]},
        margin={"t": 80, "b": 20, "l": 30, "r": 30},
        height=320,
    )
    return fig


def make_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(600, random_state=1)
    fig = px.scatter(
        sample,
        x="Temperature",
        y="Wind_Speed",
        color=sample["Risk_Event"].map({0: "No Risk", 1: "Risk"}),
        color_discrete_map={"No Risk": NEOM["success"], "Risk": NEOM["danger"]},
        opacity=0.65,
        title="<b>Historical Analysis — Weather Conditions vs Risk Events</b>",
        labels={"Temperature": "Temperature (°C)",
                "Wind_Speed":  "Wind Speed (km/h)"},
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(
        plot_bgcolor=NEOM["secondary"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": NEOM["text"]},
        title_font={"size": 14, "color": NEOM["primary"]},
        legend={"title": "Risk Status",
                "orientation": "h",
                "yanchor": "bottom", "y": 1.02,
                "xanchor": "right",  "x": 1},
        xaxis={"gridcolor": "rgba(255,255,255,0.094)", "zerolinecolor": "rgba(255,255,255,0.094)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.094)", "zerolinecolor": "rgba(255,255,255,0.094)"},
        margin={"t": 60, "b": 40, "l": 40, "r": 20},
        height=380,
    )
    return fig


def make_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    colors = [NEOM["primary"], NEOM["warning"], NEOM["danger"]][: len(importance_df)]
    fig = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker={"color": colors, "line": {"color": "rgba(255,255,255,0.133)", "width": 1}},
        text=importance_df["Importance"].map("{:.1%}".format),
        textposition="outside",
        textfont={"color": NEOM["text"]},
    ))
    fig.update_layout(
        title="<b>Model Feature Importance</b>",
        title_font={"size": 14, "color": NEOM["primary"]},
        plot_bgcolor=NEOM["secondary"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": NEOM["text"]},
        xaxis={"gridcolor": "rgba(255,255,255,0.094)", "range": [0, 0.8],
               "tickformat": ".0%"},
        yaxis={"autorange": "reversed"},
        margin={"t": 50, "b": 20, "l": 10, "r": 60},
        height=200,
    )
    return fig


def make_monthly_risk(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.groupby("Month")["Risk_Event"]
        .mean()
        .reset_index()
        .rename(columns={"Risk_Event": "Risk_Rate"})
    )
    monthly["Month_Name"] = monthly["Month"].map({
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
    })
    bar_colors = [
        NEOM["danger"] if m in MIGRATION_MONTHS else NEOM["primary"]
        for m in monthly["Month"]
    ]
    fig = go.Figure(go.Bar(
        x=monthly["Month_Name"],
        y=(monthly["Risk_Rate"] * 100).round(1),
        marker={"color": bar_colors},
        text=(monthly["Risk_Rate"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont={"color": NEOM["text"], "size": 10},
    ))
    fig.update_layout(
        title="<b>Monthly Average Risk Rate  (🔴 Migration Months)</b>",
        title_font={"size": 14, "color": NEOM["primary"]},
        plot_bgcolor=NEOM["secondary"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": NEOM["text"]},
        xaxis={"gridcolor": "rgba(255,255,255,0.094)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.094)", "title": "Risk %"},
        margin={"t": 50, "b": 30, "l": 40, "r": 20},
        height=280,
    )
    return fig


# ════════════════════════════════════════════════════════════════
# 7.  RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════

def get_recommendation(risk_pct: float) -> dict:
    if risk_pct >= 60:
        return {
            "icon":    "🚨",
            "level":   "CRITICAL ALERT",
            "action":  "HALT ALL FLIGHT OPERATIONS",
            "color":   NEOM["danger"],
            "steps":   [
                "Immediately suspend all takeoffs and landings",
                "Activate full acoustic & visual deterrent systems",
                "Issue NOTAM to all inbound/outbound aircraft",
                "Deploy runway inspection & bird dispersal teams",
                "Alert NEOM Air Traffic Control Centre",
            ],
        }
    elif risk_pct >= 30:
        return {
            "icon":    "⚠️",
            "level":   "CAUTION — ELEVATED RISK",
            "action":  "ENHANCED MONITORING PROTOCOL",
            "color":   NEOM["warning"],
            "steps":   [
                "Notify all pilots of elevated bird activity",
                "Increase visual scanning frequency (every 10 min)",
                "Activate perimeter radar bird-tracking mode",
                "Review and update weather advisories",
                "Standby dispersal team on alert",
            ],
        }
    else:
        return {
            "icon":    "✅",
            "level":   "ALL CLEAR",
            "action":  "NORMAL OPERATIONS",
            "color":   NEOM["success"],
            "steps":   [
                "Continue standard pre-flight bird-risk assessment",
                "Log routine environmental check",
                "No additional deterrent measures required",
            ],
        }


# ════════════════════════════════════════════════════════════════
# 8.  SIDEBAR
# ════════════════════════════════════════════════════════════════

def build_sidebar():
    st.sidebar.markdown(
        f"""
        <div style='text-align:center;padding:8px 0 16px 0;'>
            <span style='font-size:2.2rem;'>🦅</span><br>
            <span style='color:{NEOM["primary"]};font-size:1.05rem;font-weight:700;
                         letter-spacing:1px;'>NEOM BIO-SECURE</span><br>
            <span style='color:{NEOM["muted"]};font-size:0.72rem;'>v5.0 · Aviation Safety AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "**Operating Mode**",
        ["🔴  Live Real-Time Mode", "🛠️  Simulation Mode"],
        index=0,
    )
    st.sidebar.divider()

    if mode == "🔴  Live Real-Time Mode":
        st.sidebar.markdown(
            f"<div class='section-header'>📡 NEOM Station</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            f"<small style='color:{NEOM['muted']};'>"
            f"Lat: {NEOM_LAT} &nbsp;|&nbsp; Lon: {NEOM_LON}<br>"
            f"Source: open-meteo.com (free API)</small>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
        return mode, None, None, None

    else:  # Simulation Mode
        st.sidebar.markdown(
            f"<div class='section-header'>🎛️ Manual Controls</div>",
            unsafe_allow_html=True,
        )
        temp = st.sidebar.slider("🌡️ Temperature (°C)", 5, 52, 30, step=1)
        wind = st.sidebar.slider("💨 Wind Speed (km/h)", 0, 75, 15, step=1)
        mig_choice = st.sidebar.selectbox(
            "🦢 Migration Season",
            ["Auto (from current month)", "Active — Yes", "Inactive — No"],
        )
        if mig_choice == "Auto (from current month)":
            mig = auto_migration_season()
        elif mig_choice == "Active — Yes":
            mig = 1
        else:
            mig = 0

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"<small style='color:{NEOM['muted']};'>"
            f"Adjust sliders to simulate extreme weather scenarios "
            f"and evaluate model responses.</small>",
            unsafe_allow_html=True,
        )
        return mode, temp, wind, mig


# ════════════════════════════════════════════════════════════════
# 9.  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Header ──────────────────────────────────────────────────
    now_str = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(
        f"""
        <div class='neom-header'>
            <h1>🦅 NEOM Bio-Secure</h1>
            <p>AI-Powered Bird Strike Risk Prediction Platform &nbsp;·&nbsp;
               Aviation Safety Division &nbsp;·&nbsp;
               <span style='color:{NEOM["primary"]};'>v5.0</span>
               &nbsp;·&nbsp;
               <span style='color:{NEOM["muted"]};font-size:0.85em;'>{now_str}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data & model (cached) ───────────────────────────────
    with st.spinner("Initialising AI Model — please wait…"):
        df = generate_training_data()
        model, accuracy, importance_df = train_model(df)

    # ── Sidebar ──────────────────────────────────────────────────
    mode, sim_temp, sim_wind, sim_mig = build_sidebar()

    # ── Resolve inputs based on mode ────────────────────────────
    is_live = mode.startswith("🔴")

    if is_live:
        with st.spinner("Fetching live weather from Open-Meteo API…"):
            weather = fetch_live_weather()

        if weather["error"]:
            st.warning(
                f"⚠️ Live API unavailable: **{weather['error']}** — "
                "falling back to last-known defaults (T=30 °C, W=12 km/h).",
                icon="📡",
            )
            temperature  = 30.0
            wind_speed   = 12.0
            api_ok       = False
        else:
            temperature  = weather["temperature"]
            wind_speed   = weather["wind_speed"]
            api_ok       = True

        migration = auto_migration_season()
        current_month = datetime.now().month
        month_name = datetime.now().strftime("%B")

    else:
        temperature  = float(sim_temp)
        wind_speed   = float(sim_wind)
        migration    = sim_mig
        api_ok       = None   # N/A in simulation

    # ── Prediction ───────────────────────────────────────────────
    risk_pct = predict_risk(model, temperature, wind_speed, migration)
    rec      = get_recommendation(risk_pct)

    # ── Mode badge ───────────────────────────────────────────────
    if is_live:
        badge_color = NEOM["danger"] if api_ok else NEOM["warning"]
        badge_text  = "● LIVE — Open-Meteo API" if api_ok else "● LIVE (fallback defaults)"
    else:
        badge_color = NEOM["primary"]
        badge_text  = "⚙ SIMULATION MODE"

    st.markdown(
        f"<span class='mode-badge' style='background:{badge_color}22;"
        f"color:{badge_color};border:1px solid {badge_color}55;'>"
        f"{badge_text}</span>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════
    # ROW 1 — KPI Cards
    # ════════════════════════════════════════════════════════════
    mig_label = "Active 🦢" if migration == 1 else "Inactive"
    mig_color = NEOM["warning"] if migration == 1 else NEOM["success"]

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, sub in [
        (k1, "Temperature",      f"{temperature} °C",  "NEOM Station"),
        (k2, "Wind Speed",       f"{wind_speed} km/h", "Surface Level"),
        (k3, "Migration Season", mig_label,            "Current Status"),
        (k4, "Risk Probability", f"{risk_pct:.1f}%",   rec["level"]),
    ]:
        val_color = NEOM["primary"] if label != "Migration Season" else mig_color
        risk_color = rec["color"] if label == "Risk Probability" else val_color
        with col:
            st.markdown(
                f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{risk_color};'>{value}</div>
                    <div class='kpi-sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # ROW 2 — Gauge + Alert  |  Scatter Plot
    # ════════════════════════════════════════════════════════════
    col_left, col_right = st.columns([1, 1.7], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Risk Gauge</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(risk_pct), use_container_width=True)

        # Alert box
        st.markdown(
            f"""
            <div class='alert-box' style='background:{rec["color"]}15;
                 border:1px solid {rec["color"]}55;'>
                <div class='alert-title' style='color:{rec["color"]};'>
                    {rec["icon"]} {rec["level"]}
                </div>
                <div class='alert-action' style='color:{rec["color"]};'>
                    {rec["action"]}
                </div>
                <hr style='border-color:{rec["color"]}44;margin:10px 0;'>
                <ul>
                    {''.join(f"<li>{s}</li>" for s in rec["steps"])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("<div class='section-header'>Historical Weather vs Risk</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_scatter(df), use_container_width=True)

        # Live mode: display migration auto-detection info
        if is_live:
            st.markdown(
                f"""
                <div style='background:{NEOM["accent"]};border:1px solid {NEOM["primary"]}44;
                     border-radius:8px;padding:12px 16px;font-size:0.85rem;'>
                    📅 &nbsp;<b>Auto Migration Detection:</b> &nbsp;
                    Current month is <b>{month_name}</b>
                    ({current_month}) &nbsp;→&nbsp;
                    <span style='color:{mig_color};font-weight:700;'>{mig_label}</span>
                    &nbsp;(migration months: Mar–May &amp; Sep–Nov)
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # ROW 3 — Monthly Risk Trend  |  Feature Importance
    # ════════════════════════════════════════════════════════════
    col_a, col_b = st.columns([1.6, 1], gap="large")

    with col_a:
        st.markdown("<div class='section-header'>Monthly Risk Trend</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_monthly_risk(df), use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>Model Diagnostics</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_feature_importance(importance_df),
                        use_container_width=True)

        # Model metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>Model Accuracy</div>"
                f"<div class='kpi-value' style='font-size:1.4rem;'>{accuracy:.1%}</div>"
                f"<div class='kpi-sub'>Random Forest</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>Training Samples</div>"
                f"<div class='kpi-value' style='font-size:1.4rem;'>{len(df):,}</div>"
                f"<div class='kpi-sub'>Synthetic Data</div></div>",
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        f"""
        <div style='text-align:center;color:{NEOM["muted"]};font-size:0.78rem;
             padding:8px 0 16px 0;'>
            🦅 &nbsp; <b style='color:{NEOM["primary"]};'>NEOM Bio-Secure v5.0</b>
            &nbsp;·&nbsp; AI Aviation Safety Platform
            &nbsp;·&nbsp; NEOM Smart City &nbsp;·&nbsp; 2025
            <br><br>
            <span style='font-size:0.7rem;'>
                Powered by RandomForest · Open-Meteo Weather API ·
                Streamlit · Plotly
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
