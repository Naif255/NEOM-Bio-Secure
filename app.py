"""
╔══════════════════════════════════════════════════════════════════╗
║       NEOM Bio-Secure  —  Version 6.0  (Enterprise Edition)     ║
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
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timezone, timedelta

# ════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG & BRANDING
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEOM Bio-Secure v6.0",
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
    "dawn":       "#FF8C42",   # Dawn / Dusk Orange
    "text":       "#FFFFFF",   # White
    "muted":      "#A0AEC0",   # Muted text
}

# NEOM coordinates & timezone
NEOM_LAT = 28.2933
NEOM_LON  = 35.0000
NEOM_TZ   = timezone(timedelta(hours=3))   # Saudi Arabia Time: UTC+3
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
        margin-bottom: 6px;
        margin-right: 6px;
    }}

    /* ── Circadian Badge ── */
    .circadian-badge {{
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
        margin-right: 6px;
    }}

    /* ── Multiplier info box ── */
    .multiplier-box {{
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
        font-size: 0.85rem;
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
# 3.  CIRCADIAN RHYTHM INTELLIGENCE  (Biological Logic)
# ════════════════════════════════════════════════════════════════

def get_neom_hour() -> int:
    """Return current hour (0–23) in NEOM local time (Saudi Arabia: UTC+3)."""
    return datetime.now(tz=NEOM_TZ).hour


def get_circadian_info() -> dict:
    """
    Determine bird-activity period based on NEOM local time.

    Periods & risk multipliers:
    • Dawn  05:00–06:59  → 1.4x  (Peak — birds most active)
    • Dusk  17:00–18:59  → 1.4x  (Peak — birds most active)
    • Day   07:00–16:59  → 1.0x  (Normal activity)
    • Night 19:00–04:59  → 0.8x  (Reduced activity)
    """
    hour = get_neom_hour()

    if 5 <= hour < 7:
        return {
            "period":     "DAWN",
            "badge":      "🌅 DAWN — HIGH BIRD ACTIVITY",
            "multiplier": 1.4,
            "color":      NEOM["dawn"],
            "desc":       f"Peak activity window · Hour {hour:02d}:xx SAT",
        }
    elif 17 <= hour < 19:
        return {
            "period":     "DUSK",
            "badge":      "🌆 DUSK — HIGH BIRD ACTIVITY",
            "multiplier": 1.4,
            "color":      NEOM["dawn"],
            "desc":       f"Peak activity window · Hour {hour:02d}:xx SAT",
        }
    elif 7 <= hour < 17:
        return {
            "period":     "DAY",
            "badge":      "☀️ DAYTIME — NORMAL ACTIVITY",
            "multiplier": 1.0,
            "color":      NEOM["warning"],
            "desc":       f"Standard activity levels · Hour {hour:02d}:xx SAT",
        }
    else:
        return {
            "period":     "NIGHT",
            "badge":      "🌙 NIGHT — REDUCED ACTIVITY",
            "multiplier": 0.8,
            "color":      NEOM["muted"],
            "desc":       f"Low activity window · Hour {hour:02d}:xx SAT",
        }


# ════════════════════════════════════════════════════════════════
# 4.  SYNTHETIC DATA GENERATION  (self-contained, no CSV)
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    """Generate 5 000-row synthetic NEOM bird-strike dataset."""
    rng = np.random.default_rng(42)
    n = 5000

    dates = pd.date_range(start="2022-01-01", periods=n, freq="H")
    months = dates.month

    temp_base = 28 + 14 * np.sin((months - 3) * np.pi / 6)
    temperature = np.clip(rng.normal(loc=temp_base, scale=4.5), 8, 52)

    wind_speed = np.clip(rng.gamma(shape=2, scale=7, size=n), 0, 75)

    migration_season = np.where(np.isin(months, list(MIGRATION_MONTHS)), 1, 0)
    flip_mask = rng.random(n) < 0.05
    migration_season = np.where(flip_mask, 1 - migration_season, migration_season)

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
# 5.  RANDOM FOREST MODEL
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

    acc = accuracy_score(y_test, clf.predict(X_test))

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
# 6.  LIVE WEATHER — OPEN-METEO API  (Smart Caching — 15 min TTL)
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_neom_weather() -> dict:
    """
    Fetch current temperature & wind speed for NEOM via Open-Meteo.
    Cached for 15 minutes (TTL=900s) to prevent 429 Too Many Requests.
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
    """Return 1 if current month is a migration month for the NEOM region."""
    return 1 if datetime.now().month in MIGRATION_MONTHS else 0


# ════════════════════════════════════════════════════════════════
# 7.  PLOTLY CHARTS
# ════════════════════════════════════════════════════════════════

def make_gauge(risk_pct: float, multiplier: float = 1.0) -> go.Figure:
    """Render the risk gauge. Color and label react to the circadian multiplier."""
    if risk_pct < 30:
        bar_color, status_label = NEOM["success"], "SAFE  /  آمن"
    elif risk_pct < 60:
        bar_color, status_label = NEOM["warning"], "CAUTION  /  حذر"
    else:
        bar_color, status_label = NEOM["danger"], "DANGER  /  خطر"

    multiplier_tag = ""
    if multiplier > 1.0:
        multiplier_tag = f"<br><span style='font-size:0.72em;color:{NEOM['dawn']};'>⚡ {multiplier}x Circadian Boost</span>"
    elif multiplier < 1.0:
        multiplier_tag = f"<br><span style='font-size:0.72em;color:{NEOM['muted']};'>🌙 {multiplier}x Night Reduction</span>"

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
                f"{multiplier_tag}"
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
        margin={"t": 100, "b": 20, "l": 30, "r": 30},
        height=340,
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
        labels={"Temperature": "Temperature (°C)", "Wind_Speed": "Wind Speed (km/h)"},
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(
        plot_bgcolor=NEOM["secondary"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": NEOM["text"]},
        title_font={"size": 14, "color": NEOM["primary"]},
        legend={"title": "Risk Status", "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
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
# 8.  RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════

def get_recommendation(risk_pct: float) -> dict:
    if risk_pct >= 60:
        return {
            "icon":   "🚨",
            "level":  "CRITICAL ALERT",
            "action": "HALT ALL FLIGHT OPERATIONS",
            "color":  NEOM["danger"],
            "steps":  [
                "Immediately suspend all takeoffs and landings",
                "Activate full acoustic & visual deterrent systems",
                "Issue NOTAM to all inbound/outbound aircraft",
                "Deploy runway inspection & bird dispersal teams",
                "Alert NEOM Air Traffic Control Centre",
            ],
        }
    elif risk_pct >= 30:
        return {
            "icon":   "⚠️",
            "level":  "CAUTION — ELEVATED RISK",
            "action": "ENHANCED MONITORING PROTOCOL",
            "color":  NEOM["warning"],
            "steps":  [
                "Notify all pilots of elevated bird activity",
                "Increase visual scanning frequency (every 10 min)",
                "Activate perimeter radar bird-tracking mode",
                "Review and update weather advisories",
                "Standby dispersal team on alert",
            ],
        }
    else:
        return {
            "icon":   "✅",
            "level":  "ALL CLEAR",
            "action": "NORMAL OPERATIONS",
            "color":  NEOM["success"],
            "steps":  [
                "Continue standard pre-flight bird-risk assessment",
                "Log routine environmental check",
                "No additional deterrent measures required",
            ],
        }


# ════════════════════════════════════════════════════════════════
# 9.  SIDEBAR
# ════════════════════════════════════════════════════════════════

def build_sidebar():
    st.sidebar.markdown(
        f"""
        <div style='text-align:center;padding:8px 0 16px 0;'>
            <span style='font-size:2.2rem;'>🦅</span><br>
            <span style='color:{NEOM["primary"]};font-size:1.05rem;font-weight:700;
                         letter-spacing:1px;'>NEOM BIO-SECURE</span><br>
            <span style='color:{NEOM["muted"]};font-size:0.72rem;'>
                v6.0 Enterprise · Aviation Safety AI
            </span>
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
            f"Source: open-meteo.com (free API)<br>"
            f"Cache TTL: 15 min (anti-rate-limit)</small>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
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
# 10.  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Circadian state (computed once per run) ──────────────────
    circadian = get_circadian_info()

    # ── Header ──────────────────────────────────────────────────
    neom_now = datetime.now(tz=NEOM_TZ)
    now_str  = neom_now.strftime("%d %b %Y  %H:%M SAT (UTC+3)")
    st.markdown(
        f"""
        <div class='neom-header'>
            <h1>🦅 NEOM Bio-Secure</h1>
            <p>AI-Powered Bird Strike Risk Prediction Platform &nbsp;·&nbsp;
               Aviation Safety Division &nbsp;·&nbsp;
               <span style='color:{NEOM["primary"]};'>v6.0 Enterprise</span>
               &nbsp;·&nbsp;
               <span style='color:{NEOM["muted"]};font-size:0.85em;'>{now_str}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data & model (cached) ───────────────────────────────
    with st.spinner("Initialising AI Model — please wait…"):
        df    = generate_training_data()
        model, accuracy, importance_df = train_model(df)

    # ── Sidebar ──────────────────────────────────────────────────
    mode, sim_temp, sim_wind, sim_mig = build_sidebar()

    # ── Resolve inputs based on mode ────────────────────────────
    is_live = mode.startswith("🔴")

    if is_live:
        with st.spinner("Fetching live weather (cached 15 min)…"):
            weather = get_live_neom_weather()

        if weather["error"]:
            st.warning(
                f"⚠️ Live API unavailable: **{weather['error']}** — "
                "falling back to last-known defaults (T=30 °C, W=12 km/h).",
                icon="📡",
            )
            temperature = 30.0
            wind_speed  = 12.0
            api_ok      = False
        else:
            temperature = weather["temperature"]
            wind_speed  = weather["wind_speed"]
            api_ok      = True

        migration    = auto_migration_season()
        current_month = datetime.now().month
        month_name    = datetime.now().strftime("%B")

    else:
        temperature   = float(sim_temp)
        wind_speed    = float(sim_wind)
        migration     = sim_mig
        api_ok        = None   # N/A in simulation

    # ── Base prediction ──────────────────────────────────────────
    base_risk = predict_risk(model, temperature, wind_speed, migration)

    # ── Apply Circadian Rhythm Multiplier ────────────────────────
    risk_pct = float(np.clip(base_risk * circadian["multiplier"], 0.0, 100.0))

    rec = get_recommendation(risk_pct)

    # ── Mode badge + Circadian badge ─────────────────────────────
    if is_live:
        badge_color = NEOM["danger"] if api_ok else NEOM["warning"]
        badge_text  = "● LIVE — Open-Meteo API" if api_ok else "● LIVE (fallback defaults)"
    else:
        badge_color = NEOM["primary"]
        badge_text  = "⚙ SIMULATION MODE"

    st.markdown(
        f"<span class='mode-badge' style='background:{badge_color}22;"
        f"color:{badge_color};border:1px solid {badge_color}55;'>"
        f"{badge_text}</span>"
        f"<span class='circadian-badge' style='background:{circadian['color']}22;"
        f"color:{circadian['color']};border:1px solid {circadian['color']}55;'>"
        f"{circadian['badge']}</span>",
        unsafe_allow_html=True,
    )

    # Circadian multiplier info bar
    mult = circadian["multiplier"]
    if mult != 1.0:
        adj_type  = "elevated" if mult > 1.0 else "reduced"
        mult_sign = f"+{int((mult - 1.0)*100)}%" if mult > 1.0 else f"{int((mult - 1.0)*100)}%"
        st.markdown(
            f"""
            <div class='multiplier-box' style='background:{circadian["color"]}18;
                 border:1px solid {circadian["color"]}44;color:{circadian["color"]};'>
                🧬 &nbsp;<b>Circadian Rhythm Active:</b> &nbsp;
                {circadian["desc"]} &nbsp;→&nbsp;
                Risk score {adj_type} by <b>{mult_sign}</b>
                (base: {base_risk:.1f}% → adjusted: <b>{risk_pct:.1f}%</b>)
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════
    # ROW 1 — KPI Cards  (5 cards — adds Circadian Period)
    # ════════════════════════════════════════════════════════════
    mig_label = "Active 🦢" if migration == 1 else "Inactive"
    mig_color = NEOM["warning"] if migration == 1 else NEOM["success"]

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, "Temperature",      f"{temperature} °C",          "NEOM Station",   NEOM["primary"]),
        (k2, "Wind Speed",       f"{wind_speed} km/h",         "Surface Level",  NEOM["primary"]),
        (k3, "Migration Season", mig_label,                    "Current Status", mig_color),
        (k4, "Circadian Period", circadian["period"],          "NEOM Local Time",circadian["color"]),
        (k5, "Risk Probability", f"{risk_pct:.1f}%",           rec["level"],     rec["color"]),
    ]
    for col, label, value, sub, val_color in kpi_data:
        with col:
            st.markdown(
                f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{val_color};font-size:1.5rem;'>{value}</div>
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
        st.plotly_chart(
            make_gauge(risk_pct, multiplier=circadian["multiplier"]),
            use_container_width=True,
        )

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

        # Live mode: migration info + API cache notice
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
                    &nbsp;|&nbsp;
                    <span style='color:{NEOM["primary"]};'>🛡️ API cache: 15 min</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # ROW 3 — NEOM Command Center Map
    # ════════════════════════════════════════════════════════════
    st.markdown(
        "<div class='section-header'>🗺️ NEOM Command Center — Geospatial View</div>",
        unsafe_allow_html=True,
    )

    map_col, info_col = st.columns([2, 1], gap="large")

    with map_col:
        # Build map dataframe with NEOM airspace zone markers
        map_df = pd.DataFrame({
            "lat":   [NEOM_LAT, NEOM_LAT + 0.18, NEOM_LAT - 0.14, NEOM_LAT + 0.05],
            "lon":   [NEOM_LON, NEOM_LON - 0.22, NEOM_LON + 0.19, NEOM_LON - 0.08],
            "label": ["NEOM HQ", "North Airstrip", "South Corridor", "Bio-Sensor Array"],
        })
        st.map(
            map_df,
            latitude="lat",
            longitude="lon",
            zoom=10,
            use_container_width=True,
        )

    with info_col:
        st.markdown(
            f"""
            <div class='kpi-card' style='text-align:left;padding:20px 22px;'>
                <div class='kpi-label' style='margin-bottom:14px;'>
                    📍 NEOM Airspace Intelligence
                </div>
                <div style='font-size:0.88rem;line-height:1.8;'>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Location:</span>
                    NEOM Smart City, Tabuk Province<br>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Lat / Lon:</span>
                    {NEOM_LAT}° N &nbsp;/ {NEOM_LON}° E<br>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Timezone:</span>
                    Asia/Riyadh (UTC+3)<br>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Local Time:</span>
                    {neom_now.strftime("%H:%M:%S")}<br>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Circadian:</span>
                    <span style='color:{circadian["color"]};'>{circadian["period"]}</span>
                    (×{circadian["multiplier"]})<br>
                    <span style='color:{NEOM["primary"]};font-weight:600;'>Risk Zone:</span>
                    <span style='color:{rec["color"]};font-weight:700;'>{rec["level"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        # Circadian period breakdown reference
        st.markdown(
            f"""
            <div style='background:{NEOM["accent"]};border:1px solid {NEOM["primary"]}33;
                 border-radius:10px;padding:14px 16px;font-size:0.8rem;'>
                <div style='color:{NEOM["primary"]};font-weight:700;margin-bottom:8px;'>
                    🧬 Circadian Rhythm Schedule
                </div>
                <div style='line-height:2.0;'>
                    <span style='color:{NEOM["dawn"]};'>🌅 Dawn 05–07h</span> → ×1.4 HIGH<br>
                    <span style='color:{NEOM["warning"]};'>☀️ Day 07–17h</span> → ×1.0 NORMAL<br>
                    <span style='color:{NEOM["dawn"]};'>🌆 Dusk 17–19h</span> → ×1.4 HIGH<br>
                    <span style='color:{NEOM["muted"]};'>🌙 Night 19–05h</span> → ×0.8 LOW
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # ROW 4 — Monthly Risk Trend  |  Feature Importance
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
            🦅 &nbsp; <b style='color:{NEOM["primary"]};'>NEOM Bio-Secure v6.0 Enterprise</b>
            &nbsp;·&nbsp; AI Aviation Safety Platform
            &nbsp;·&nbsp; NEOM Smart City &nbsp;·&nbsp; 2025–2026
            <br><br>
            <span style='font-size:0.7rem;'>
                Powered by RandomForest · Open-Meteo (cached 15 min) ·
                Circadian Rhythm Engine · Streamlit · Plotly
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
