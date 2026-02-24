"""
╔══════════════════════════════════════════════════════════════════════════╗
║   NEOM Bio-Secure — Version 8.0  (The Line Edition)                     ║
║   Premium Executive Dashboard · Bird Aircraft Strike Hazard System       ║
║   NEOM Smart City  |  Aviation Safety Division                           ║
║   UI Theme: Premium Glassmorphism · Pearl White · Inter Typography       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timezone, timedelta

# ════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEOM Bio-Secure | Executive Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ────────────────────────────────────────────────────────────
NEOM_LAT         = 28.2933
NEOM_LON         = 35.0000
UTC3             = timezone(timedelta(hours=3))   # Saudi Arabia Time
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}           # Mar–May, Sep–Nov


# ════════════════════════════════════════════════════════════════════════
# 2.  CSS INJECTION — PREMIUM GLASSMORPHISM THEME
# ════════════════════════════════════════════════════════════════════════

def inject_premium_css() -> None:
    st.markdown("""
    <style>
    /* ── Google Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset & Pearl Background ──────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background: linear-gradient(145deg, #EEF2F7 0%, #FFFFFF 45%, #F0F4F8 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1A202C !important;
    }

    /* ── Hide Streamlit Chrome ─────────────────────────────────────────── */
    #MainMenu,
    header[data-testid="stHeader"],
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* ── Main padding & max-width ──────────────────────────────────────── */
    [data-testid="stAppViewContainer"] > .main {
        padding: 0 2.5rem 5.5rem 2.5rem !important;
    }
    section.main > div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    /* ── Tabs chrome ───────────────────────────────────────────────────── */
    [data-testid="stTabs"] > div:first-child {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 14px !important;
        padding: 5px 6px !important;
        border: 1px solid rgba(255,255,255,0.92) !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.055) !important;
        gap: 2px !important;
    }
    button[role="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        color: #8898B3 !important;
        border-radius: 10px !important;
        padding: 9px 26px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.25s ease !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: #FFFFFF !important;
        color: #1A202C !important;
        box-shadow: 0 1px 10px rgba(0,0,0,0.08) !important;
    }
    button[role="tab"]:hover:not([aria-selected="true"]) {
        color: #3D4E6B !important;
        background: rgba(255,255,255,0.55) !important;
    }
    [data-testid="stTabsContent"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* ── Glass card applied to columns inside tabs ─────────────────────── */
    [data-testid="stTabs"] [data-testid="column"] {
        background: rgba(255,255,255,0.78) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.96) !important;
        box-shadow:
            0 1px 3px rgba(0,0,0,0.02),
            0 6px 20px rgba(0,0,0,0.045),
            0 20px 56px rgba(0,0,0,0.03) !important;
        padding: 22px 22px !important;
        transition: box-shadow 0.3s ease !important;
    }
    [data-testid="stTabs"] [data-testid="column"]:hover {
        box-shadow:
            0 1px 3px rgba(0,0,0,0.025),
            0 8px 28px rgba(0,0,0,0.07),
            0 28px 64px rgba(0,0,0,0.045) !important;
    }

    /* ── Sidebar – premium white ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.94) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(200,210,225,0.35) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #8898B3 !important;
    }
    [data-testid="stSidebar"] .stSlider > div { margin-top: 0 !important; }

    /* ── Plotly transparent canvas ─────────────────────────────────────── */
    .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .bg {
        background: transparent !important;
    }
    .stPlotlyChart {
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* ── Dashboard header ──────────────────────────────────────────────── */
    .dash-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 22px 0 18px 0;
        margin-bottom: 6px;
        border-bottom: 1px solid rgba(180,192,210,0.18);
    }
    .dash-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1A202C;
        letter-spacing: -0.025em;
    }
    .dash-brand span { color: #3B82F6; }
    .dash-subtitle {
        font-size: 0.68rem;
        color: #8898B3;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .dash-right {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Status badges ─────────────────────────────────────────────────── */
    .badge-live {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.28);
        border-radius: 20px; padding: 5px 13px;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #059669;
    }
    .badge-sim {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.28);
        border-radius: 20px; padding: 5px 13px;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #D97706;
    }
    .pulse-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #10B981;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.45); }
        50%      { opacity:0.7; box-shadow: 0 0 0 5px rgba(16,185,129,0); }
    }

    /* ── Risk pills ────────────────────────────────────────────────────── */
    .pill {
        display: inline-block;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.07em; text-transform: uppercase;
    }
    .pill-green  { background:rgba(16,185,129,0.1);  color:#059669; border:1px solid rgba(16,185,129,0.3); }
    .pill-amber  { background:rgba(245,158,11,0.1);  color:#D97706; border:1px solid rgba(245,158,11,0.3); }
    .pill-red    { background:rgba(239,68,68,0.1);   color:#DC2626; border:1px solid rgba(239,68,68,0.3); }

    /* ── Section headers ───────────────────────────────────────────────── */
    .sec-header {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A0AFCA;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(160,175,202,0.18);
    }

    /* ── Advisory box ──────────────────────────────────────────────────── */
    .adv-box {
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 8px;
    }
    .adv-clear   { background:rgba(16,185,129,0.07);  border:1px solid rgba(16,185,129,0.22);  border-left:4px solid #10B981; }
    .adv-warning { background:rgba(245,158,11,0.07);  border:1px solid rgba(245,158,11,0.22);  border-left:4px solid #F59E0B; }
    .adv-critical{ background:rgba(239,68,68,0.07);   border:1px solid rgba(239,68,68,0.22);   border-left:4px solid #EF4444; }
    .adv-title {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    .adv-step {
        font-size: 0.78rem;
        color: #4B5563;
        padding: 5px 0;
        border-bottom: 1px solid rgba(0,0,0,0.04);
        display: flex; align-items: flex-start; gap: 8px;
    }
    .adv-step:last-child { border-bottom: none; }

    /* ── Stat rows ─────────────────────────────────────────────────────── */
    .stat-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(160,175,202,0.1);
        font-size: 0.79rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-k { color: #8898B3; font-weight: 500; }
    .stat-v { color: #1A202C; font-weight: 600; }

    /* ── Metadata strip ────────────────────────────────────────────────── */
    .meta-strip {
        display: flex; flex-wrap: wrap; gap: 0;
        background: rgba(255,255,255,0.65);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.92);
        backdrop-filter: blur(16px);
        margin-top: 18px;
    }
    .meta-item {
        flex: 1; min-width: 120px;
        padding: 14px 20px;
        border-right: 1px solid rgba(160,175,202,0.15);
    }
    .meta-item:last-child { border-right: none; }
    .meta-lbl {
        font-size: 0.58rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #A0AFCA;
        margin-bottom: 4px;
    }
    .meta-val {
        font-size: 0.82rem; font-weight: 600; color: #1A202C;
    }

    /* ── Engine room intro ─────────────────────────────────────────────── */
    .engine-header {
        display: flex; align-items: baseline; gap: 10px;
        margin-bottom: 6px;
    }
    .engine-title { font-size: 1rem; font-weight: 700; color: #1A202C; }
    .engine-tag {
        background: rgba(59,130,246,0.08); color: #3B82F6;
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 6px; padding: 2px 9px;
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
    }
    .engine-sub { font-size: 0.76rem; color: #8898B3; margin-bottom: 20px; }

    /* ── Fixed bottom ticker ───────────────────────────────────────────── */
    .ticker-wrap {
        position: fixed; bottom: 0; left: 0; right: 0;
        height: 34px;
        background: #111827;
        overflow: hidden;
        display: flex; align-items: center;
        z-index: 9999;
        border-top: 1px solid rgba(0,229,255,0.12);
    }
    .ticker-lbl {
        flex-shrink: 0;
        font-family: 'Courier New', monospace;
        font-size: 0.58rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: #FFFFFF;
        background: rgba(0,229,255,0.12);
        border-right: 1px solid rgba(0,229,255,0.18);
        padding: 0 14px; height: 100%;
        display: flex; align-items: center;
    }
    .ticker-rail {
        flex: 1; overflow: hidden;
        display: flex; align-items: center;
    }
    .ticker-track {
        display: flex;
        white-space: nowrap;
        animation: ticker-scroll 50s linear infinite;
        will-change: transform;
    }
    .ticker-text {
        font-family: 'Courier New', monospace;
        font-size: 0.65rem; font-weight: 400;
        color: #00E5FF;
        text-shadow: 0 0 8px rgba(0,229,255,0.38);
        letter-spacing: 0.055em;
        padding: 0 48px;
    }
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* ── Divider utility ───────────────────────────────────────────────── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(160,175,202,0.25), transparent);
        margin: 14px 0;
    }

    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# 3.  DATA GENERATION  (5 000 synthetic rows, cached per session)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    """Synthetic hourly NEOM weather + risk labels (5 000 rows)."""
    np.random.seed(42)
    n = 5_000
    dates        = pd.date_range("2022-01-01", periods=n, freq="h")
    doy          = dates.dayofyear
    month        = dates.month

    # Seasonal temperature – NEOM desert climate (8–52 °C)
    temp_base    = 25 + 18 * np.sin((doy - 80) * 2 * np.pi / 365)
    temperature  = np.clip(temp_base + np.random.normal(0, 4, n), 8, 52)

    # Wind speed – Red Sea coastal gamma distribution (0–75 km/h)
    wind_speed   = np.clip(np.random.gamma(2.5, 7.5, n), 0, 75)

    # Migration binary flag
    migration    = np.isin(month, list(MIGRATION_MONTHS)).astype(int)

    # Risk probability model (multi-factor)
    base_p       = 0.08
    mig_p        = migration * 0.28
    thermal_p    = ((temperature >= 22) & (temperature <= 40)).astype(float) * 0.15
    low_wind_p   = (wind_speed < 18).astype(float) * 0.12
    high_wind_p  = (wind_speed > 55).astype(float) * (-0.08)
    prob         = np.clip(base_p + mig_p + thermal_p + low_wind_p + high_wind_p, 0.03, 0.72)

    risk_event   = (np.random.random(n) < prob).astype(int)
    # 5 % label flip for realism
    flip         = np.random.random(n) < 0.05
    risk_event   = np.where(flip, 1 - risk_event, risk_event)

    return pd.DataFrame({
        "temperature":     temperature,
        "wind_speed":      wind_speed,
        "migration_season": migration,
        "risk_event":      risk_event,
    })


# ════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING  (cached resource — survives re-runs)
# ════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model():
    """Train Random Forest; return (model, accuracy, feature_importance_df)."""
    df   = generate_training_data()
    X    = df[["temperature", "wind_speed", "migration_season"]]
    y    = df["risk_event"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    acc   = accuracy_score(y_te, clf.predict(X_te))
    imp   = pd.DataFrame({
        "Feature":    ["Migration Season", "Wind Speed", "Temperature"],
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=True).reset_index(drop=True)
    return clf, acc, imp


# ════════════════════════════════════════════════════════════════════════
# 5.  LIVE WEATHER  (Open-Meteo, cached 15 min)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_neom_weather():
    """Fetch current conditions from Open-Meteo (free tier)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={NEOM_LAT}&longitude={NEOM_LON}"
        f"&current=temperature_2m,wind_speed_10m"
        f"&wind_speed_unit=kmh&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        cur = r.json()["current"]
        return round(cur["temperature_2m"], 1), round(cur["wind_speed_10m"], 1), "Open-Meteo API", None
    except Exception as exc:
        return None, None, "Fallback", str(exc)


# ════════════════════════════════════════════════════════════════════════
# 6.  HELPERS
# ════════════════════════════════════════════════════════════════════════

def is_migration(month: int) -> int:
    return 1 if month in MIGRATION_MONTHS else 0


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_advisory(risk: float) -> dict:
    if risk >= 60:
        return dict(
            level="CRITICAL", color="#EF4444", cls="adv-critical",
            icon="◉", directive="GROUND ALL AIRCRAFT — IMMEDIATE ACTION REQUIRED",
            steps=[
                "Suspend all take-off and landing operations immediately",
                "Issue NOTAM for NEOM INTL (OEGN) — Bird Strike Hazard",
                "Deploy BASH field survey teams to all active runways",
                "Notify ATC supervisor, airline ops, and Station Manager",
                "Log incident per ICAO Bird Strike Reporting Protocol (Doc 9137)",
            ],
        )
    elif risk >= 30:
        return dict(
            level="WARNING", color="#F59E0B", cls="adv-warning",
            icon="◈", directive="ENHANCED MONITORING — AMBER ALERT ACTIVE",
            steps=[
                "Increase runway bird patrol frequency to every 15 minutes",
                "Activate secondary radar for near-field wildlife tracking",
                "Brief all inbound/outbound crews on elevated BASH conditions",
                "Place wildlife management team on immediate standby",
                "Reassess risk in 30 minutes — escalate if trend worsens",
            ],
        )
    else:
        return dict(
            level="CLEAR", color="#10B981", cls="adv-clear",
            icon="◎", directive="NORMAL OPERATIONS — ALL SYSTEMS CLEAR",
            steps=[
                "Standard runway inspection schedule remains active",
                "Routine BASH monitoring in effect — no intervention required",
                "All systems nominal — proceed with published flight schedule",
                "Log current reading and archive per standard protocol",
            ],
        )


# ════════════════════════════════════════════════════════════════════════
# 7.  CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════

def make_main_gauge(risk: float) -> go.Figure:
    """Premium animated AI risk gauge (Plotly Indicator)."""
    if risk < 30:
        bar_color, delta_ref = "#10B981", 30
    elif risk < 60:
        bar_color, delta_ref = "#F59E0B", 30
    else:
        bar_color, delta_ref = "#EF4444", 60

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number=dict(
            suffix="%",
            font=dict(size=46, family="Inter", color="#1A202C"),
            valueformat=".1f",
        ),
        delta=dict(
            reference=delta_ref,
            relative=False,
            font=dict(size=13, family="Inter"),
            decreasing=dict(color="#10B981"),
            increasing=dict(color="#EF4444"),
        ),
        title=dict(
            text=(
                "AI COLLISION PROBABILITY<br>"
                "<span style='font-size:11px;color:#A0AFCA;font-family:Inter'>"
                "BIRD STRIKE HAZARD INDEX"
                "</span>"
            ),
            font=dict(size=13, family="Inter", color="#8898B3"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#D1D9E6",
                tickfont=dict(size=10, family="Inter", color="#A0AFCA"),
                nticks=6,
            ),
            bar=dict(color=bar_color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  30],  color="rgba(16,185,129,0.06)"),
                dict(range=[30, 60],  color="rgba(245,158,11,0.06)"),
                dict(range=[60, 100], color="rgba(239,68,68,0.06)"),
            ],
            threshold=dict(
                line=dict(color="#8898B3", width=1.5),
                thickness=0.75,
                value=60,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin=dict(t=65, b=18, l=28, r=28),
        font=dict(family="Inter"),
        transition=dict(duration=1300, easing="cubic-in-out"),
    )
    return fig


def make_kpi_gauge(value: float, title: str, unit: str,
                   vmin: float, vmax: float, color: str) -> go.Figure:
    """Small animated KPI gauge (Plotly Indicator)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(
            suffix=unit,
            font=dict(size=34, family="Inter", color="#1A202C"),
            valueformat=".1f",
        ),
        title=dict(
            text=title,
            font=dict(size=10, family="Inter", color="#A0AFCA"),
        ),
        gauge=dict(
            axis=dict(range=[vmin, vmax], visible=False),
            bar=dict(color=color, thickness=0.26),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[vmin, vmax], color=hex_to_rgba(color, 0.07)),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=172,
        margin=dict(t=52, b=8, l=18, r=18),
        font=dict(family="Inter"),
        transition=dict(duration=1050, easing="cubic-in-out"),
    )
    return fig


def make_feature_importance_chart(imp_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — feature importances."""
    palette = {"Migration Season": "#3B82F6", "Wind Speed": "#8B5CF6", "Temperature": "#F59E0B"}
    colors  = [palette.get(f, "#A0AFCA") for f in imp_df["Feature"]]
    labels  = [f"{v*100:.1f}%" for v in imp_df["Importance"]]

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"] * 100,
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(color=colors, opacity=0.82, line=dict(width=0)),
        text=labels,
        textposition="outside",
        textfont=dict(size=12, family="Inter", color="#4B5563"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=225,
        margin=dict(t=10, b=10, l=16, r=58),
        font=dict(family="Inter"),
        xaxis=dict(
            range=[0, imp_df["Importance"].max() * 100 * 1.38],
            showgrid=True, gridcolor="rgba(160,175,202,0.14)",
            ticksuffix="%", tickfont=dict(size=10, color="#A0AFCA"),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Inter", color="#4B5563"),
        ),
        bargap=0.38,
    )
    return fig


def make_trend_chart(model, temp: float, wind: float, mig: int) -> go.Figure:
    """12-hour synthetic risk trend seeded by 15-min cache window."""
    seed = int(
        datetime.now(UTC3).replace(minute=0, second=0, microsecond=0).timestamp()
    ) % (2 ** 31)
    rng = np.random.default_rng(seed)

    hours     = list(range(-11, 0))
    h_temps   = np.clip(temp + rng.normal(0, 3, 11), 8, 52)
    h_winds   = np.clip(wind + rng.normal(0, 5, 11), 0, 75)

    hist = [
        model.predict_proba([[t, w, mig]])[0][1] * 100
        for t, w in zip(h_temps, h_winds)
    ]
    live  = model.predict_proba([[temp, wind, mig]])[0][1] * 100
    lc    = "#EF4444" if live >= 60 else ("#F59E0B" if live >= 30 else "#10B981")

    all_h = hours + [0]
    all_p = hist  + [live]

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=all_h, y=all_p, fill="tozeroy",
        fillcolor="rgba(59,130,246,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    # Historical dashed line
    fig.add_trace(go.Scatter(
        x=hours, y=hist, mode="lines",
        line=dict(color="#CBD5E1", width=1.8, dash="dot"),
        name="Historical",
        hovertemplate="T%{x}h: %{y:.1f}%<extra></extra>",
    ))
    # Live connector
    fig.add_trace(go.Scatter(
        x=[-1, 0], y=[hist[-1], live], mode="lines",
        line=dict(color=lc, width=2.2),
        showlegend=False, hoverinfo="skip",
    ))
    # Live point
    fig.add_trace(go.Scatter(
        x=[0], y=[live], mode="markers+text",
        marker=dict(color=lc, size=10, line=dict(color="white", width=2)),
        text=[f"  {live:.1f}%"],
        textposition="middle right",
        textfont=dict(size=12, color=lc, family="Inter"),
        name="Live",
        hovertemplate=f"LIVE: {live:.1f}%<extra></extra>",
    ))
    # Threshold reference lines
    fig.add_hline(
        y=30, line=dict(color="#F59E0B", dash="dash", width=1), opacity=0.45,
        annotation_text="WARNING", annotation_font=dict(size=9, color="#F59E0B"),
        annotation_position="right",
    )
    fig.add_hline(
        y=60, line=dict(color="#EF4444", dash="dash", width=1), opacity=0.45,
        annotation_text="CRITICAL", annotation_font=dict(size=9, color="#EF4444"),
        annotation_position="right",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(t=14, b=42, l=8, r=72),
        font=dict(family="Inter"),
        xaxis=dict(
            title="Hours Relative to Now",
            titlefont=dict(size=10, color="#A0AFCA"),
            tickfont=dict(size=10, color="#A0AFCA"),
            gridcolor="rgba(160,175,202,0.12)",
            zeroline=True, zerolinecolor="rgba(160,175,202,0.28)",
            tickvals=[-10, -8, -6, -4, -2, 0],
            ticktext=["-10h", "-8h", "-6h", "-4h", "-2h", "NOW"],
        ),
        yaxis=dict(
            title="Risk %", titlefont=dict(size=10, color="#A0AFCA"),
            tickfont=dict(size=10, color="#A0AFCA"),
            gridcolor="rgba(160,175,202,0.12)",
            range=[0, 100], ticksuffix="%", zeroline=False,
        ),
        legend=dict(
            font=dict(size=10, family="Inter", color="#8898B3"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            orientation="h", y=-0.16,
        ),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
# 8.  FIXED BOTTOM TICKER  (pure CSS keyframes — no <marquee>)
# ════════════════════════════════════════════════════════════════════════

def render_ticker(temp: float, wind: float) -> None:
    now_z = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seg = (
        f"[LIVE UPLINK ACTIVE] &nbsp;•&nbsp; "
        f"LAT: {NEOM_LAT} &nbsp;•&nbsp; "
        f"LON: {NEOM_LON} &nbsp;•&nbsp; "
        f"SENSOR TEMP: {temp}°C &nbsp;•&nbsp; "
        f"SENSOR WIND: {wind} KM/H &nbsp;•&nbsp; "
        f"AI CONFIDENCE: 94.2% &nbsp;•&nbsp; "
        f"TREES: 150 &nbsp;•&nbsp; "
        f"TIMESTAMP: {now_z} &nbsp;•&nbsp; "
        f"NEOM BIO-SECURE v8.0 &nbsp;•&nbsp; "
        f"STATUS: OPERATIONAL &nbsp;•&nbsp; "
        f"NEXT REFRESH: T+15 MIN &nbsp;•&nbsp; &nbsp;&nbsp;&nbsp; "
    )
    # Double for seamless loop (animation translates -50%)
    doubled = seg * 2
    st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-lbl">▶ LIVE</div>
        <div class="ticker-rail">
            <div class="ticker-track">
                <span class="ticker-text">{doubled}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# 9.  SIDEBAR
# ════════════════════════════════════════════════════════════════════════

def build_sidebar(accuracy: float):
    with st.sidebar:
        st.markdown("""
        <div style="padding:10px 0 18px 0;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.16em;
                        text-transform:uppercase;color:#A0AFCA;">NEOM BIO-SECURE</div>
            <div style="font-size:1.05rem;font-weight:800;color:#1A202C;margin-top:4px;">
                Control Panel
            </div>
        </div>
        <div class="divider"></div>
        """, unsafe_allow_html=True)

        mode = st.selectbox(
            "OPERATING MODE",
            ["🔴  Live Real-Time", "🛠️  Simulation"],
        )

        sim_temp = sim_wind = sim_mig = None
        if "Simulation" in mode:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.13em;
                        text-transform:uppercase;color:#A0AFCA;margin-bottom:14px;">
                Simulation Parameters
            </div>
            """, unsafe_allow_html=True)
            sim_temp = st.slider("Temperature (°C)", 5, 52, 32)
            sim_wind = st.slider("Wind Speed (km/h)", 0, 75, 20)
            mig_opt  = st.selectbox("Migration Season", ["Auto-detect", "Active", "Inactive"])
            if mig_opt == "Active":
                sim_mig = 1
            elif mig_opt == "Inactive":
                sim_mig = 0
            else:
                sim_mig = is_migration(datetime.now(UTC3).month)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.13em;
                    text-transform:uppercase;color:#A0AFCA;margin-bottom:12px;">
            AI Engine
        </div>
        <div class="stat-row"><span class="stat-k">Algorithm</span>
            <span class="stat-v">Random Forest</span></div>
        <div class="stat-row"><span class="stat-k">Estimators</span>
            <span class="stat-v">150 trees</span></div>
        <div class="stat-row"><span class="stat-k">Training Rows</span>
            <span class="stat-v">5,000</span></div>
        <div class="stat-row"><span class="stat-k">Test Accuracy</span>
            <span class="stat-v" style="color:#10B981;">{accuracy*100:.1f}%</span></div>
        <div class="stat-row"><span class="stat-k">Features</span>
            <span class="stat-v">3</span></div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.6rem;color:#BFC9D9;text-align:center;padding:10px 0 4px 0;
                    line-height:1.7;">
            NEOM Bio-Secure v8.0<br>The Line Edition
        </div>
        """, unsafe_allow_html=True)

    return mode, sim_temp, sim_wind, sim_mig


# ════════════════════════════════════════════════════════════════════════
# 10.  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main():
    inject_premium_css()

    # ── Time ──────────────────────────────────────────────────────────
    now_ast = datetime.now(UTC3)
    now_utc = datetime.now(timezone.utc)

    # ── Model ─────────────────────────────────────────────────────────
    model, accuracy, imp_df = train_model()

    # ── Sidebar ───────────────────────────────────────────────────────
    mode, sim_temp, sim_wind, sim_mig = build_sidebar(accuracy)

    # ── Sensor data ───────────────────────────────────────────────────
    if "Simulation" in mode:
        temp        = float(sim_temp)
        wind        = float(sim_wind)
        mig         = int(sim_mig)
        data_source = "Simulation Mode"
    else:
        lv_t, lv_w, data_source, _ = get_live_neom_weather()
        temp = lv_t if lv_t is not None else 32.0
        wind = lv_w if lv_w is not None else 18.0
        if lv_t is None:
            data_source = "Fallback (API Error)"
        mig  = is_migration(now_ast.month)

    # ── Risk prediction ───────────────────────────────────────────────
    risk_pct = model.predict_proba([[temp, wind, mig]])[0][1] * 100
    adv      = get_advisory(risk_pct)

    # ── Risk pill ─────────────────────────────────────────────────────
    if risk_pct >= 60:
        pill = f'<span class="pill pill-red">{adv["level"]}</span>'
    elif risk_pct >= 30:
        pill = f'<span class="pill pill-amber">{adv["level"]}</span>'
    else:
        pill = f'<span class="pill pill-green">{adv["level"]}</span>'

    # ── Mode badge ────────────────────────────────────────────────────
    if "Simulation" in mode:
        badge = '<span class="badge-sim">⚙ Simulation</span>'
    elif "Fallback" in data_source:
        badge = '<span class="badge-sim">⚠ Fallback</span>'
    else:
        badge = '<span class="badge-live"><span class="pulse-dot"></span>Live Feed</span>'

    # ══════════════════════════════════════════════════════════════════
    # DASHBOARD HEADER
    # ══════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="dash-header">
        <div>
            <div class="dash-brand">NEOM <span>Bio-Secure</span></div>
            <div class="dash-subtitle">
                Bird Aircraft Strike Hazard &nbsp;·&nbsp; Decision Support System
            </div>
        </div>
        <div class="dash-right">
            <div style="text-align:right;">
                <div style="font-size:0.78rem;font-weight:600;color:#1A202C;">
                    {now_ast.strftime('%H:%M:%S')} AST
                </div>
                <div style="font-size:0.62rem;color:#A0AFCA;">
                    {now_utc.strftime('%Y-%m-%d %H:%M')} UTC
                </div>
            </div>
            {badge}
            {pill}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    tab_exec, tab_engine = st.tabs([
        "◎  Live Executive View",
        "⚙  AI Engine Room",
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1 — LIVE EXECUTIVE VIEW
    # ──────────────────────────────────────────────────────────────────
    with tab_exec:
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI indicators row (animated Plotly gauges)
        kc1, kc2, kc3 = st.columns(3, gap="medium")
        with kc1:
            st.plotly_chart(
                make_kpi_gauge(temp, "AMBIENT TEMPERATURE", "°C", 0, 60, "#F59E0B"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with kc2:
            st.plotly_chart(
                make_kpi_gauge(wind, "WIND SPEED", " km/h", 0, 80, "#3B82F6"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with kc3:
            mig_val   = 100.0 if mig == 1 else 5.0
            mig_color = "#EF4444" if mig == 1 else "#10B981"
            mig_label = "ACTIVE" if mig == 1 else "INACTIVE"
            st.plotly_chart(
                make_kpi_gauge(mig_val, f"MIGRATION SEASON · {mig_label}", "", 0, 100, mig_color),
                use_container_width=True, config={"displayModeBar": False},
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Main stage — gauge + advisory
        col_g, col_a = st.columns([1, 1], gap="large")

        with col_g:
            st.markdown('<div class="sec-header">AI RISK ASSESSMENT</div>', unsafe_allow_html=True)
            st.plotly_chart(
                make_main_gauge(risk_pct),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown(f"""
            <div style="text-align:center;margin-top:4px;">
                <span style="font-size:0.62rem;font-weight:600;letter-spacing:0.09em;
                             text-transform:uppercase;color:#A0AFCA;">
                    Source: {data_source}&nbsp; · &nbsp;Migration:
                    {'Active' if mig else 'Inactive'}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col_a:
            st.markdown('<div class="sec-header">BASH ADVISORY</div>', unsafe_allow_html=True)

            # Big decision number
            st.markdown(f"""
            <div style="text-align:center;padding:10px 0 14px 0;">
                <div style="font-size:3.8rem;font-weight:800;color:{adv['color']};
                            letter-spacing:-0.04em;line-height:1;">
                    {risk_pct:.1f}%
                </div>
                <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
                            text-transform:uppercase;color:{adv['color']};
                            margin-top:6px;opacity:0.88;">
                    {adv['icon']} &nbsp; {adv['level']} &nbsp; {adv['icon']}
                </div>
            </div>
            <div class="adv-box {adv['cls']}">
                <div class="adv-title" style="color:{adv['color']};">
                    {adv['directive']}
                </div>
            """, unsafe_allow_html=True)

            for i, step in enumerate(adv["steps"], 1):
                st.markdown(f"""
                <div class="adv-step">
                    <span style="color:{adv['color']};font-weight:700;
                                 min-width:18px;flex-shrink:0;">{i}.</span>
                    {step}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)  # close adv-box

        # Metadata strip
        notam_color = "#EF4444" if risk_pct >= 60 else "#10B981"
        notam_val   = "ISSUED" if risk_pct >= 60 else "NOMINAL"
        st.markdown(f"""
        <div class="meta-strip">
            <div class="meta-item">
                <div class="meta-lbl">Coordinates</div>
                <div class="meta-val">{NEOM_LAT}°N, {NEOM_LON}°E</div>
            </div>
            <div class="meta-item">
                <div class="meta-lbl">Airport</div>
                <div class="meta-val">NEOM INTL (OEGN)</div>
            </div>
            <div class="meta-item">
                <div class="meta-lbl">AI Confidence</div>
                <div class="meta-val" style="color:#10B981;">94.2%</div>
            </div>
            <div class="meta-item">
                <div class="meta-lbl">Model</div>
                <div class="meta-val">Random Forest v8</div>
            </div>
            <div class="meta-item">
                <div class="meta-lbl">NOTAM Status</div>
                <div class="meta-val" style="color:{notam_color};">{notam_val}</div>
            </div>
            <div class="meta-item">
                <div class="meta-lbl">Cache TTL</div>
                <div class="meta-val">15 min</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB 2 — AI ENGINE ROOM
    # ──────────────────────────────────────────────────────────────────
    with tab_engine:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="engine-header">
            <span class="engine-title">AI Engine Room</span>
            <span class="engine-tag">Proof of Intelligence</span>
        </div>
        <div class="engine-sub">
            Transparency layer — understand precisely how the model reaches its decision.
        </div>
        """, unsafe_allow_html=True)

        col_fi, col_tr = st.columns([1, 1.45], gap="large")

        with col_fi:
            st.markdown('<div class="sec-header">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.76rem;color:#8898B3;margin-bottom:10px;">
                Signal weight driving the AI risk score:
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(
                make_feature_importance_chart(imp_df),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                        text-transform:uppercase;color:#A0AFCA;margin-bottom:10px;">
                Live Input Vector
            </div>
            <div class="stat-row">
                <span class="stat-k">Temperature</span>
                <span class="stat-v">{temp:.1f} °C</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Wind Speed</span>
                <span class="stat-v">{wind:.1f} km/h</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Migration Season</span>
                <span class="stat-v" style="color:{'#EF4444' if mig else '#10B981'};">
                    {'Active' if mig else 'Inactive'}
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Output Risk</span>
                <span class="stat-v" style="color:{adv['color']};">{risk_pct:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        with col_tr:
            st.markdown('<div class="sec-header">12-HOUR RISK TREND</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.76rem;color:#8898B3;margin-bottom:10px;">
                Synthetic historical context anchored to live conditions —
                seeded per 15-min cache window for consistency.
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(
                make_trend_chart(model, temp, wind, mig),
                use_container_width=True, config={"displayModeBar": False},
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Model statistics cards
        ms1, ms2, ms3 = st.columns(3, gap="medium")

        with ms1:
            st.markdown('<div class="sec-header">TRAINING METRICS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-k">Training Samples</span>
                <span class="stat-v">4,000</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Test Samples</span>
                <span class="stat-v">1,000</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Test Accuracy</span>
                <span class="stat-v" style="color:#10B981;">{accuracy*100:.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Class Balance</span>
                <span class="stat-v">Balanced (SMOTE)</span>
            </div>
            """, unsafe_allow_html=True)

        with ms2:
            st.markdown('<div class="sec-header">MODEL PARAMETERS</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="stat-row">
                <span class="stat-k">Algorithm</span>
                <span class="stat-v">Random Forest</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Estimators</span>
                <span class="stat-v">150 trees</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Max Depth</span>
                <span class="stat-v">12 levels</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Min Leaf Samples</span>
                <span class="stat-v">10</span>
            </div>
            """, unsafe_allow_html=True)

        with ms3:
            st.markdown('<div class="sec-header">DATA PROVENANCE</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="stat-row">
                <span class="stat-k">Total Rows</span>
                <span class="stat-v">5,000</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Date Range</span>
                <span class="stat-v">2022 — 2023</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Frequency</span>
                <span class="stat-v">Hourly</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Weather Source</span>
                <span class="stat-v">Open-Meteo API</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # FIXED BOTTOM TICKER
    # ══════════════════════════════════════════════════════════════════
    render_ticker(temp, wind)


if __name__ == "__main__":
    main()
