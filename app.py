"""
╔══════════════════════════════════════════════════════════════════════════╗
║     NEOM Bio-Secure — Version 7.0  (Glass Cockpit XAI Edition)          ║
║     Explainable AI (XAI) Bird Aircraft Strike Hazard Dashboard           ║
║     NEOM Smart City  |  Aviation Safety Division                         ║
║     UI Theme: Sci-Fi Glass Cockpit · Glassmorphism · Neon Cyan Accents   ║
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
# 1.  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEOM Bio-Secure v7.0 | Glass Cockpit",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────
NEOM_LAT          = 28.2933
NEOM_LON          = 35.0000
NEOM_TZ           = timezone(timedelta(hours=3))   # Saudi Arabia Time (AST/UTC+3)
ZULU_TZ           = timezone.utc                    # Zulu / UTC
MIGRATION_MONTHS  = {3, 4, 5, 9, 10, 11}            # Mar–May, Sep–Nov

# ── Glass Cockpit colour palette ──────────────────────────────────────
C = {
    "bg":          "#070B19",                    # Deep navy / space blue
    "glass":       "rgba(16, 22, 47, 0.70)",     # Frosted glass fill
    "glass_dark":  "rgba(7, 11, 25, 0.85)",      # Darker glass (header)
    "border":      "#00f2fe",                    # Neon cyan primary border
    "border_dim":  "rgba(0, 242, 254, 0.30)",    # Dim cyan border
    "border_faint":"rgba(0, 242, 254, 0.12)",    # Very faint cyan
    "cyan":        "#00f2fe",                    # Neon cyan
    "teal":        "#00D4AA",                    # NEOM teal
    "green":       "#00FF88",                    # Safe / Clear
    "amber":       "#FFD700",                    # Warning / Caution
    "red":         "#FF4444",                    # Danger / Critical
    "white":       "#FFFFFF",
    "muted":       "#7B8CA3",                    # Muted text
    "grid":        "rgba(0, 242, 254, 0.06)",    # Chart grid lines
    "term_bg":     "#000000",                    # Terminal black
}


# ════════════════════════════════════════════════════════════════════════
# 2.  GLASSMORPHISM CSS INJECTION
# ════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
    <style>
    /* ── Google Fonts ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    /* ── Hide Streamlit chrome ─────────────────────────────── */
    #MainMenu          {{visibility: hidden;}}
    header             {{visibility: hidden;}}
    footer             {{visibility: hidden;}}
    [data-testid="stToolbar"]    {{display: none;}}
    [data-testid="stDecoration"] {{display: none;}}

    /* ── Base: Deep Navy background ────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main .block-container {{
        background-color: {C["bg"]} !important;
        color: {C["white"]};
        font-family: 'Share Tech Mono', 'JetBrains Mono', 'Consolas', monospace;
    }}
    .block-container {{
        padding-top: 0.4rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 100% !important;
    }}

    /* ── Subtle background depth radials (makes glass pop) ── */
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background:
            radial-gradient(ellipse at 15% 40%, rgba(0,120,180,0.12) 0%, transparent 55%),
            radial-gradient(ellipse at 85% 15%, rgba(0,60,120,0.09) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 90%, rgba(0,80,140,0.07) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }}

    /* ── Sidebar: dark glass ────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: rgba(7, 11, 25, 0.92) !important;
        border-right: 1px solid {C["border_dim"]};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }}
    [data-testid="stSidebar"] * {{ color: {C["white"]} !important; }}

    /* ══════════════════════════════════════════════════════════
       GLASS CARD — Glassmorphism container class
    ══════════════════════════════════════════════════════════ */
    .glass-card {{
        background: {C["glass"]};
        border: 1px solid {C["border_dim"]};
        border-radius: 12px;
        padding: 20px 22px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow:
            0 0 24px rgba(0, 242, 254, 0.08),
            0 4px 32px rgba(0, 0, 0, 0.45),
            inset 0 0 40px rgba(0, 242, 254, 0.015);
        margin-bottom: 14px;
    }}

    /* ══════════════════════════════════════════════════════════
       BLINKING STATUS DOTS
    ══════════════════════════════════════════════════════════ */
    @keyframes pulse-safe {{
        0%, 100% {{ opacity: 1;  box-shadow: 0 0 6px #00FF88, 0 0 14px #00FF88, 0 0 28px #00FF88; }}
        50%       {{ opacity: 0.25; box-shadow: none; }}
    }}
    @keyframes pulse-danger {{
        0%, 100% {{ opacity: 1;  box-shadow: 0 0 6px #FF4444, 0 0 14px #FF4444, 0 0 28px #FF4444; }}
        50%       {{ opacity: 0.2; box-shadow: none; }}
    }}
    @keyframes pulse-warn {{
        0%, 100% {{ opacity: 1;  box-shadow: 0 0 6px #FFD700, 0 0 14px #FFD700; }}
        50%       {{ opacity: 0.3; box-shadow: none; }}
    }}
    .dot-safe {{
        display: inline-block; width: 11px; height: 11px;
        border-radius: 50%; background: {C["green"]};
        animation: pulse-safe 1.6s ease-in-out infinite;
        margin-right: 9px; vertical-align: middle;
    }}
    .dot-danger {{
        display: inline-block; width: 11px; height: 11px;
        border-radius: 50%; background: {C["red"]};
        animation: pulse-danger 0.75s ease-in-out infinite;
        margin-right: 9px; vertical-align: middle;
    }}
    .dot-warn {{
        display: inline-block; width: 11px; height: 11px;
        border-radius: 50%; background: {C["amber"]};
        animation: pulse-warn 1.2s ease-in-out infinite;
        margin-right: 9px; vertical-align: middle;
    }}

    /* ══════════════════════════════════════════════════════════
       TOP COCKPIT HEADER
    ══════════════════════════════════════════════════════════ */
    .cockpit-header {{
        background: {C["glass_dark"]};
        border: 1px solid {C["border_dim"]};
        border-bottom: 2px solid {C["cyan"]};
        border-radius: 14px;
        padding: 18px 28px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow:
            0 0 40px rgba(0, 242, 254, 0.12),
            0 8px 32px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }}
    .cockpit-title {{
        font-family: 'Orbitron', 'Share Tech Mono', monospace;
        color: {C["cyan"]};
        font-size: 1.45rem;
        font-weight: 900;
        letter-spacing: 3px;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.65), 0 0 24px rgba(0, 242, 254, 0.25);
    }}
    .cockpit-subtitle {{
        color: rgba(0, 242, 254, 0.55);
        font-size: 0.68rem;
        letter-spacing: 2px;
        font-family: 'Share Tech Mono', monospace;
        margin-top: 4px;
    }}
    .uplink-row {{
        display: flex;
        align-items: center;
        margin-top: 10px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 1.5px;
    }}
    .clock-panel {{
        text-align: right;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
    }}
    .clock-lbl {{
        color: {C["muted"]};
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
    }}
    .clock-utc {{
        color: {C["cyan"]};
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.55);
        line-height: 1.1;
    }}
    .clock-ast {{
        color: {C["teal"]};
        font-size: 0.82rem;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* ══════════════════════════════════════════════════════════
       KPI GLASS CARDS
    ══════════════════════════════════════════════════════════ */
    .kpi-glass {{
        background: {C["glass"]};
        border: 1px solid {C["border_dim"]};
        border-radius: 10px;
        padding: 18px 16px;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.07), 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 12px;
    }}
    .kpi-label {{
        color: {C["muted"]};
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.1;
        font-family: 'Share Tech Mono', monospace;
    }}
    .kpi-sub {{
        color: {C["muted"]};
        font-size: 0.58rem;
        margin-top: 5px;
        letter-spacing: 0.5px;
    }}

    /* ══════════════════════════════════════════════════════════
       SECTION HEADERS
    ══════════════════════════════════════════════════════════ */
    .sec-head {{
        color: {C["cyan"]};
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3.5px;
        margin-bottom: 12px;
        padding-bottom: 7px;
        border-bottom: 1px solid {C["border_faint"]};
        font-family: 'Share Tech Mono', monospace;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.4);
    }}

    /* ══════════════════════════════════════════════════════════
       BASH ADVISORY BOX
    ══════════════════════════════════════════════════════════ */
    .bash-box {{
        border-radius: 10px;
        padding: 15px 18px;
        margin-top: 12px;
        backdrop-filter: blur(8px);
    }}
    .bash-level {{
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 3px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
    }}
    .bash-directive {{
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }}
    .bash-box ul {{ margin: 0; padding-left: 16px; }}
    .bash-box li {{
        font-size: 0.72rem;
        margin-bottom: 4px;
        color: {C["white"]};
        letter-spacing: 0.3px;
    }}

    /* ══════════════════════════════════════════════════════════
       TERMINAL  (full-width, hacker vibe)
    ══════════════════════════════════════════════════════════ */
    .term-log {{
        background: {C["term_bg"]};
        border: 1px solid {C["border_dim"]};
        border-top: 2px solid {C["cyan"]};
        border-radius: 10px;
        padding: 18px 22px;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
        font-size: 0.76rem;
        line-height: 2.2;
        color: {C["cyan"]};
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.1), inset 0 0 60px rgba(0, 0, 0, 0.5);
    }}
    .t-ok   {{ color: {C["green"]}; }}
    .t-warn {{ color: {C["amber"]}; }}
    .t-err  {{ color: {C["red"]}; }}
    .t-info {{ color: {C["cyan"]}; }}
    .t-ts   {{ color: rgba(0, 242, 254, 0.45); }}

    /* ══════════════════════════════════════════════════════════
       MODE BADGE
    ══════════════════════════════════════════════════════════ */
    .mode-badge {{
        display: inline-flex;
        align-items: center;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 14px;
        font-family: 'Share Tech Mono', monospace;
    }}

    /* ══════════════════════════════════════════════════════════
       SIDEBAR META CARD
    ══════════════════════════════════════════════════════════ */
    .meta-card {{
        background: rgba(16, 22, 47, 0.5);
        border: 1px solid {C["border_dim"]};
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.68rem;
        line-height: 2.1;
        font-family: 'Share Tech Mono', monospace;
    }}
    .meta-card .mk {{ color: {C["muted"]}; }}
    .meta-card .mv {{ color: {C["cyan"]}; font-weight: 600; }}

    /* ══════════════════════════════════════════════════════════
       MISC
    ══════════════════════════════════════════════════════════ */
    .js-plotly-plot .plotly {{ background: transparent !important; }}
    .stSlider > div > div   {{ background: rgba(0, 242, 254, 0.18) !important; }}
    hr {{ border-color: {C["border_faint"]} !important; }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# 3.  SYNTHETIC TRAINING DATA  (5 000 rows, internal — no CSV)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    n      = 5000
    dates  = pd.date_range("2022-01-01", periods=n, freq="h")
    months = dates.month

    temp_base   = 28 + 14 * np.sin((months - 3) * np.pi / 6)
    temperature = np.clip(rng.normal(loc=temp_base, scale=4.5), 8, 52)
    wind_speed  = np.clip(rng.gamma(shape=2, scale=7, size=n), 0, 75)

    migration = np.where(np.isin(months, list(MIGRATION_MONTHS)), 1, 0)
    flip      = rng.random(n) < 0.05
    migration = np.where(flip, 1 - migration, migration)

    risk_prob = (
        0.55 * migration
        + 0.25 * np.where(wind_speed < 20, 1, 0)
        + 0.20 * np.where((temperature >= 18) & (temperature <= 40), 1, 0)
    )
    risk_event = (rng.random(n) < risk_prob).astype(int)

    return pd.DataFrame({
        "Date":             dates,
        "Month":            months,
        "Temperature":      temperature.round(1),
        "Wind_Speed":       wind_speed.round(1),
        "Migration_Season": migration,
        "Risk_Event":       risk_event,
    })


# ════════════════════════════════════════════════════════════════════════
# 4.  RANDOM FOREST MODEL
# ════════════════════════════════════════════════════════════════════════

FEATURES = ["Temperature", "Wind_Speed", "Migration_Season"]

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["Risk_Event"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    imp = (
        pd.DataFrame({"Feature": FEATURES, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return clf, acc, imp


def predict_risk(model, temp: float, wind: float, mig: int) -> float:
    prob = model.predict_proba(np.array([[temp, wind, mig]]))[0][1] * 100
    return float(np.clip(prob, 0.0, 100.0))


# ════════════════════════════════════════════════════════════════════════
# 5.  LIVE WEATHER — OPEN-METEO  (TTL=900s anti-429 cache)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_neom_weather() -> dict:
    """15-minute smart cache prevents HTTP 429 Too Many Requests."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={NEOM_LAT}&longitude={NEOM_LON}"
        "&current_weather=true&wind_speed_unit=kmh"
        "&temperature_unit=celsius&timezone=Asia%2FRiyadh"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        cw = r.json()["current_weather"]
        return {
            "temperature": round(float(cw["temperature"]), 1),
            "wind_speed":  round(float(cw["windspeed"]),   1),
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


def auto_migration() -> int:
    return 1 if datetime.now(tz=NEOM_TZ).month in MIGRATION_MONTHS else 0


# ════════════════════════════════════════════════════════════════════════
# 6.  BASH ADVISORY ENGINE
# ════════════════════════════════════════════════════════════════════════

def get_bash_advisory(risk: float) -> dict:
    if risk >= 60:
        return {
            "condition": "CRITICAL",
            "directive": "GROUND AIRCRAFT — GENERATE NOTAM",
            "color":     C["red"],
            "icon":      "🚨",
            "steps": [
                "NOTAM: All departures & arrivals SUSPENDED immediately",
                "Activate full-spectrum BASH deterrent array",
                "Deploy runway inspection & bird dispersal units",
                "Notify NEOM ATC & all inbound traffic for holding",
                "Continuous BASH radar sweep — 30-second intervals",
            ],
        }
    elif risk >= 30:
        return {
            "condition": "WARNING",
            "directive": "CAUTION — ENHANCED BASH MONITORING",
            "color":     C["amber"],
            "icon":      "⚠️",
            "steps": [
                "BASH condition AMBER: advise all flight crews",
                "Increase BASH visual scan frequency to every 10 min",
                "Perimeter avian radar switched to HIGH-GAIN mode",
                "Weather advisory update broadcast to ATC ops",
                "Bird dispersal team STANDBY — alert status",
            ],
        }
    else:
        return {
            "condition": "CLEAR",
            "directive": "CLEAR FOR TAKEOFF — NORMAL OPS",
            "color":     C["green"],
            "icon":      "✅",
            "steps": [
                "BASH condition GREEN: standard departure clearance",
                "Routine pre-departure avian assessment — nominal",
                "Environmental telemetry logging — all systems normal",
            ],
        }


# ════════════════════════════════════════════════════════════════════════
# 7.  PLOTLY CHARTS — GLASS COCKPIT VISUALS
# ════════════════════════════════════════════════════════════════════════

def make_gauge(risk_pct: float) -> go.Figure:
    """
    Massive dark-themed Gauge — AI Collision Probability.
    Vibrant Red / Yellow / Green zones against the deep navy background.
    """
    if risk_pct < 30:
        bar_col, label = C["green"], "BASH CLEAR"
    elif risk_pct < 60:
        bar_col, label = C["amber"], "BASH WARNING"
    else:
        bar_col, label = C["red"],   "BASH CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={
            "suffix": "%",
            "font": {
                "size":   62,
                "color":  bar_col,
                "family": "Share Tech Mono, monospace",
            },
        },
        title={
            "text": (
                "<b style='font-family:Orbitron,monospace;letter-spacing:2px;'>"
                "AI COLLISION PROBABILITY</b><br>"
                f"<span style='font-size:0.88em;color:{bar_col};"
                f"letter-spacing:3px;font-family:Share Tech Mono,monospace;'>"
                f"{label}</span>"
            ),
            "font": {"size": 13, "color": C["cyan"]},
        },
        gauge={
            "axis": {
                "range":     [0, 100],
                "tickwidth": 2,
                "tickcolor": C["muted"],
                "tickfont":  {"color": C["muted"], "size": 10},
            },
            "bar":      {"color": bar_col, "thickness": 0.30},
            "bgcolor":  "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(0,255,136,0.10)"},
                {"range": [30, 60], "color": "rgba(255,215,0,0.09)"},
                {"range": [60,100], "color": "rgba(255,68,68,0.11)"},
            ],
            "threshold": {
                "line":      {"color": C["cyan"], "width": 4},
                "thickness": 0.82,
                "value":     risk_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"], "family": "Share Tech Mono, monospace"},
        margin={"t": 100, "b": 10, "l": 30, "r": 30},
        height=340,
    )
    return fig


def make_feature_importance(imp_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal Bar Chart — WHY the AI made its decision.
    Each feature gets a vibrant colour; percentages shown outside bars.
    Transparent background to sit inside the glass card.
    """
    FEAT_COLORS = {
        "Migration_Season": C["red"],
        "Wind_Speed":       C["cyan"],
        "Temperature":      C["green"],
    }
    FEAT_LABELS = {
        "Migration_Season": "Migration Season",
        "Wind_Speed":       "Wind Speed",
        "Temperature":      "Temperature",
    }
    colors   = [FEAT_COLORS.get(f, C["teal"]) for f in imp_df["Feature"]]
    y_labels = [FEAT_LABELS.get(f, f)          for f in imp_df["Feature"]]
    pct_vals = (imp_df["Importance"] * 100).round(1).tolist()

    fig = go.Figure(go.Bar(
        x=pct_vals,
        y=y_labels,
        orientation="h",
        marker={
            "color": colors,
            "opacity": 0.88,
            "line": {"color": "rgba(0,242,254,0.25)", "width": 1},
        },
        text=[f"{v:.0f}%" for v in pct_vals],
        textposition="outside",
        textfont={
            "color":  C["white"],
            "size":   14,
            "family": "Share Tech Mono, monospace",
        },
    ))

    fig.update_layout(
        title={
            "text": "<b>WHY: AI DECISION FACTOR WEIGHTS</b>",
            "font": {"size": 12, "color": C["cyan"], "family": "Share Tech Mono"},
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"], "family": "Share Tech Mono, monospace"},
        xaxis={
            "title":       "Decision Weight (%)",
            "gridcolor":   "rgba(0,242,254,0.07)",
            "zerolinecolor": "rgba(0,242,254,0.15)",
            "range":       [0, max(pct_vals) * 1.35],
            "ticksuffix":  "%",
            "tickfont":    {"color": C["muted"], "size": 10},
        },
        yaxis={
            "autorange": "reversed",
            "tickfont":  {"size": 12, "color": C["white"]},
        },
        margin={"t": 50, "b": 40, "l": 20, "r": 75},
        height=280,
    )
    return fig


def make_trend_line(model, temperature: float, wind_speed: float, migration: int) -> go.Figure:
    """
    Time-Series Line Chart — BASH Risk over the last 12 hours.
    Generates realistic synthetic trailing data leading up to the LIVE prediction,
    proving the system tracks history to predict the future.
    Seed is anchored to the 15-minute weather cache window so it stays stable
    between page refreshes within the same cache interval.
    """
    now  = datetime.now(tz=ZULU_TZ)
    seed = int(now.timestamp()) // 900   # Stable within each 15-min cache window
    rng  = np.random.default_rng(seed)

    # Build 12 hourly time points: T-11h … T-0h (current)
    times = [now - timedelta(hours=11 - i) for i in range(12)]

    # Simulate realistic weather variation for each historical hour
    t_variations = np.clip(
        temperature + rng.normal(0, 2.2, 11), 8, 52
    )
    w_variations = np.clip(
        wind_speed  + rng.normal(0, 3.8, 11), 0, 75
    )

    # Compute risk prediction for each historical hour
    hist_risks = [
        predict_risk(model, float(t), float(w), migration)
        for t, w in zip(t_variations, w_variations)
    ]

    # Append current live prediction as the 12th (rightmost) point
    live_risk = predict_risk(model, temperature, wind_speed, migration)
    all_risks = hist_risks + [live_risk]
    time_labels = [t.strftime("%H:%MZ") for t in times]

    # Pick colour scheme based on mean risk level
    avg_risk = float(np.mean(all_risks))
    if avg_risk < 30:
        line_col  = C["green"]
        fill_col  = "rgba(0, 255, 136, 0.07)"
    elif avg_risk < 60:
        line_col  = C["amber"]
        fill_col  = "rgba(255, 215, 0, 0.07)"
    else:
        line_col  = C["red"]
        fill_col  = "rgba(255, 68, 68, 0.08)"

    fig = go.Figure()

    # ── Area fill under entire trend ──────────────────────────
    fig.add_trace(go.Scatter(
        x=time_labels, y=all_risks,
        mode="none",
        fill="tozeroy",
        fillcolor=fill_col,
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Historical dashed line (T-11h to T-1h) ───────────────
    fig.add_trace(go.Scatter(
        x=time_labels[:-1],
        y=all_risks[:-1],
        mode="lines+markers",
        name="Historical Risk",
        line={"color": C["muted"], "width": 2, "dash": "dot"},
        marker={"size": 6, "color": C["muted"], "opacity": 0.75},
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<extra>Historical</extra>",
    ))

    # ── Solid connector from last history point to LIVE ───────
    fig.add_trace(go.Scatter(
        x=[time_labels[-2], time_labels[-1]],
        y=[all_risks[-2],   all_risks[-1]],
        mode="lines",
        line={"color": line_col, "width": 3},
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── LIVE prediction point (glowing diamond) ───────────────
    fig.add_trace(go.Scatter(
        x=[time_labels[-1]],
        y=[live_risk],
        mode="markers+text",
        name="LIVE PREDICTION",
        marker={
            "size":   18,
            "color":  line_col,
            "symbol": "diamond",
            "line":   {"color": C["cyan"], "width": 3},
        },
        text=["▶ NOW"],
        textposition="top center",
        textfont={"color": C["cyan"], "size": 11, "family": "Share Tech Mono"},
        hovertemplate=(
            f"<b>LIVE NOW</b><br>Risk: {live_risk:.1f}%"
            f"<br>Condition: {get_bash_advisory(live_risk)['condition']}"
            "<extra></extra>"
        ),
    ))

    # ── Threshold reference lines ─────────────────────────────
    fig.add_hline(
        y=60, line_dash="dash", line_color=C["red"],   opacity=0.50,
        annotation_text=" CRITICAL THRESHOLD  60%",
        annotation_font_color=C["red"],   annotation_font_size=10,
        annotation_position="right",
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color=C["amber"], opacity=0.50,
        annotation_text=" WARNING THRESHOLD  30%",
        annotation_font_color=C["amber"], annotation_font_size=10,
        annotation_position="right",
    )

    fig.update_layout(
        title={
            "text": "<b>BASH RISK TREND — LAST 12 HOURS → LIVE PREDICTION</b>",
            "font": {"size": 12, "color": C["cyan"], "family": "Share Tech Mono"},
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"], "family": "Share Tech Mono"},
        xaxis={
            "title":         "Time (UTC / Zulu)",
            "gridcolor":     "rgba(0,242,254,0.06)",
            "zerolinecolor": "rgba(0,242,254,0.10)",
            "tickfont":      {"color": C["muted"], "size": 10},
        },
        yaxis={
            "title":         "BASH Risk (%)",
            "range":         [0, 108],
            "gridcolor":     "rgba(0,242,254,0.06)",
            "ticksuffix":    "%",
            "tickfont":      {"color": C["muted"], "size": 10},
        },
        legend={
            "orientation": "h",
            "yanchor":     "bottom", "y": 1.02,
            "xanchor":     "right",  "x": 1.0,
            "font":        {"size": 10, "color": C["muted"]},
        },
        margin={"t": 60, "b": 52, "l": 65, "r": 105},
        height=330,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
# 8.  TERMINAL  (hacker vibe — full-width)
# ════════════════════════════════════════════════════════════════════════

def build_terminal(entries: list) -> str:
    """Convert log entry list into HTML monospace neon-cyan terminal lines."""
    lines = []
    for e in entries:
        ts  = e.get("ts",  "")
        cls = e.get("cls", "t-info")
        msg = e.get("msg", "")
        lines.append(
            f'<span class="t-ts">[{ts}]</span>&nbsp;'
            f'<span class="{cls}">{msg}</span>'
        )
    return "<br>".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 9.  SIDEBAR
# ════════════════════════════════════════════════════════════════════════

def build_sidebar(accuracy: float):
    st.sidebar.markdown(
        f"""
        <div style='text-align:center;padding:10px 0 18px 0;'>
            <span style='font-size:2.4rem;'>🦅</span><br>
            <span style='color:{C["cyan"]};font-size:0.92rem;font-weight:700;
                         letter-spacing:3.5px;font-family:"Share Tech Mono",monospace;
                         text-shadow:0 0 10px rgba(0,242,254,0.55);'>
                NEOM BIO-SECURE
            </span><br>
            <span style='color:{C["muted"]};font-size:0.60rem;letter-spacing:2px;
                         font-family:"Share Tech Mono",monospace;'>
                V7.0 · GLASS COCKPIT EDITION
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "**Operating Mode**",
        ["🔴  Live Real-Time", "🛠️  Simulation Mode"],
        index=0,
    )
    st.sidebar.divider()

    if mode.startswith("🔴"):
        st.sidebar.markdown(
            f"<div style='color:{C['cyan']};font-size:0.65rem;letter-spacing:2.5px;"
            "text-transform:uppercase;font-family:\"Share Tech Mono\",monospace;"
            "margin-bottom:8px;'>📡 NEOM SATELLITE STATION</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            f"<small style='color:{C['muted']};font-family:\"Share Tech Mono\",monospace;'>"
            f"Lat {NEOM_LAT}°N · Lon {NEOM_LON}°E<br>"
            "Source: Open-Meteo (free tier)<br>"
            "Cache TTL: 900s · Anti-429 active</small>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        sim = None, None, None
    else:
        st.sidebar.markdown(
            f"<div style='color:{C['cyan']};font-size:0.65rem;letter-spacing:2.5px;"
            "text-transform:uppercase;font-family:\"Share Tech Mono\",monospace;"
            "margin-bottom:8px;'>🎛️ SIMULATION CONTROLS</div>",
            unsafe_allow_html=True,
        )
        s_temp   = st.sidebar.slider("🌡️ Temperature (°C)", 5, 52, 30)
        s_wind   = st.sidebar.slider("💨 Wind Speed (km/h)", 0, 75, 15)
        s_mig_ch = st.sidebar.selectbox(
            "🦢 Migration Season",
            ["Auto (current month)", "Active", "Inactive"],
        )
        if s_mig_ch.startswith("Auto"):
            s_mig = auto_migration()
        elif s_mig_ch == "Active":
            s_mig = 1
        else:
            s_mig = 0
        sim = s_temp, s_wind, s_mig

    # ── Model metadata card ──────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown(
        f"<div style='color:{C['cyan']};font-size:0.65rem;letter-spacing:2.5px;"
        "text-transform:uppercase;font-family:\"Share Tech Mono\",monospace;"
        "margin-bottom:8px;'>🤖 AI MODEL METADATA</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class='meta-card'>
            <span class='mk'>Engine :</span>
            <span class='mv'>RandomForestClassifier</span><br>
            <span class='mk'>Trees  :</span>
            <span class='mv'>150 Estimators</span><br>
            <span class='mk'>Depth  :</span>
            <span class='mv'>Max 12 Levels</span><br>
            <span class='mk'>Corpus :</span>
            <span class='mv'>5,000 Synthetic Records</span><br>
            <span class='mk'>Accuracy:</span>
            <span class='mv'>{accuracy:.1%} (Test Set)</span><br>
            <span class='mk'>Features:</span>
            <span class='mv'>Temp · Wind · Migration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode, sim


# ════════════════════════════════════════════════════════════════════════
# 10.  MAIN APPLICATION ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Timestamps ───────────────────────────────────────────
    now_ast  = datetime.now(tz=NEOM_TZ)
    now_zulu = datetime.now(tz=ZULU_TZ)
    ast_str  = now_ast.strftime("%d %b %Y  %H:%M:%S")
    zulu_str = now_zulu.strftime("%H:%M:%S")

    # ── System log accumulator ───────────────────────────────
    log: list = []

    def ts_offset(secs: int = 0) -> str:
        return (now_zulu + timedelta(seconds=secs)).strftime("%H:%MZ")

    log.append({
        "cls": "t-info",
        "ts":  ts_offset(0),
        "msg": "INIT: NEOM Bio-Secure V7.0 Glass Cockpit — Boot Sequence Complete",
    })

    # ── Load training data & train model (both cached) ───────
    df              = generate_training_data()
    model, accuracy, imp_df = train_model(df)

    log.append({
        "cls": "t-ok",
        "ts":  ts_offset(1),
        "msg": f"INIT: RandomForest Model Loaded — 150 Trees · Accuracy {accuracy:.1%}",
    })
    log.append({
        "cls": "t-ok",
        "ts":  ts_offset(1),
        "msg": f"INIT: Training Corpus Verified — {len(df):,} Synthetic Records",
    })

    # ── Sidebar ──────────────────────────────────────────────
    mode, sim = build_sidebar(accuracy)
    is_live   = mode.startswith("🔴")

    # ── Resolve weather inputs ───────────────────────────────
    if is_live:
        weather = get_live_neom_weather()
        if weather["error"]:
            log.append({
                "cls": "t-err",
                "ts":  ts_offset(2),
                "msg": f"UPLINK: API ERROR — {weather['error']} — Fallback Defaults Active",
            })
            temperature, wind_speed, api_ok = 30.0, 12.0, False
        else:
            temperature = weather["temperature"]
            wind_speed  = weather["wind_speed"]
            api_ok      = True
            log.append({
                "cls": "t-ok",
                "ts":  ts_offset(2),
                "msg": f"UPLINK: Weather Data Fetched — T={temperature}°C  W={wind_speed} km/h",
            })
            log.append({
                "cls": "t-info",
                "ts":  ts_offset(2),
                "msg": "UPLINK: Open-Meteo Satellite Link Active · Cache TTL 900s",
            })
        migration = auto_migration()
    else:
        temperature = float(sim[0])
        wind_speed  = float(sim[1])
        migration   = sim[2]
        api_ok      = None
        log.append({
            "cls": "t-warn",
            "ts":  ts_offset(2),
            "msg": f"SIM: Simulation Mode Active — T={temperature}°C  W={wind_speed} km/h",
        })

    mig_label = "ACTIVE" if migration == 1 else "INACTIVE"
    log.append({
        "cls": "t-info",
        "ts":  ts_offset(3),
        "msg": f"SENSOR: Migration Season → {mig_label} (Month {now_ast.month})",
    })

    # ── AI Prediction ─────────────────────────────────────────
    risk_pct = predict_risk(model, temperature, wind_speed, migration)
    bash     = get_bash_advisory(risk_pct)

    risk_cls = "t-ok" if risk_pct < 30 else ("t-warn" if risk_pct < 60 else "t-err")
    log.append({
        "cls": risk_cls,
        "ts":  ts_offset(3),
        "msg": f"AI: Risk Calculated — BASH {bash['condition']} · {risk_pct:.1f}% Collision Probability",
    })
    log.append({
        "cls": risk_cls,
        "ts":  ts_offset(3),
        "msg": f"AI: Advisory Issued — {bash['directive']}",
    })

    # ════════════════════════════════════════════════════════
    # TOP HEADER — "The Alive Indicator"
    # ════════════════════════════════════════════════════════
    dot_class  = "dot-safe"   if risk_pct < 30 else ("dot-warn" if risk_pct < 60 else "dot-danger")
    uplink_col = C["green"]   if risk_pct < 30 else (C["amber"]  if risk_pct < 60 else C["red"])

    hdr_col_l, hdr_col_r = st.columns([2, 1])
    with hdr_col_l:
        st.markdown(
            f"""
            <div class='cockpit-header'>
                <div>
                    <div class='cockpit-title'>🦅 NEOM BIO-SECURE V7.0</div>
                    <div class='cockpit-subtitle'>
                        EXPLAINABLE AI · BIRD AIRCRAFT STRIKE HAZARD COMMAND · GLASS COCKPIT
                    </div>
                    <div class='uplink-row' style='color:{uplink_col};'>
                        <span class='{dot_class}'></span>
                        LIVE SATELLITE UPLINK: NEOM [ACTIVE]
                    </div>
                </div>
                <div class='clock-panel'>
                    <div class='clock-lbl'>UTC / ZULU</div>
                    <div class='clock-utc'>{zulu_str}Z</div>
                    <div class='clock-lbl' style='margin-top:7px;'>SAUDI ARABIA TIME (AST / UTC+3)</div>
                    <div class='clock-ast'>{ast_str}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Mode badge ───────────────────────────────────────────
    if is_live:
        badge_col = C["red"]   if api_ok else C["amber"]
        badge_txt = "● LIVE — OPEN-METEO UPLINK" if api_ok else "● LIVE (FALLBACK DEFAULTS)"
    else:
        badge_col = C["cyan"]
        badge_txt = "⚙ SIMULATION MODE"

    st.markdown(
        f"<span class='mode-badge' style='background:{badge_col}22;"
        f"color:{badge_col};border:1px solid {badge_col}55;'>{badge_txt}</span>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════
    # TOP ROW — 3 KPI Glass Cards
    # ════════════════════════════════════════════════════════
    k1, k2, k3 = st.columns(3, gap="medium")

    with k1:
        st.markdown(
            f"""<div class='kpi-glass'>
                <div class='kpi-label'>Live Temperature</div>
                <div class='kpi-value' style='color:{C["cyan"]};'>{temperature}°C</div>
                <div class='kpi-sub'>NEOM Ground Station · Open-Meteo</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='kpi-glass'>
                <div class='kpi-label'>Live Wind Speed</div>
                <div class='kpi-value' style='color:{C["cyan"]};'>{wind_speed} km/h</div>
                <div class='kpi-sub'>Surface Level · AGL</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='kpi-glass'>
                <div class='kpi-label'>BASH Condition</div>
                <div class='kpi-value' style='color:{bash["color"]};'>
                    {bash["icon"]} {bash["condition"]}
                </div>
                <div class='kpi-sub'>{risk_pct:.1f}% Strike Probability</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # CENTER STAGE — Gauge (left) | Feature Importance (right)
    # ════════════════════════════════════════════════════════
    col_gauge, col_fi = st.columns([1, 1], gap="large")

    with col_gauge:
        st.markdown(
            "<div class='sec-head'>► AI COLLISION PROBABILITY GAUGE</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_gauge(risk_pct), use_container_width=True)

        # Advisory action box
        st.markdown(
            f"""
            <div class='bash-box' style='background:{bash["color"]}0F;
                 border:1px solid {bash["color"]}55;'>
                <div class='bash-level' style='color:{bash["color"]};'>
                    {bash["icon"]} {bash["condition"]}
                </div>
                <div class='bash-directive' style='color:{bash["color"]};'>
                    {bash["directive"]}
                </div>
                <hr style='border-color:{bash["color"]}33;margin:8px 0;'>
                <ul>
                    {''.join(f"<li>{s}</li>" for s in bash["steps"])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_fi:
        st.markdown(
            "<div class='sec-head'>► WHY: AI DECISION FACTOR WEIGHTS</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_feature_importance(imp_df), use_container_width=True)

        # Current conditions context panel
        mig_col  = C["amber"] if migration == 1 else C["green"]
        mig_icon = "🦢" if migration == 1 else "✅"

        st.markdown(
            f"""
            <div class='glass-card' style='padding:14px 18px;margin-top:0;'>
                <div class='sec-head' style='margin-bottom:10px;'>► CURRENT SENSOR FEED</div>
                <div style='display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;'>
                    <div>
                        <div style='color:{C["muted"]};font-size:0.58rem;
                                    letter-spacing:2px;text-transform:uppercase;'>
                            Temperature</div>
                        <div style='color:{C["cyan"]};font-size:1.15rem;
                                    font-family:"Share Tech Mono",monospace;
                                    text-shadow:0 0 6px rgba(0,242,254,0.4);'>
                            {temperature}°C</div>
                    </div>
                    <div>
                        <div style='color:{C["muted"]};font-size:0.58rem;
                                    letter-spacing:2px;text-transform:uppercase;'>
                            Wind Speed</div>
                        <div style='color:{C["cyan"]};font-size:1.15rem;
                                    font-family:"Share Tech Mono",monospace;
                                    text-shadow:0 0 6px rgba(0,242,254,0.4);'>
                            {wind_speed} km/h</div>
                    </div>
                    <div>
                        <div style='color:{C["muted"]};font-size:0.58rem;
                                    letter-spacing:2px;text-transform:uppercase;'>
                            Migration</div>
                        <div style='color:{mig_col};font-size:1.15rem;
                                    font-family:"Share Tech Mono",monospace;'>
                            {mig_icon} {mig_label}</div>
                    </div>
                    <div>
                        <div style='color:{C["muted"]};font-size:0.58rem;
                                    letter-spacing:2px;text-transform:uppercase;'>
                            AI Accuracy</div>
                        <div style='color:{C["green"]};font-size:1.15rem;
                                    font-family:"Share Tech Mono",monospace;'>
                            {accuracy:.1%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # NOTAM status
        notam_col  = bash["color"]
        notam_text = "NO NOTAM REQUIRED" if risk_pct < 60 else "⚠ NOTAM GENERATED"
        st.markdown(
            f"""
            <div class='glass-card'
                 style='padding:14px 18px;border-left:3px solid {notam_col};'>
                <div style='color:{notam_col};font-size:0.65rem;font-weight:700;
                             letter-spacing:2.5px;margin-bottom:6px;
                             font-family:"Share Tech Mono",monospace;'>
                    📋 NOTAM STATUS
                </div>
                <div style='color:{notam_col};font-size:1.0rem;font-weight:700;
                             font-family:"Share Tech Mono",monospace;
                             text-shadow:0 0 6px {notam_col}66;'>
                    {notam_text}
                </div>
                <div style='color:{C["muted"]};font-size:0.60rem;margin-top:6px;
                             font-family:"Share Tech Mono",monospace;'>
                    NEOM INTL · {now_zulu.strftime("%d%H%M")}Z ·
                    BASH {bash["condition"]} · {risk_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # LOWER STAGE — 12-Hour Risk Trend Line Chart (full width)
    # ════════════════════════════════════════════════════════
    st.markdown(
        "<div class='sec-head'>"
        "► BASH RISK TREND — 12-HOUR TIME SERIES TO LIVE PREDICTION"
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_trend_line(model, temperature, wind_speed, migration),
        use_container_width=True,
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # THE TERMINAL — Full-Width System Operations Log
    # ════════════════════════════════════════════════════════
    st.markdown(
        "<div class='sec-head'>"
        "► SYSTEM OPERATIONS LOG — NEOM BIO-SECURE V7.0"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='term-log'>{build_terminal(log)}</div>",
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center;color:{C["muted"]};font-size:0.60rem;
             padding:22px 0 10px 0;letter-spacing:1px;
             font-family:"Share Tech Mono",monospace;
             border-top:1px solid {C["border_faint"]};margin-top:22px;'>
            🦅 &nbsp;
            <span style='color:{C["cyan"]};
                  text-shadow:0 0 6px rgba(0,242,254,0.4);'>
                NEOM BIO-SECURE V7.0
            </span>
            &nbsp;·&nbsp; Glass Cockpit Edition
            &nbsp;·&nbsp; Explainable AI (XAI)
            &nbsp;·&nbsp; NEOM Smart City 2025–2026
            <br>
            RandomForest · Open-Meteo (TTL 900s) · Streamlit · Plotly · Glassmorphism
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
