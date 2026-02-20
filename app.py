"""
╔══════════════════════════════════════════════════════════════════════╗
║       NEOM Bio-Secure  —  Version 6.0  (Enterprise Aviation Ed.)    ║
║    Explainable AI (XAI) Bird Aircraft Strike Hazard Dashboard       ║
║    NEOM Smart City  |  Aviation Safety Division                     ║
╚══════════════════════════════════════════════════════════════════════╝
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

# ════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEOM Bio-Secure v6.0 | BASH Command",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────
NEOM_LAT = 28.2933
NEOM_LON = 35.0000
NEOM_TZ  = timezone(timedelta(hours=3))          # Saudi Arabia Time (AST)
ZULU_TZ  = timezone.utc                           # Zulu / UTC
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}           # Mar–May, Sep–Nov

# ── Colour palette ───────────────────────────────────────────────
C = {
    "bg":       "#000000",    # Absolute black
    "panel":    "#080C14",    # Near-black panel
    "card":     "#0B1120",    # Card fill
    "border":   "#0F2847",    # Subtle border
    "teal":     "#00D4AA",    # NEOM teal (primary)
    "green":    "#00FF88",    # Safe
    "amber":    "#FFD700",    # Caution
    "red":      "#FF4444",    # Danger
    "cyan":     "#00E5FF",    # Accent cyan
    "white":    "#FFFFFF",
    "muted":    "#7B8CA3",    # Muted text
    "grid":     "rgba(255,255,255,0.06)",
    "term_bg":  "#030810",    # Terminal background
}


# ════════════════════════════════════════════════════════════════════
# 2.  ATC DARKROOM CSS
# ════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(f"""
    <style>
    /* ── Hide Streamlit chrome ── */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{display: none;}}

    /* ── Base darkroom ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main .block-container {{
        background-color: {C["bg"]} !important;
        color: {C["white"]};
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    }}
    .block-container {{ padding-top: 1.5rem !important; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {C["panel"]} 0%, {C["card"]} 100%);
        border-right: 1px solid {C["border"]};
    }}
    [data-testid="stSidebar"] * {{ color: {C["white"]} !important; }}

    /* ── Header bar ── */
    .atc-header {{
        background: linear-gradient(135deg, {C["panel"]}, {C["card"]});
        border: 1px solid {C["border"]};
        border-left: 3px solid {C["teal"]};
        border-radius: 4px;
        padding: 16px 24px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .atc-header .title {{
        color: {C["teal"]};
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    .atc-header .subtitle {{
        color: {C["muted"]};
        font-size: 0.78rem;
        letter-spacing: 0.8px;
    }}
    .atc-clock {{
        text-align: right;
    }}
    .atc-clock .zulu {{
        color: {C["cyan"]};
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .atc-clock .ast {{
        color: {C["teal"]};
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .atc-clock .lbl {{
        color: {C["muted"]};
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    /* ── KPI card ── */
    .kpi {{
        background: {C["card"]};
        border: 1px solid {C["border"]};
        border-radius: 6px;
        padding: 16px 18px;
        text-align: center;
    }}
    .kpi .label {{
        color: {C["muted"]};
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }}
    .kpi .value {{
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.15;
    }}
    .kpi .sub {{
        color: {C["muted"]};
        font-size: 0.62rem;
        margin-top: 4px;
    }}

    /* ── Section head ── */
    .sec {{
        color: {C["teal"]};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid {C["border"]};
    }}

    /* ── BASH advisory box ── */
    .bash-box {{
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 10px;
    }}
    .bash-box .level {{
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .bash-box .action {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }}
    .bash-box ul {{
        margin: 0; padding-left: 16px;
    }}
    .bash-box li {{
        font-size: 0.78rem;
        margin-bottom: 3px;
        color: {C["white"]};
    }}

    /* ── Terminal log ── */
    .term-log {{
        background: {C["term_bg"]};
        border: 1px solid {C["border"]};
        border-radius: 6px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.72rem;
        line-height: 1.8;
        max-height: 340px;
        overflow-y: auto;
        color: {C["muted"]};
    }}
    .term-log .ok   {{ color: {C["green"]}; }}
    .term-log .warn {{ color: {C["amber"]}; }}
    .term-log .err  {{ color: {C["red"]}; }}
    .term-log .info {{ color: {C["cyan"]}; }}

    /* ── Mode pill ── */
    .mode-pill {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
        margin-right: 6px;
    }}

    /* ── Model meta card ── */
    .meta-card {{
        background: {C["card"]};
        border: 1px solid {C["border"]};
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 0.72rem;
        line-height: 1.9;
    }}
    .meta-card .k {{ color: {C["muted"]}; }}
    .meta-card .v {{ color: {C["teal"]}; font-weight: 600; }}

    /* ── Plotly ── */
    .js-plotly-plot .plotly {{ background: transparent !important; }}

    /* ── Streamlit widget overrides ── */
    .stSlider > div > div {{ background: {C["teal"]}44; }}
    hr {{ border-color: {C["border"]} !important; }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 3.  SYNTHETIC DATA  (5 000 rows, no CSV)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 5000
    dates   = pd.date_range("2022-01-01", periods=n, freq="H")
    months  = dates.month

    temp_base   = 28 + 14 * np.sin((months - 3) * np.pi / 6)
    temperature = np.clip(rng.normal(loc=temp_base, scale=4.5), 8, 52)
    wind_speed  = np.clip(rng.gamma(shape=2, scale=7, size=n), 0, 75)

    migration = np.where(np.isin(months, list(MIGRATION_MONTHS)), 1, 0)
    flip = rng.random(n) < 0.05
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


# ════════════════════════════════════════════════════════════════════
# 4.  RANDOM FOREST MODEL
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# 5.  LIVE WEATHER  —  OPEN-METEO  (15-min smart cache)
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_neom_weather() -> dict:
    """Cached 15 min to prevent 429 Too Many Requests."""
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
            "wind_speed":  round(float(cw["windspeed"]), 1),
            "source": "live", "error": None,
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


# ════════════════════════════════════════════════════════════════════
# 6.  BASH ADVISORY ENGINE  (Bird Aircraft Strike Hazard)
# ════════════════════════════════════════════════════════════════════

def get_bash_advisory(risk: float) -> dict:
    if risk >= 60:
        return {
            "condition": "CRITICAL",
            "directive": "GROUND AIRCRAFT — GENERATE NOTAM",
            "color":     C["red"],
            "icon":      "🚨",
            "steps": [
                "NOTAM: All departures & arrivals SUSPENDED",
                "Activate BASH deterrent array (full spectrum)",
                "Deploy runway inspection & bird dispersal units",
                "Notify NEOM ATC & inbound traffic for holding",
                "Continuous BASH radar sweep — 30s intervals",
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
                "Increase BASH visual scan to every 10 min",
                "Perimeter avian radar to HIGH-GAIN mode",
                "Weather advisory update to ATC ops",
                "Dispersal team STANDBY — alert status",
            ],
        }
    else:
        return {
            "condition": "CLEAR",
            "directive": "CLEAR FOR TAKEOFF — NORMAL OPS",
            "color":     C["green"],
            "icon":      "✅",
            "steps": [
                "BASH condition GREEN: standard clearance",
                "Routine pre-departure avian assessment",
                "Log environmental telemetry — nominal",
            ],
        }


# ════════════════════════════════════════════════════════════════════
# 7.  PLOTLY  —  XAI CHARTS
# ════════════════════════════════════════════════════════════════════

def make_gauge(risk_pct: float) -> go.Figure:
    if risk_pct < 30:
        bar, label = C["green"],  "BASH CLEAR"
    elif risk_pct < 60:
        bar, label = C["amber"],  "BASH WARNING"
    else:
        bar, label = C["red"],    "BASH CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 52, "color": bar,
                                         "family": "JetBrains Mono, monospace"}},
        title={
            "text": (
                f"<b>BASH Risk Index</b><br>"
                f"<span style='font-size:0.82em;color:{bar}'>{label}</span>"
            ),
            "font": {"size": 14, "color": C["white"]},
        },
        gauge={
            "axis": {"range": [0, 100],
                     "tickwidth": 1, "tickcolor": C["muted"],
                     "tickfont": {"color": C["muted"], "size": 10}},
            "bar":  {"color": bar, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(0,255,136,0.07)"},
                {"range": [30, 60], "color": "rgba(255,215,0,0.07)"},
                {"range": [60,100], "color": "rgba(255,68,68,0.07)"},
            ],
            "threshold": {
                "line": {"color": C["cyan"], "width": 3},
                "thickness": 0.78, "value": risk_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"]},
        margin={"t": 90, "b": 10, "l": 30, "r": 30},
        height=310,
    )
    return fig


def make_xai_scatter(df: pd.DataFrame,
                     live_temp: float, live_wind: float) -> go.Figure:
    """Full 5 000-point historical scatter + glowing LIVE overlay marker."""
    fig = go.Figure()

    # ── Historical: No-Risk points ──
    no_risk = df[df["Risk_Event"] == 0]
    fig.add_trace(go.Scattergl(
        x=no_risk["Temperature"], y=no_risk["Wind_Speed"],
        mode="markers",
        marker={"size": 3.5, "color": C["green"], "opacity": 0.35},
        name="Historical — No Risk",
        hovertemplate="Temp: %{x}°C<br>Wind: %{y} km/h<extra>No Risk</extra>",
    ))

    # ── Historical: Risk points ──
    risk = df[df["Risk_Event"] == 1]
    fig.add_trace(go.Scattergl(
        x=risk["Temperature"], y=risk["Wind_Speed"],
        mode="markers",
        marker={"size": 3.5, "color": C["red"], "opacity": 0.35},
        name="Historical — Risk Event",
        hovertemplate="Temp: %{x}°C<br>Wind: %{y} km/h<extra>Risk</extra>",
    ))

    # ── LIVE point: outer glow (large semi-transparent ring) ──
    fig.add_trace(go.Scatter(
        x=[live_temp], y=[live_wind],
        mode="markers",
        marker={"size": 32, "color": "rgba(255,255,255,0.12)",
                "line": {"width": 2, "color": "rgba(255,255,255,0.25)"}},
        name="_glow",
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── LIVE point: main star marker ──
    fig.add_trace(go.Scatter(
        x=[live_temp], y=[live_wind],
        mode="markers+text",
        marker={"size": 18, "color": C["white"],
                "symbol": "star",
                "line": {"width": 2, "color": C["cyan"]}},
        text=["LIVE NOW"],
        textposition="top center",
        textfont={"color": C["cyan"], "size": 11, "family": "monospace"},
        name="Current Conditions (LIVE)",
        hovertemplate=(
            f"<b>LIVE</b><br>"
            f"Temp: {live_temp}°C<br>"
            f"Wind: {live_wind} km/h<extra></extra>"
        ),
    ))

    fig.update_layout(
        title={"text": "<b>XAI: Live Conditions vs. 5,000 Historical Data Points</b>",
               "font": {"size": 13, "color": C["teal"]}},
        plot_bgcolor=C["panel"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"], "size": 11},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1,
                "font": {"size": 10}},
        xaxis={"title": "Temperature (°C)",
               "gridcolor": C["grid"], "zerolinecolor": C["grid"]},
        yaxis={"title": "Wind Speed (km/h)",
               "gridcolor": C["grid"], "zerolinecolor": C["grid"]},
        margin={"t": 55, "b": 40, "l": 50, "r": 20},
        height=420,
    )
    return fig


def make_feature_importance(imp_df: pd.DataFrame) -> go.Figure:
    colors = [C["teal"], C["amber"], C["red"]][:len(imp_df)]
    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"],
        orientation="h",
        marker={"color": colors,
                "line": {"color": "rgba(255,255,255,0.12)", "width": 1}},
        text=imp_df["Importance"].map("{:.1%}".format),
        textposition="outside",
        textfont={"color": C["white"], "size": 11},
    ))
    fig.update_layout(
        title={"text": "<b>XAI: Feature Decision Weights</b>",
               "font": {"size": 13, "color": C["teal"]}},
        plot_bgcolor=C["panel"],
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": C["white"]},
        xaxis={"gridcolor": C["grid"], "range": [0, 0.75],
               "tickformat": ".0%"},
        yaxis={"autorange": "reversed"},
        margin={"t": 45, "b": 15, "l": 10, "r": 60},
        height=220,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# 8.  SYSTEM ACTION LOG  (terminal-style)
# ════════════════════════════════════════════════════════════════════

def build_action_log(entries: list[dict]) -> str:
    """Render a list of {cls, msg} dicts into HTML terminal lines."""
    lines = []
    for e in entries:
        cls = e.get("cls", "")
        msg = e.get("msg", "")
        lines.append(f"<span class='{cls}'>&gt; {msg}</span>")
    return "<br>".join(lines)


# ════════════════════════════════════════════════════════════════════
# 9.  SIDEBAR
# ════════════════════════════════════════════════════════════════════

def build_sidebar(accuracy: float):
    st.sidebar.markdown(
        f"""
        <div style='text-align:center;padding:6px 0 14px 0;'>
            <span style='font-size:2rem;'>🦅</span><br>
            <span style='color:{C["teal"]};font-size:1rem;font-weight:700;
                         letter-spacing:2px;'>NEOM BIO-SECURE</span><br>
            <span style='color:{C["muted"]};font-size:0.65rem;
                         letter-spacing:1px;'>
                v6.0 ENTERPRISE AVIATION EDITION
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
            f"<div class='sec'>📡 NEOM STATION</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            f"<small style='color:{C['muted']};'>"
            f"Lat {NEOM_LAT}° N · Lon {NEOM_LON}° E<br>"
            f"Source: Open-Meteo (free tier)<br>"
            f"Cache: 15 min TTL (anti-429)</small>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        sim = None, None, None
    else:
        st.sidebar.markdown(
            f"<div class='sec'>🎛️ SIMULATION CONTROLS</div>",
            unsafe_allow_html=True,
        )
        s_temp = st.sidebar.slider("🌡️ Temperature (°C)", 5, 52, 30)
        s_wind = st.sidebar.slider("💨 Wind Speed (km/h)", 0, 75, 15)
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

    # ── Model metadata card ──
    st.sidebar.divider()
    st.sidebar.markdown(
        f"<div class='sec'>🤖 MODEL METADATA</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class='meta-card'>
            <span class='k'>Engine:</span>
            <span class='v'>RandomForestClassifier</span><br>
            <span class='k'>Ensemble:</span>
            <span class='v'>150 Trees</span><br>
            <span class='k'>Max Depth:</span>
            <span class='v'>12</span><br>
            <span class='k'>Training Size:</span>
            <span class='v'>5,000 samples</span><br>
            <span class='k'>Accuracy:</span>
            <span class='v'>{accuracy:.1%}</span><br>
            <span class='k'>Features:</span>
            <span class='v'>Temp · Wind · Migration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return mode, sim


# ════════════════════════════════════════════════════════════════════
# 10.  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Timestamps ──────────────────────────────────────────────
    now_ast  = datetime.now(tz=NEOM_TZ)
    now_zulu = datetime.now(tz=ZULU_TZ)
    ast_str  = now_ast.strftime("%d %b %Y  %H:%M:%S")
    zulu_str = now_zulu.strftime("%H:%M:%S")

    # ── System log accumulator ──────────────────────────────────
    log = []
    log.append({"cls": "info", "msg": f"System boot — {ast_str} AST"})

    # ── Train model (cached) ────────────────────────────────────
    df = generate_training_data()
    model, accuracy, imp_df = train_model(df)
    log.append({"cls": "ok",   "msg": f"AI Model loaded — RandomForest 150T — Accuracy {accuracy:.1%}"})
    log.append({"cls": "ok",   "msg": f"Training corpus: {len(df):,} synthetic records"})

    # ── Sidebar ─────────────────────────────────────────────────
    mode, sim = build_sidebar(accuracy)
    is_live = mode.startswith("🔴")

    # ── Resolve weather inputs ──────────────────────────────────
    if is_live:
        weather = get_live_neom_weather()
        if weather["error"]:
            log.append({"cls": "err",
                        "msg": f"API FAIL: {weather['error']} — using fallback defaults"})
            temperature, wind_speed, api_ok = 30.0, 12.0, False
        else:
            temperature = weather["temperature"]
            wind_speed  = weather["wind_speed"]
            api_ok = True
            log.append({"cls": "ok",
                        "msg": f"Live weather acquired — T={temperature}°C  W={wind_speed} km/h"})
            log.append({"cls": "info",
                        "msg": "Source: Open-Meteo API · Cache TTL 900s"})
        migration = auto_migration()
    else:
        temperature = float(sim[0])
        wind_speed  = float(sim[1])
        migration   = sim[2]
        api_ok      = None
        log.append({"cls": "warn",
                    "msg": f"SIMULATION MODE — T={temperature}°C  W={wind_speed} km/h"})

    mig_label = "ACTIVE" if migration == 1 else "INACTIVE"
    log.append({"cls": "info",
                "msg": f"Migration season: {mig_label} (month {now_ast.month})"})

    # ── Prediction ──────────────────────────────────────────────
    risk_pct = predict_risk(model, temperature, wind_speed, migration)
    bash     = get_bash_advisory(risk_pct)
    log.append({"cls": "ok",
                "msg": f"Prediction generated — BASH Risk {risk_pct:.1f}%"})
    log.append({"cls": "warn" if risk_pct >= 30 else "ok",
                "msg": f"BASH Condition: {bash['condition']} — {bash['directive']}"})

    # ════════════════════════════════════════════════════════════
    # HEADER BAR  (title + dual clock)
    # ════════════════════════════════════════════════════════════
    st.markdown(
        f"""
        <div class='atc-header'>
            <div>
                <div class='title'>🦅 NEOM BIO-SECURE</div>
                <div class='subtitle'>
                    EXPLAINABLE AI (XAI) · BIRD AIRCRAFT STRIKE HAZARD COMMAND
                    &nbsp;·&nbsp; v6.0 ENTERPRISE
                </div>
            </div>
            <div class='atc-clock'>
                <div class='lbl'>ZULU (UTC)</div>
                <div class='zulu'>{zulu_str}Z</div>
                <div class='lbl' style='margin-top:4px;'>SAUDI (AST / UTC+3)</div>
                <div class='ast'>{ast_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Mode pill ───────────────────────────────────────────────
    if is_live:
        pill_c = C["red"] if api_ok else C["amber"]
        pill_t = "● LIVE — OPEN-METEO" if api_ok else "● LIVE (FALLBACK)"
    else:
        pill_c = C["teal"]
        pill_t = "⚙ SIMULATION"
    st.markdown(
        f"<span class='mode-pill' style='background:{pill_c}22;"
        f"color:{pill_c};border:1px solid {pill_c}55;'>{pill_t}</span>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════
    # TOP ROW  —  3 KPI Cards
    # ════════════════════════════════════════════════════════════
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(
            f"""<div class='kpi'>
                <div class='label'>Live Temperature</div>
                <div class='value' style='color:{C["teal"]};'>{temperature}°C</div>
                <div class='sub'>NEOM Ground Station</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class='kpi'>
                <div class='label'>Live Wind Speed</div>
                <div class='value' style='color:{C["teal"]};'>{wind_speed} km/h</div>
                <div class='sub'>Surface Level · AGL</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class='kpi'>
                <div class='label'>BASH Condition</div>
                <div class='value' style='color:{bash["color"]};'>
                    {bash["icon"]} {bash["condition"]}
                </div>
                <div class='sub'>{risk_pct:.1f}% Strike Probability</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # MIDDLE ROW  —  Gauge (left)  |  XAI Scatter (right)
    # ════════════════════════════════════════════════════════════
    col_gauge, col_scatter = st.columns([1, 2], gap="large")

    with col_gauge:
        st.markdown("<div class='sec'>BASH RISK GAUGE</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(risk_pct), use_container_width=True)

        # Advisory box
        st.markdown(
            f"""
            <div class='bash-box' style='background:{bash["color"]}12;
                 border:1px solid {bash["color"]}44;'>
                <div class='level' style='color:{bash["color"]};'>
                    {bash["icon"]} {bash["condition"]}
                </div>
                <div class='action' style='color:{bash["color"]};'>
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

    with col_scatter:
        st.markdown("<div class='sec'>XAI: LIVE vs. HISTORICAL ANALYSIS</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_xai_scatter(df, temperature, wind_speed),
            use_container_width=True,
        )
        # Live-mode context bar
        if is_live:
            mig_color = C["amber"] if migration == 1 else C["green"]
            st.markdown(
                f"""
                <div style='background:{C["card"]};border:1px solid {C["border"]};
                     border-radius:4px;padding:10px 14px;font-size:0.72rem;'>
                    📡 &nbsp;<b style='color:{C["teal"]}'>API:</b>
                    Open-Meteo · cached 15 min
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    📅 &nbsp;<b style='color:{C["teal"]}'>Migration:</b>
                    <span style='color:{mig_color};font-weight:700;'>
                        {mig_label}
                    </span> (month {now_ast.month})
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    🎯 &nbsp;<b style='color:{C["teal"]}'>LIVE point</b>
                    overlaid on 5,000 records
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # BOTTOM ROW  —  Feature Importance (left)  |  System Log (right)
    # ════════════════════════════════════════════════════════════
    col_fi, col_log = st.columns([1, 1], gap="large")

    with col_fi:
        st.markdown("<div class='sec'>XAI: MODEL DECISION WEIGHTS</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_feature_importance(imp_df),
                        use_container_width=True)

        # Compact model stats
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""<div class='kpi' style='padding:10px 12px;'>
                    <div class='label'>Accuracy</div>
                    <div class='value' style='color:{C["teal"]};
                         font-size:1.2rem;'>{accuracy:.1%}</div>
                    <div class='sub'>Test Set</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class='kpi' style='padding:10px 12px;'>
                    <div class='label'>Ensemble</div>
                    <div class='value' style='color:{C["teal"]};
                         font-size:1.2rem;'>150</div>
                    <div class='sub'>Decision Trees</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class='kpi' style='padding:10px 12px;'>
                    <div class='label'>Corpus</div>
                    <div class='value' style='color:{C["teal"]};
                         font-size:1.2rem;'>5,000</div>
                    <div class='sub'>Training Records</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_log:
        st.markdown("<div class='sec'>SYSTEM ACTION LOG</div>",
                    unsafe_allow_html=True)
        log_html = build_action_log(log)
        st.markdown(
            f"<div class='term-log'>{log_html}</div>",
            unsafe_allow_html=True,
        )

        # NOTAM generation box
        notam_color = bash["color"]
        notam_text  = "NO NOTAM REQUIRED" if risk_pct < 60 else "NOTAM GENERATED"
        st.markdown(
            f"""
            <div style='margin-top:12px;background:{C["card"]};
                 border:1px solid {notam_color}44;border-left:3px solid {notam_color};
                 border-radius:4px;padding:12px 16px;'>
                <div style='color:{notam_color};font-size:0.72rem;
                     font-weight:700;letter-spacing:1px;margin-bottom:6px;'>
                    📋 NOTAM STATUS
                </div>
                <div style='color:{notam_color};font-size:0.88rem;
                     font-weight:700;'>{notam_text}</div>
                <div style='color:{C["muted"]};font-size:0.65rem;
                     margin-top:4px;'>
                    NEOM INTL · {now_zulu.strftime("%d%H%M")}Z ·
                    BASH {bash["condition"]} · {risk_pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════════════
    st.markdown(f"<hr style='margin-top:24px;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='text-align:center;color:{C["muted"]};font-size:0.65rem;
             padding:4px 0 16px 0;letter-spacing:0.5px;'>
            🦅 &nbsp;<b style='color:{C["teal"]};'>NEOM BIO-SECURE v6.0</b>
            &nbsp;·&nbsp; Enterprise Aviation Edition
            &nbsp;·&nbsp; Explainable AI (XAI)
            &nbsp;·&nbsp; NEOM Smart City 2025–2026
            <br>
            RandomForest · Open-Meteo (cached 900s) · Streamlit · Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
