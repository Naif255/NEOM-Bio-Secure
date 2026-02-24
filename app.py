"""
NEOM Bio-Secure V8.0 — Midnight Bio-Radar Dashboard
Bird Aircraft Strike Hazard (BASH) Decision Support System
NEOM Smart City | Aviation Safety Division

Architecture: Native Streamlit columns + plotly.graph_objects (template="plotly_dark").
CSS injection for NEOM Midnight theme, hidden chrome, animated ticker.
PyDeck interactive map with dynamic threat vectors.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
import pydeck as pdk
from datetime import datetime, timezone, timedelta
import math

# ─── PAGE CONFIG (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="NEOM Bio-Secure V8.0 | BASH ATC",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CONSTANTS ───────────────────────────────────────────────────────
NEOM_LAT = 28.03
NEOM_LON = 34.64
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}

# ─── COLOR PALETTE: NEOM Midnight Bio-Radar ─────────────────────────
CLR_BG = "#0A0F1D"
CLR_SURFACE = "#0F1729"
CLR_CARD = "#111B30"
CLR_CYAN = "#00E5FF"
CLR_GREEN = "#00E676"
CLR_AMBER = "#FFAB00"
CLR_RED = "#FF1744"
CLR_TEXT = "#E0E6ED"
CLR_DIM = "#5A6A7A"
CLR_BORDER = "#1A2744"


# ═════════════════════════════════════════════════════════════════════
# CSS INJECTION — Midnight theme, hidden chrome, animated ticker
# ═════════════════════════════════════════════════════════════════════

CUSTOM_CSS = f"""
<style>
    /* Global background */
    .stApp {{
        background-color: {CLR_BG};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CLR_SURFACE};
    }}

    /* Hide Streamlit header, menu, footer */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    #MainMenu {{
        display: none !important;
    }}
    footer {{
        display: none !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Monospace for metrics */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
        color: {CLR_CYAN} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {CLR_DIM} !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 1.5px;
    }}

    /* Ticker animation at absolute bottom */
    .ticker-wrap {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        overflow: hidden;
        background: linear-gradient(90deg, {CLR_BG}, {CLR_SURFACE}, {CLR_BG});
        border-top: 1px solid {CLR_BORDER};
        z-index: 9999;
        height: 32px;
        display: flex;
        align-items: center;
    }}
    .ticker-content {{
        display: inline-block;
        white-space: nowrap;
        animation: ticker-slide 45s linear infinite;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.72rem;
        color: {CLR_CYAN};
        letter-spacing: 1.2px;
        padding-left: 100%;
    }}
    @keyframes ticker-slide {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-100%); }}
    }}

    /* Header container */
    .cmd-header {{
        background: linear-gradient(135deg, {CLR_SURFACE} 0%, {CLR_CARD} 100%);
        border: 1px solid {CLR_BORDER};
        border-radius: 6px;
        padding: 12px 20px;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .cmd-header .logo-text {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {CLR_CYAN};
        letter-spacing: 2px;
    }}
    .cmd-header .status-text {{
        font-size: 0.82rem;
        color: {CLR_GREEN};
        letter-spacing: 1px;
    }}
    .cmd-header .time-text {{
        font-size: 0.82rem;
        color: {CLR_DIM};
        letter-spacing: 1px;
    }}

    /* Sensor card styling */
    .sensor-card {{
        background: {CLR_CARD};
        border: 1px solid {CLR_BORDER};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
    }}
    .sensor-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: {CLR_CYAN};
        line-height: 1.1;
    }}
    .sensor-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: {CLR_DIM};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}
    .sensor-status {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: {CLR_GREEN};
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* Section headers */
    .section-title {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: {CLR_DIM};
        letter-spacing: 2.5px;
        text-transform: uppercase;
        border-bottom: 1px solid {CLR_BORDER};
        padding-bottom: 6px;
        margin-bottom: 12px;
    }}

    /* Bottom padding so content doesn't hide behind ticker */
    .main .block-container {{
        padding-bottom: 50px !important;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# DATA GENERATION — 5000 synthetic rows for Risk 0-100%
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    month = dates.month

    temp = np.clip(25 + 18 * np.sin((dates.dayofyear - 80) * 2 * np.pi / 365)
                   + np.random.normal(0, 4, n), 5, 52)
    wind = np.clip(np.random.gamma(2.5, 7.5, n), 0, 75)
    season = np.isin(month, list(MIGRATION_MONTHS)).astype(float)

    # Continuous risk 0-100
    base = 15.0
    risk = (
        base
        + season * 30.0
        + np.where((temp >= 20) & (temp <= 38), 15.0, 0.0)
        + np.where(wind < 18, 12.0, 0.0)
        - np.where(wind > 55, 10.0, 0.0)
        + np.random.normal(0, 8, n)
    )
    risk = np.clip(risk, 0, 100)

    return pd.DataFrame({
        "temperature": temp,
        "wind_speed": wind,
        "migration_season": season,
        "risk_pct": risk,
    })


# ═════════════════════════════════════════════════════════════════════
# MODEL TRAINING — Random Forest Regressor -> Risk 0-100%
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model():
    df = generate_training_data()
    X = df[["temperature", "wind_speed", "migration_season"]]
    y = df["risk_pct"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42,
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=14,
        min_samples_leaf=8, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    score = model.score(X_te, y_te)

    importances = model.feature_importances_
    feat_names = ["Temperature", "Wind Speed", "Migration Season"]
    imp_df = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).reset_index(drop=True)

    return model, score, imp_df


# ═════════════════════════════════════════════════════════════════════
# LIVE WEATHER — Open-Meteo: Current + 12-Hour Hourly Forecast
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_weather():
    """Fetch current weather AND next 12 hours hourly forecast from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={NEOM_LAT}&longitude={NEOM_LON}"
        f"&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        f"&hourly=temperature_2m,wind_speed_10m"
        f"&forecast_hours=13"
        f"&wind_speed_unit=kmh&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        current = data["current"]
        cur_temp = round(current["temperature_2m"], 1)
        cur_wind = round(current["wind_speed_10m"], 1)
        cur_dir = current.get("wind_direction_10m", 0)

        hourly = data["hourly"]
        # Take next 12 hours (skip index 0 which is the current hour)
        h_times = hourly["time"][1:13]
        h_temps = [round(t, 1) for t in hourly["temperature_2m"][1:13]]
        h_winds = [round(w, 1) for w in hourly["wind_speed_10m"][1:13]]

        forecast = {
            "times": h_times,
            "temps": h_temps,
            "winds": h_winds,
        }

        return cur_temp, cur_wind, cur_dir, forecast, None

    except Exception as exc:
        return None, None, None, None, str(exc)


# ═════════════════════════════════════════════════════════════════════
# PREDICTION HELPER
# ═════════════════════════════════════════════════════════════════════

def predict_risk(model, temp, wind, mig):
    """Predict risk 0-100% from model."""
    val = model.predict([[temp, wind, mig]])[0]
    return float(np.clip(val, 0, 100))


def is_migration(month: int) -> int:
    return 1 if month in MIGRATION_MONTHS else 0


def risk_color(risk):
    if risk < 30:
        return CLR_GREEN
    elif risk <= 70:
        return CLR_AMBER
    else:
        return CLR_RED


def risk_label(risk):
    if risk < 30:
        return "SAFE"
    elif risk <= 70:
        return "WARNING"
    else:
        return "DANGER"


# ═════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════════════

def make_risk_gauge(risk: float) -> go.Figure:
    """Massive Plotly Gauge — AI Risk 0-100%."""
    color = risk_color(risk)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={
            "suffix": "%",
            "font": {"size": 62, "color": color, "family": "JetBrains Mono, monospace"},
            "valueformat": ".1f",
        },
        title={
            "text": f"<b>BASH RISK INDEX</b><br><span style='font-size:12px;color:{CLR_DIM}'>{risk_label(risk)}</span>",
            "font": {"size": 14, "color": CLR_DIM, "family": "JetBrains Mono, monospace"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": CLR_BORDER,
                "tickfont": {"size": 10, "color": CLR_DIM, "family": "monospace"},
                "dtick": 10,
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#0D1321",
            "borderwidth": 1,
            "bordercolor": CLR_BORDER,
            "steps": [
                {"range": [0, 30], "color": "rgba(0,230,118,0.06)"},
                {"range": [30, 70], "color": "rgba(255,171,0,0.06)"},
                {"range": [70, 100], "color": "rgba(255,23,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": CLR_RED, "width": 3},
                "thickness": 0.85,
                "value": 70,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(t=90, b=10, l=30, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
    )
    return fig


def make_feature_chart(imp_df: pd.DataFrame) -> go.Figure:
    """Compact horizontal bar — XAI Feature Importance."""
    colors = []
    for f in imp_df["Feature"]:
        if "Migration" in f:
            colors.append(CLR_CYAN)
        elif "Wind" in f:
            colors.append(CLR_AMBER)
        else:
            colors.append(CLR_RED)

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"] * 100,
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color=colors, width=1)),
        text=[f"{v*100:.1f}%" for v in imp_df["Importance"]],
        textposition="outside",
        textfont=dict(size=11, color=CLR_TEXT, family="monospace"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(t=10, b=10, l=10, r=70),
        xaxis=dict(
            range=[0, imp_df["Importance"].max() * 100 * 1.5],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10, color=CLR_DIM),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=CLR_TEXT, family="monospace"),
        ),
        bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_forecast_chart(hours_labels, risk_values) -> go.Figure:
    """12-hour forecast line chart with danger threshold highlighting."""
    # Determine line color segments
    colors = [risk_color(r) for r in risk_values]

    fig = go.Figure()

    # Fill area under curve
    fig.add_trace(go.Scatter(
        x=hours_labels,
        y=risk_values,
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=hours_labels,
        y=risk_values,
        mode="lines+markers",
        line=dict(color=CLR_CYAN, width=2.5),
        marker=dict(
            size=8,
            color=colors,
            line=dict(color=CLR_BG, width=1.5),
        ),
        name="Predicted Risk",
        hovertemplate="%{x}<br>Risk: %{y:.1f}%<extra></extra>",
    ))

    # Danger segments overlay
    danger_x = []
    danger_y = []
    for i, r in enumerate(risk_values):
        if r > 70:
            danger_x.append(hours_labels[i])
            danger_y.append(r)
        else:
            if danger_x:
                fig.add_trace(go.Scatter(
                    x=danger_x, y=danger_y,
                    mode="lines+markers",
                    line=dict(color=CLR_RED, width=3.5),
                    marker=dict(size=9, color=CLR_RED,
                                line=dict(color="#FFFFFF", width=1)),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                danger_x, danger_y = [], []
    if danger_x:
        fig.add_trace(go.Scatter(
            x=danger_x, y=danger_y,
            mode="lines+markers",
            line=dict(color=CLR_RED, width=3.5),
            marker=dict(size=9, color=CLR_RED,
                        line=dict(color="#FFFFFF", width=1)),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Threshold lines
    fig.add_hline(
        y=30, line=dict(color=CLR_AMBER, dash="dash", width=1), opacity=0.5,
        annotation_text="WARN 30%",
        annotation_font=dict(size=9, color=CLR_AMBER, family="monospace"),
        annotation_position="right",
    )
    fig.add_hline(
        y=70, line=dict(color=CLR_RED, dash="dash", width=1), opacity=0.6,
        annotation_text="DANGER 70%",
        annotation_font=dict(size=9, color=CLR_RED, family="monospace"),
        annotation_position="right",
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(t=20, b=50, l=50, r=80),
        xaxis=dict(
            title="Forecast Horizon",
            tickangle=-45,
            tickfont=dict(size=9, color=CLR_DIM, family="monospace"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title="Risk %",
            range=[0, 105],
            ticksuffix="%",
            tickfont=dict(size=10, color=CLR_DIM, family="monospace"),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        legend=dict(
            orientation="h", y=-0.2,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=CLR_DIM),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# PYDECK MAP BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_pydeck_map(risk: float, wind_dir: float):
    """Interactive dark-themed PyDeck map centered on NEOM with threat vectors."""
    rc = risk_color(risk)
    # Convert hex color to RGB list
    if rc == CLR_GREEN:
        vec_rgb = [0, 230, 118, 200]
        glow_rgb = [0, 230, 118, 60]
    elif rc == CLR_AMBER:
        vec_rgb = [255, 171, 0, 200]
        glow_rgb = [255, 171, 0, 60]
    else:
        vec_rgb = [255, 23, 68, 220]
        glow_rgb = [255, 23, 68, 80]

    # Airport marker
    airport_data = pd.DataFrame({
        "lat": [NEOM_LAT],
        "lon": [NEOM_LON],
        "name": ["NEOM INTL (OEGN)"],
        "risk": [f"{risk:.1f}%"],
    })

    # Generate wind/threat vector lines emanating from airport
    # Directions based on wind_dir with spread
    num_vectors = 8
    vec_data = []
    base_angle = math.radians(wind_dir) if wind_dir else 0

    for i in range(num_vectors):
        angle = base_angle + (i * 2 * math.pi / num_vectors)
        # Scale vector length by risk level
        length = 0.08 + (risk / 100) * 0.15
        end_lat = NEOM_LAT + length * math.cos(angle)
        end_lon = NEOM_LON + length * math.sin(angle)

        vec_data.append({
            "start_lat": NEOM_LAT,
            "start_lon": NEOM_LON,
            "end_lat": end_lat,
            "end_lon": end_lon,
        })

    vec_df = pd.DataFrame(vec_data)

    # Additional threat approach vectors (longer arcs representing bird migration paths)
    arc_data = []
    migration_bearings = [0, 45, 135, 180, 225, 315]  # Common migration directions
    for bearing in migration_bearings:
        angle = math.radians(bearing)
        # Source far away, target is airport
        src_lat = NEOM_LAT + 0.35 * math.cos(angle)
        src_lon = NEOM_LON + 0.35 * math.sin(angle)
        arc_data.append({
            "src_lat": src_lat,
            "src_lon": src_lon,
            "dst_lat": NEOM_LAT,
            "dst_lon": NEOM_LON,
        })

    arc_df = pd.DataFrame(arc_data)

    # Cyan for safe, Red for danger
    safe_color = [0, 229, 255, 180]
    map_vec_color = vec_rgb

    layers = [
        # Glow ring around airport
        pdk.Layer(
            "ScatterplotLayer",
            data=airport_data,
            get_position="[lon, lat]",
            get_radius=2500,
            get_fill_color=glow_rgb,
            pickable=False,
            opacity=0.4,
        ),
        # Airport marker
        pdk.Layer(
            "ScatterplotLayer",
            data=airport_data,
            get_position="[lon, lat]",
            get_radius=600,
            get_fill_color=map_vec_color,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=2,
            pickable=True,
            stroked=True,
        ),
        # Wind/threat direction lines
        pdk.Layer(
            "LineLayer",
            data=vec_df,
            get_source_position="[start_lon, start_lat]",
            get_target_position="[end_lon, end_lat]",
            get_color=map_vec_color,
            get_width=3,
            pickable=False,
        ),
        # Migration arc vectors
        pdk.Layer(
            "ArcLayer",
            data=arc_df,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[dst_lon, dst_lat]",
            get_source_color=safe_color if risk < 30 else map_vec_color,
            get_target_color=map_vec_color,
            get_width=2,
            pickable=False,
        ),
    ]

    view_state = pdk.ViewState(
        latitude=NEOM_LAT,
        longitude=NEOM_LON,
        zoom=10.5,
        pitch=45,
        bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={"text": "{name}\nRisk: {risk}"},
    )


# ═════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════

def main():
    # ── Timestamps ───────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    # ── Train model ──────────────────────────────────────────────────
    model, r2_score, imp_df = train_model()

    # ── Fetch live weather ───────────────────────────────────────────
    cur_temp, cur_wind, cur_dir, forecast, api_err = get_live_weather()

    # Fallback values if API fails
    if cur_temp is None:
        cur_temp = 33.5
        cur_wind = 14.2
        cur_dir = 180
        data_source = "FALLBACK"
    else:
        data_source = "OPEN-METEO"

    mig = is_migration(now_utc.month)

    # ── Predict current risk ─────────────────────────────────────────
    live_risk = predict_risk(model, cur_temp, cur_wind, mig)

    # ═════════════════════════════════════════════════════════════════
    # A. TOP COMMAND HEADER
    # ═════════════════════════════════════════════════════════════════
    hdr_left, hdr_center, hdr_right = st.columns([4, 4, 4])

    with hdr_left:
        st.markdown(
            f'<div class="cmd-header">'
            f'<span class="logo-text">⬡ NEOM BIO-SECURE ATC</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hdr_center:
        uplink_color = CLR_GREEN if api_err is None else CLR_AMBER
        st.markdown(
            f'<div class="cmd-header">'
            f'<span class="status-text" style="color:{uplink_color}">'
            f'&#x1F7E2; LIVE UPLINK: {"ACTIVE" if api_err is None else "DEGRADED"}'
            f'</span>'
            f'<br><span style="font-size:0.7rem;color:{CLR_DIM};font-family:monospace;">'
            f'SRC: {data_source} | LAT {NEOM_LAT}N LON {NEOM_LON}E'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hdr_right:
        st.markdown(
            f'<div class="cmd-header">'
            f'<span class="time-text">LAST SYNC: {utc_str} ZULU</span>'
            f'<br><span style="font-size:0.7rem;color:{CLR_DIM};font-family:monospace;">'
            f'MODEL: RF-200 | R2: {r2_score:.3f} | V8.0'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ═════════════════════════════════════════════════════════════════
    # B. MAIN SPLIT — [4, 6] Columns
    # ═════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([4, 6])

    # ─── LEFT COLUMN: "The Expert Core" ──────────────────────────────
    with left_col:
        st.markdown('<div class="section-title">SENSOR TELEMETRY — LIVE</div>',
                    unsafe_allow_html=True)

        # Sensor Data: Temp & Wind
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f'<div class="sensor-card">'
                f'<div class="sensor-label">TEMPERATURE</div>'
                f'<div class="sensor-value">{cur_temp:.1f} °C</div>'
                f'<div class="sensor-status">&#x1F7E2; SENSOR: ONLINE</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="sensor-card">'
                f'<div class="sensor-label">WIND SPEED</div>'
                f'<div class="sensor-value">{cur_wind:.1f} km/h</div>'
                f'<div class="sensor-status">&#x1F7E2; SENSOR: ONLINE</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Additional quick stats row
        qs1, qs2 = st.columns(2)
        with qs1:
            mig_text = "ACTIVE" if mig else "INACTIVE"
            mig_clr = CLR_AMBER if mig else CLR_GREEN
            st.markdown(
                f'<div class="sensor-card">'
                f'<div class="sensor-label">MIGRATION SEASON</div>'
                f'<div class="sensor-value" style="font-size:1.4rem;color:{mig_clr}">{mig_text}</div>'
                f'<div class="sensor-status">MONTH {now_utc.month} / SEASON AUTO-DETECT</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with qs2:
            dir_text = f"{cur_dir:.0f}°" if cur_dir is not None else "N/A"
            st.markdown(
                f'<div class="sensor-card">'
                f'<div class="sensor-label">WIND DIRECTION</div>'
                f'<div class="sensor-value" style="font-size:1.4rem">{dir_text}</div>'
                f'<div class="sensor-status">&#x1F7E2; ANEMOMETER: ACTIVE</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Risk Gauge — Massive
        st.markdown('<div class="section-title">AI RISK ASSESSMENT</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_risk_gauge(live_risk),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Explainable AI — Feature Importance
        st.markdown('<div class="section-title">EXPLAINABLE AI — FEATURE IMPORTANCE</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_feature_chart(imp_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ─── RIGHT COLUMN: "The Visual Twin" ────────────────────────────
    with right_col:
        st.markdown('<div class="section-title">NEOM THREAT VECTOR MAP — INTERACTIVE</div>',
                    unsafe_allow_html=True)

        # Risk status banner
        rc = risk_color(live_risk)
        rl = risk_label(live_risk)
        st.markdown(
            f'<div style="background:{CLR_CARD};border:1px solid {rc};border-radius:6px;'
            f'padding:10px 16px;margin-bottom:8px;font-family:monospace;">'
            f'<span style="color:{rc};font-size:1.1rem;font-weight:700;">'
            f'THREAT LEVEL: {rl} — {live_risk:.1f}%</span>'
            f'<span style="color:{CLR_DIM};font-size:0.75rem;float:right;">'
            f'VECTORS: {"CYAN/SAFE" if live_risk < 30 else "RED/DANGER" if live_risk > 70 else "AMBER/CAUTION"}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # PyDeck Map
        deck_map = build_pydeck_map(live_risk, cur_dir if cur_dir else 0)
        st.pydeck_chart(deck_map, height=560)

    # ═════════════════════════════════════════════════════════════════
    # C. BOTTOM SECTION — "The Horizon"
    # ═════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-title">12-HOUR PREDICTIVE HORIZON — AI FORECAST</div>',
                unsafe_allow_html=True)

    # Build forecast risk values
    if forecast is not None and len(forecast["temps"]) > 0:
        # Use actual API hourly forecast data
        forecast_risks = []
        forecast_labels = []
        for i in range(len(forecast["temps"])):
            f_temp = forecast["temps"][i]
            f_wind = forecast["winds"][i]
            f_risk = predict_risk(model, f_temp, f_wind, mig)
            forecast_risks.append(f_risk)

            # Parse time label
            try:
                t = datetime.fromisoformat(forecast["times"][i])
                forecast_labels.append(t.strftime("%H:%M"))
            except Exception:
                forecast_labels.append(f"+{i+1}h")

        # Pad to 12 if needed
        while len(forecast_risks) < 12:
            forecast_risks.append(forecast_risks[-1] if forecast_risks else live_risk)
            forecast_labels.append(f"+{len(forecast_labels)+1}h")
    else:
        # Fallback: synthetic forecast
        rng = np.random.default_rng(int(now_utc.timestamp()) % (2**31))
        forecast_labels = [f"+{i+1}h" for i in range(12)]
        forecast_risks = []
        for i in range(12):
            f_temp = cur_temp + rng.normal(0, 2.5)
            f_wind = max(0, cur_wind + rng.normal(0, 4))
            f_risk = predict_risk(model, f_temp, f_wind, mig)
            forecast_risks.append(f_risk)

    # Prepend "NOW" as first point
    all_labels = ["NOW"] + forecast_labels
    all_risks = [live_risk] + forecast_risks

    st.plotly_chart(
        make_forecast_chart(all_labels, all_risks),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Forecast summary row
    max_risk = max(forecast_risks)
    max_idx = forecast_risks.index(max_risk)
    min_risk = min(forecast_risks)
    avg_risk = sum(forecast_risks) / len(forecast_risks)
    danger_hours = sum(1 for r in forecast_risks if r > 70)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("PEAK RISK", f"{max_risk:.1f}%", delta=f"at {forecast_labels[max_idx]}")
    with fc2:
        st.metric("MIN RISK", f"{min_risk:.1f}%")
    with fc3:
        st.metric("AVG RISK", f"{avg_risk:.1f}%")
    with fc4:
        st.metric("DANGER HOURS", f"{danger_hours}/12",
                  delta="ALERT" if danger_hours > 0 else "CLEAR",
                  delta_color="inverse" if danger_hours == 0 else "normal")

    # ═════════════════════════════════════════════════════════════════
    # DATA TICKER — CSS animated at absolute bottom
    # ═════════════════════════════════════════════════════════════════
    ticker_text = (
        f"RAW TELEMETRY: SENSORS NOMINAL "
        f"| TEMP: {cur_temp:.1f}C "
        f"| WIND: {cur_wind:.1f}km/h DIR: {cur_dir if cur_dir else 'N/A'}deg "
        f"| MIGRATION: {'ACTIVE' if mig else 'INACTIVE'} "
        f"| AI RISK: {live_risk:.1f}% [{risk_label(live_risk)}] "
        f"| PREDICTIVE HORIZON: 12H "
        f"| PEAK FORECAST: {max_risk:.1f}% AT {forecast_labels[max_idx]} "
        f"| DANGER HOURS: {danger_hours}/12 "
        f"| BASH PROTOCOL: ACTIVE "
        f"| MODEL: RANDOM FOREST 200T R2={r2_score:.3f} "
        f"| COORD: {NEOM_LAT}N {NEOM_LON}E "
        f"| AIRPORT: NEOM INTL OEGN "
        f"| CACHE TTL: 900s "
        f"| DATA SRC: {data_source} "
        f"| SYNC: {utc_str}Z "
        f"| NEOM BIO-SECURE V8.0 "
        f"| {'=' * 20} "
    )

    st.markdown(
        f'<div class="ticker-wrap">'
        f'<div class="ticker-content">{ticker_text}{ticker_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
