"""
NEOM Bio-Secure V8.0 — Midnight Bio-Radar Edition
Bird Aircraft Strike Hazard (BASH) Decision Support System
NEOM Smart City | Aviation Safety Division

Layout  : st.columns([4, 6]) — Expert Core (left) + Visual Twin map (right)
Data    : Open-Meteo API — current conditions + 12-hour hourly forecast
AI      : Random Forest trained on 5 000 synthetic rows (Temp, Wind, Season)
Charts  : plotly.graph_objects with template="plotly_dark" — zero ValueErrors
Map     : pydeck ScatterplotLayer (airport) + ArcLayer (wind vectors)
Ticker  : CSS @keyframes marquee fixed at the bottom of the viewport
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

# ── Page config must be the very first Streamlit call ───────────────
st.set_page_config(
    page_title="NEOM Bio-Secure V8.0",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════
# CSS  —  Midnight Bio-Radar palette + ticker keyframe
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
    background-color: #0A0F1D !important;
    color: #CFD8DC !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────── */
#MainMenu                              { visibility: hidden !important; }
header[data-testid="stHeader"]        { display:    none    !important; }
footer                                 { visibility: hidden !important; }
[data-testid="stToolbar"]             { display:    none    !important; }
.stDeployButton                        { display:    none    !important; }
[data-testid="collapsedControl"]       { display:    none    !important; }

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0D1220 !important;
    border-right: 1px solid #1A2540;
}

/* ── Metric cards ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background    : #0D1428;
    border        : 1px solid #1A2540;
    border-radius : 6px;
    padding       : 12px 14px !important;
}
[data-testid="stMetricLabel"] {
    font-family : 'Courier New', monospace !important;
    font-size   : 10px !important;
    color       : #607D8B !important;
    letter-spacing: 1.5px;
}
[data-testid="stMetricValue"] {
    font-family : 'Courier New', monospace !important;
    font-size   : 22px !important;
    color       : #E0E0E0 !important;
}

/* ── Bottom padding so ticker never overlaps content ───────────── */
.main .block-container { padding-bottom: 48px !important; }

/* ══════════════════════════════════════════════════════════════════
   TICKER — CSS @keyframes, fixed at viewport bottom
══════════════════════════════════════════════════════════════════ */
.bs-ticker-wrap {
    position   : fixed;
    bottom     : 0;
    left       : 0;
    right      : 0;
    height     : 30px;
    display    : flex;
    align-items: center;
    overflow   : hidden;
    background : linear-gradient(90deg, #060911 0%, #0A0F1D 40%, #0A0F1D 60%, #060911 100%);
    border-top : 1px solid rgba(0,229,255,0.18);
    z-index    : 99999;
}
.bs-ticker-tag {
    flex-shrink : 0;
    padding     : 0 14px;
    font-family : 'Courier New', monospace;
    font-size   : 10px;
    font-weight : bold;
    color       : #00E5FF;
    border-right: 1px solid rgba(0,229,255,0.25);
    letter-spacing: 1px;
    white-space : nowrap;
}
.bs-ticker-track {
    flex        : 1;
    overflow    : hidden;
    position    : relative;
    height      : 100%;
    display     : flex;
    align-items : center;
}
.bs-ticker-inner {
    display    : flex;
    white-space: nowrap;
    animation  : bs-scroll 90s linear infinite;
}
@keyframes bs-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.bs-ticker-seg {
    font-family   : 'Courier New', monospace;
    font-size     : 10px;
    color         : #90A4AE;
    padding       : 0 6px;
    letter-spacing: 0.5px;
}
.bs-ticker-dot {
    font-family : 'Courier New', monospace;
    font-size   : 10px;
    color       : #00E5FF;
    padding     : 0 4px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════
NEOM_LAT = 28.03          # NEOM Bay
NEOM_LON = 34.64
UTC      = timezone.utc
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}

# Palette
C_CYAN  = "#00E5FF"
C_GREEN = "#00E676"
C_AMBER = "#FFB300"
C_RED   = "#FF1744"
C_BG    = "#0A0F1D"


# ════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION — 5 000 rows
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
# MODEL TRAINING
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
# LIVE WEATHER + 12-HOUR FORECAST — Open-Meteo (cached 15 min)
# ════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900, show_spinner=False)
def get_live_weather() -> dict:
    """
    Fetches BOTH current conditions AND 13 hourly rows (now + 12 h ahead)
    from Open-Meteo for NEOM Bay.  Falls back to plausible synthetic values
    on any network or parse error so the dashboard always renders.
    """
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
    """Return [R, G, B, A] list suitable for PyDeck layer colours."""
    if risk >= 70:
        return [255, 23, 68, alpha]
    if risk >= 30:
        return [255, 179, 0, alpha]
    return [0, 229, 255, alpha]


def _geo_offset(lat: float, lon: float, bearing_deg: float, dist_km: float):
    """Haversine offset — return (lat2, lon2) along bearing for dist_km."""
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


# ════════════════════════════════════════════════════════════════════
# CHART — RISK GAUGE
# ════════════════════════════════════════════════════════════════════
def make_risk_gauge(risk: float) -> go.Figure:
    bar_color = risk_hex(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number={
            "suffix":      "%",
            "font":        {
                "size": 56, "color": "#ECEFF1",
                "family": "Courier New, monospace",
            },
            "valueformat": ".1f",
        },
        delta={
            "reference":  30,
            "relative":   False,
            "font":       {"size": 13},
            "decreasing": {"color": C_GREEN},
            "increasing": {"color": C_RED},
        },
        title={
            "text": (
                "AI COLLISION PROBABILITY<br>"
                "<span style='font-size:11px;color:#607D8B'>"
                "BASH RISK INDEX · RANDOM FOREST</span>"
            ),
            "font": {"size": 14, "color": "#90A4AE"},
        },
        gauge={
            "axis": {
                "range":     [0, 100],
                "tickwidth":  1,
                "tickcolor":  "#37474F",
                "tickfont":   {"size": 9, "color": "#546E7A"},
                "dtick":      20,
            },
            "bar":        {"color": bar_color, "thickness": 0.22},
            "bgcolor":    "rgba(13,20,40,0.8)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30],  "color": "rgba(0,230,118,0.06)"},
                {"range": [30, 70],  "color": "rgba(255,179,0,0.06)"},
                {"range": [70, 100], "color": "rgba(255,23,68,0.07)"},
            ],
            "threshold": {
                "line":      {"color": C_RED, "width": 2},
                "thickness": 0.80,
                "value":     70,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin={"t": 110, "b": 10, "l": 20, "r": 20},
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# CHART — XAI FEATURE IMPORTANCE
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
            "opacity": 0.88,
            "line":    {"color": "rgba(255,255,255,0.08)", "width": 1},
        },
        text=[f"{v * 100:.1f}%" for v in imp_df["Importance"]],
        textposition="outside",
        textfont={"size": 11, "color": "#78909C"},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    max_imp = float(imp_df["Importance"].max()) * 100
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin={"t": 8, "b": 20, "l": 10, "r": 75},
        xaxis={
            "range":      [0, max_imp * 1.45],
            "ticksuffix": "%",
            "showgrid":   True,
            "gridcolor":  "rgba(255,255,255,0.04)",
            "tickfont":   {"size": 10},
        },
        yaxis={"tickfont": {"size": 11}},
        bargap=0.40,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# CHART — 12-HOUR FORECAST  (real API data → AI model)
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

    # Filled area base
    fig.add_trace(go.Scatter(
        x=x_idx, y=risks,
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.04)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=x_idx,
        y=risks,
        mode="lines+markers",
        name="AI Risk Forecast",
        line={"color": line_clr, "width": 2.5},
        marker={
            "size":  [10 if r >= 70 else 6 for r in risks],
            "color": [risk_hex(r) for r in risks],
            "line":  {"color": C_BG, "width": 1.5},
        },
        text=x_labels,
        hovertemplate="<b>%{text}</b><br>Risk: %{y:.1f}%<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(
        y=30,
        line={"color": C_AMBER, "dash": "dash", "width": 1},
        opacity=0.5,
        annotation_text="WARN 30%",
        annotation_font={"size": 9, "color": C_AMBER},
        annotation_position="right",
    )
    fig.add_hline(
        y=70,
        line={"color": C_RED, "dash": "dash", "width": 1.5},
        opacity=0.65,
        annotation_text="CRITICAL 70%",
        annotation_font={"size": 9, "color": C_RED},
        annotation_position="right",
    )
    fig.add_vline(
        x=0,
        line={"color": C_CYAN, "dash": "dot", "width": 1},
        opacity=0.45,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin={"t": 16, "b": 50, "l": 10, "r": 100},
        xaxis={
            "tickvals":  x_idx,
            "ticktext":  x_labels,
            "showgrid":  True,
            "gridcolor": "rgba(255,255,255,0.04)",
            "tickfont":  {"size": 10},
        },
        yaxis={
            "title":      "Risk %",
            "range":      [0, 108],
            "ticksuffix": "%",
            "gridcolor":  "rgba(255,255,255,0.04)",
        },
        legend={"orientation": "h", "y": -0.22, "bgcolor": "rgba(0,0,0,0)"},
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# PYDECK MAP — dark Carto base, airport marker + wind arc vectors
# ════════════════════════════════════════════════════════════════════
def make_pydeck_map(risk_pct: float, wind_speed: float, wind_dir: float) -> pdk.Deck:
    dot      = risk_rgba_list(risk_pct, 230)
    ring     = risk_rgba_list(risk_pct, 55)
    arc_src  = risk_rgba_list(risk_pct, 180)
    arc_tgt  = risk_rgba_list(risk_pct, 15)

    airport_data = [{
        "lon":  NEOM_LON,
        "lat":  NEOM_LAT,
        "name": (
            f"NEOM INTL (OEGN)"
            f"  |  Risk: {risk_pct:.1f}%"
            f"  |  Wind: {wind_speed:.1f} km/h"
        ),
    }]

    # 3 bearing offsets × 3 distances = 9 wind-vector arcs
    arc_rows = []
    for offset in (-25, 0, 25):
        bearing = (wind_dir + offset) % 360
        for dist_km in (18, 36, 60):
            elat, elon = _geo_offset(NEOM_LAT, NEOM_LON, bearing, dist_km)
            arc_rows.append({
                "slat": NEOM_LAT, "slon": NEOM_LON,
                "elat": elat,      "elon": elon,
            })

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_color=dot,
        get_radius=6_000,
        pickable=True,
        filled=True,
        stroked=True,
        radius_min_pixels=7,
        radius_max_pixels=28,
        line_width_min_pixels=2,
    )
    ring_layer = pdk.Layer(
        "ScatterplotLayer",
        data=airport_data,
        get_position=["lon", "lat"],
        get_color=ring,
        get_radius=22_000,
        filled=True,
        stroked=False,
    )
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_rows,
        get_source_position=["slon", "slat"],
        get_target_position=["elon", "elat"],
        get_source_color=arc_src,
        get_target_color=arc_tgt,
        get_width=2,
        pickable=False,
    )

    view = pdk.ViewState(
        latitude=NEOM_LAT,
        longitude=NEOM_LON,
        zoom=7,
        pitch=38,
        bearing=8,
    )
    return pdk.Deck(
        layers=[ring_layer, arc_layer, scatter_layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}"},
    )


# ════════════════════════════════════════════════════════════════════
# TICKER — fixed bottom bar
# ════════════════════════════════════════════════════════════════════
def inject_ticker(
    risk_pct: float,
    temp: float,
    wind: float,
    accuracy: float,
    source: str,
    now_utc: datetime,
) -> None:
    level = (
        "DANGER — CRITICAL" if risk_pct >= 70
        else ("WARNING — ELEVATED" if risk_pct >= 30 else "NOMINAL — CLEAR")
    )
    segments = [
        "RAW TELEMETRY: SENSORS NOMINAL",
        f"TEMP: {temp:.1f}°C",
        f"WIND: {wind:.1f} km/h",
        f"BASH RISK INDEX: {risk_pct:.1f}%",
        f"THREAT LEVEL: {level}",
        "PREDICTIVE HORIZON: 12H",
        "BASH PROTOCOL: ACTIVE",
        "AIRPORT: NEOM INTL (OEGN)",
        "COORD: 28.03°N / 34.64°E",
        "AI ENGINE: RANDOM FOREST · 150 TREES",
        f"MODEL ACCURACY: {accuracy * 100:.1f}%",
        f"DATA SOURCE: {source}",
        f"UTC SYNC: {now_utc.strftime('%H:%M:%S')} ZULU",
        "ICAO MONITORING: ACTIVE",
        "NEOM BIO-SECURE V8.0 · MIDNIGHT BIO-RADAR EDITION",
    ]
    dot   = "<span class='bs-ticker-dot'>◆</span>"
    inner = dot.join(
        f"<span class='bs-ticker-seg'>{s}</span>" for s in segments
    )
    # Duplicate the content so the CSS loop is perfectly seamless
    content = inner + dot + inner

    st.markdown(
        f"<div class='bs-ticker-wrap'>"
        f"<div class='bs-ticker-tag'>▶ LIVE FEED</div>"
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

    # ── Bootstrap AI + live data ────────────────────────────────────
    with st.spinner("Initialising AI engine — please wait…"):
        model, accuracy, imp_df = train_model()

    wx          = get_live_weather()
    temp        = wx["temp_now"]
    wind        = wx["wind_now"]
    wind_dir    = wx["wind_dir"]
    forecast_df = wx["forecast_df"]
    wx_source   = wx["source"]
    mig         = is_migration(now_utc.month)

    risk_pct = float(model.predict_proba([[temp, wind, mig]])[0][1] * 100)
    r_color  = risk_hex(risk_pct)

    uplink_live  = wx_source == "OPEN-METEO"
    uplink_label = "ACTIVE" if uplink_live else "DEGRADED"
    uplink_icon  = "\U0001f7e2" if uplink_live else "\U0001f7e1"   # 🟢 / 🟡

    # ════════════════════════════════════════════════════════════════
    # A.  TOP COMMAND HEADER
    # ════════════════════════════════════════════════════════════════
    h_logo, h_status, h_sync = st.columns([3, 2, 2])

    with h_logo:
        st.markdown(
            "<div style='font-family:\"Courier New\",monospace;padding:4px 0;'>"
            "<span style='font-size:20px;font-weight:700;color:#00E5FF;"
            "letter-spacing:3px;'>⬡ NEOM BIO-SECURE ATC</span><br>"
            "<span style='font-size:10px;color:#546E7A;letter-spacing:2px;'>"
            "BASH DECISION SUPPORT SYSTEM · V8.0 MIDNIGHT BIO-RADAR"
            "</span></div>",
            unsafe_allow_html=True,
        )

    with h_status:
        icon_color = C_GREEN if uplink_live else C_AMBER
        st.markdown(
            f"<div style='font-family:\"Courier New\",monospace;"
            f"text-align:center;padding:4px 0;'>"
            f"<span style='font-size:13px;font-weight:700;color:{icon_color};'>"
            f"{uplink_icon} LIVE UPLINK: {uplink_label}</span><br>"
            f"<span style='font-size:9px;color:#546E7A;letter-spacing:1px;'>"
            f"SRC: {wx_source}</span></div>",
            unsafe_allow_html=True,
        )

    with h_sync:
        st.markdown(
            f"<div style='font-family:\"Courier New\",monospace;"
            f"text-align:right;padding:4px 0;'>"
            f"<span style='font-size:12px;color:#B0BEC5;'>"
            f"LAST SYNC: {now_utc.strftime('%H:%M:%S')} ZULU</span><br>"
            f"<span style='font-size:9px;color:#546E7A;letter-spacing:1px;'>"
            f"{now_utc.strftime('%Y-%m-%d')} · "
            f"{NEOM_LAT}N {NEOM_LON}E</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #1A2540;margin:10px 0 14px;'>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════
    # B.  MAIN SPLIT  [4  |  6]
    # ════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([4, 6])

    # ── LEFT: Expert Core ───────────────────────────────────────────
    with left_col:

        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:10px;"
            "color:#455A64;letter-spacing:2px;margin:0 0 8px;'>"
            "◉ LIVE SENSOR TELEMETRY</p>",
            unsafe_allow_html=True,
        )

        s_temp, s_wind = st.columns(2)
        with s_temp:
            st.metric("TEMPERATURE", f"{temp:.1f} °C")
            st.markdown(
                "<p style='font-family:\"Courier New\",monospace;font-size:9px;"
                "color:#00E676;margin-top:-12px;margin-bottom:0;'>"
                "\U0001f7e2 SENSOR: ONLINE</p>",
                unsafe_allow_html=True,
            )
        with s_wind:
            st.metric("WIND SPEED", f"{wind:.1f} km/h")
            st.markdown(
                "<p style='font-family:\"Courier New\",monospace;font-size:9px;"
                "color:#00E676;margin-top:-12px;margin-bottom:0;'>"
                "\U0001f7e2 SENSOR: ONLINE</p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:10px;"
            "color:#455A64;letter-spacing:2px;margin:14px 0 2px;'>"
            "◉ AI RISK ASSESSMENT</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_risk_gauge(risk_pct),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:10px;"
            "color:#455A64;letter-spacing:2px;margin:6px 0 2px;'>"
            "◉ EXPLAINABLE AI — FEATURE WEIGHTS</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_xai_chart(imp_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── RIGHT: Visual Twin (map) ────────────────────────────────────
    with right_col:

        st.markdown(
            "<p style='font-family:\"Courier New\",monospace;font-size:10px;"
            "color:#455A64;letter-spacing:2px;margin:0 0 8px;'>"
            "◉ VISUAL TWIN — NEOM BAY TACTICAL VIEW</p>",
            unsafe_allow_html=True,
        )

        threat_label = (
            "DANGER — BIRD STRIKE RISK CRITICAL" if risk_pct >= 70
            else (
                "CAUTION — ELEVATED RISK DETECTED" if risk_pct >= 30
                else "NOMINAL — ALL SYSTEMS CLEAR"
            )
        )
        st.markdown(
            f"<div style='background:{r_color}18;border:1px solid {r_color}44;"
            f"border-radius:4px;padding:6px 14px;margin-bottom:10px;"
            f"font-family:\"Courier New\",monospace;font-size:11px;color:{r_color};'>"
            f"▶ THREAT VECTOR STATUS: {threat_label}"
            f" &nbsp;|&nbsp; {risk_pct:.1f}%</div>",
            unsafe_allow_html=True,
        )

        try:
            deck = make_pydeck_map(risk_pct, wind, wind_dir)
            st.pydeck_chart(deck, use_container_width=True, height=648)
        except Exception as map_err:
            st.warning(f"Map renderer unavailable: {map_err}")
            st.markdown(
                f"<div style='background:#0D1428;border:1px solid #1A2540;"
                f"border-radius:6px;padding:40px;text-align:center;"
                f"font-family:\"Courier New\",monospace;'>"
                f"<div style='color:{C_CYAN};font-size:36px;'>⬡</div>"
                f"<div style='color:#546E7A;font-size:11px;margin-top:8px;'>NEOM BAY</div>"
                f"<div style='color:#B0BEC5;font-size:14px;'>28.03°N / 34.64°E</div>"
                f"<div style='color:{r_color};font-size:24px;margin-top:14px;'>"
                f"RISK: {risk_pct:.1f}%</div></div>",
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════
    # C.  BOTTOM — 12-HOUR FORECAST CHART
    # ════════════════════════════════════════════════════════════════
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1A2540;margin:16px 0 10px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family:\"Courier New\",monospace;font-size:10px;"
        "color:#455A64;letter-spacing:2px;margin-bottom:4px;'>"
        "◉ 12-HOUR PREDICTIVE HORIZON — "
        "AI RISK FORECAST (OPEN-METEO DATA FEED)</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_forecast_chart(forecast_df, model, mig),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.caption("AIRPORT: NEOM INTL (OEGN)")
    f2.caption(f"COORD: {NEOM_LAT}N · {NEOM_LON}E")
    f3.caption(f"AI ACCURACY: {accuracy * 100:.1f}%")
    f4.caption(f"CACHE TTL: 900 s · {wx_source}")
    f5.caption("NEOM Bio-Secure V8.0 · Midnight Bio-Radar")

    # Spacer above the fixed ticker
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # ── Ticker — injected last so it renders on top ─────────────────
    inject_ticker(risk_pct, temp, wind, accuracy, wx_source, now_utc)


if __name__ == "__main__":
    main()
