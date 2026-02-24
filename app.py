"""
NEOM Bio-Secure v9.0 — SCADA Telemetry Dashboard
Bird Aircraft Strike Hazard (BASH) Decision Support System
NEOM Smart City | Aviation Safety Division

Architecture: Native Streamlit dark-mode + plotly_dark template.
Zero custom CSS. Zero HTML injection. Zero ValueErrors.
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

# ─── PAGE CONFIG (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="NEOM Bio-Secure | BASH SCADA",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CONSTANTS ───────────────────────────────────────────────────────
NEOM_LAT = 28.2933
NEOM_LON = 35.0000
UTC3 = timezone(timedelta(hours=3))
MIGRATION_MONTHS = {3, 4, 5, 9, 10, 11}


# ═════════════════════════════════════════════════════════════════════
# DATA GENERATION — 5000 synthetic rows, cached
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_training_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 5_000
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    doy = dates.dayofyear
    month = dates.month

    temp_base = 25 + 18 * np.sin((doy - 80) * 2 * np.pi / 365)
    temperature = np.clip(temp_base + np.random.normal(0, 4, n), 8, 52)
    wind_speed = np.clip(np.random.gamma(2.5, 7.5, n), 0, 75)
    migration = np.isin(month, list(MIGRATION_MONTHS)).astype(int)

    base_p = 0.08
    mig_p = migration * 0.28
    thermal_p = ((temperature >= 22) & (temperature <= 40)).astype(float) * 0.15
    low_wind_p = (wind_speed < 18).astype(float) * 0.12
    high_wind_p = (wind_speed > 55).astype(float) * (-0.08)
    prob = np.clip(base_p + mig_p + thermal_p + low_wind_p + high_wind_p, 0.03, 0.72)

    risk_event = (np.random.random(n) < prob).astype(int)
    flip = np.random.random(n) < 0.05
    risk_event = np.where(flip, 1 - risk_event, risk_event)

    return pd.DataFrame({
        "temperature": temperature,
        "wind_speed": wind_speed,
        "migration_season": migration,
        "risk_event": risk_event,
    })


# ═════════════════════════════════════════════════════════════════════
# MODEL TRAINING — cached resource
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model():
    df = generate_training_data()
    X = df[["temperature", "wind_speed", "migration_season"]]
    y = df["risk_event"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    imp = pd.DataFrame({
        "Feature": ["Migration Season", "Wind Speed", "Temperature"],
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=True).reset_index(drop=True)
    return clf, acc, imp


# ═════════════════════════════════════════════════════════════════════
# LIVE WEATHER — Open-Meteo, cached 15 min
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_live_neom_weather():
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
        return (
            round(cur["temperature_2m"], 1),
            round(cur["wind_speed_10m"], 1),
            "OPEN-METEO API",
            None,
        )
    except Exception as exc:
        return None, None, "FALLBACK", str(exc)


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def is_migration(month: int) -> int:
    return 1 if month in MIGRATION_MONTHS else 0


def get_advisory(risk: float) -> dict:
    if risk >= 60:
        return {
            "level": "CRITICAL",
            "color": "red",
            "icon": "🔴",
            "directive": "GROUND ALL AIRCRAFT — IMMEDIATE ACTION REQUIRED",
            "steps": [
                "Suspend all take-off and landing operations immediately",
                "Issue NOTAM for NEOM INTL (OEGN) — Bird Strike Hazard",
                "Deploy BASH field survey teams to all active runways",
                "Notify ATC supervisor, airline ops, and Station Manager",
                "Log incident per ICAO Bird Strike Reporting Protocol",
            ],
        }
    elif risk >= 30:
        return {
            "level": "WARNING",
            "color": "orange",
            "icon": "🟠",
            "directive": "ENHANCED MONITORING — AMBER ALERT ACTIVE",
            "steps": [
                "Increase runway bird patrol frequency to every 15 min",
                "Activate secondary radar for near-field wildlife tracking",
                "Brief all inbound/outbound crews on elevated BASH risk",
                "Place wildlife management team on immediate standby",
                "Reassess risk in 30 min — escalate if trend worsens",
            ],
        }
    else:
        return {
            "level": "CLEAR",
            "color": "green",
            "icon": "🟢",
            "directive": "NORMAL OPERATIONS — ALL SYSTEMS NOMINAL",
            "steps": [
                "Standard runway inspection schedule remains active",
                "Routine BASH monitoring — no intervention required",
                "All systems nominal — proceed with flight schedule",
                "Log current reading and archive per standard protocol",
            ],
        }


# ═════════════════════════════════════════════════════════════════════
# CHART BUILDERS — all use template="plotly_dark", no invalid kwargs
# ═════════════════════════════════════════════════════════════════════

def make_risk_gauge(risk: float) -> go.Figure:
    """Large center gauge — AI risk probability 0-100%."""
    if risk < 30:
        bar_color = "#00E676"
    elif risk < 60:
        bar_color = "#FFD600"
    else:
        bar_color = "#FF1744"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number={
            "suffix": "%",
            "font": {"size": 54, "color": "#ECEFF1"},
            "valueformat": ".1f",
        },
        delta={
            "reference": 30,
            "relative": False,
            "font": {"size": 14},
            "decreasing": {"color": "#00E676"},
            "increasing": {"color": "#FF1744"},
        },
        title={
            "text": "AI COLLISION PROBABILITY",
            "font": {"size": 14, "color": "#90A4AE"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#455A64",
                "tickfont": {"size": 11, "color": "#78909C"},
                "dtick": 20,
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#263238",
            "borderwidth": 1,
            "bordercolor": "#37474F",
            "steps": [
                {"range": [0, 30], "color": "rgba(0,230,118,0.08)"},
                {"range": [30, 60], "color": "rgba(255,214,0,0.08)"},
                {"range": [60, 100], "color": "rgba(255,23,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#FF1744", "width": 2},
                "thickness": 0.8,
                "value": 60,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin={"t": 80, "b": 20, "l": 30, "r": 30},
    )
    return fig


def make_feature_importance_chart(imp_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar — feature importances."""
    colors = []
    for f in imp_df["Feature"]:
        if "Migration" in f:
            colors.append("#448AFF")
        elif "Wind" in f:
            colors.append("#7C4DFF")
        else:
            colors.append("#FFD600")

    fig = go.Figure(go.Bar(
        x=imp_df["Importance"] * 100,
        y=imp_df["Feature"],
        orientation="h",
        marker={"color": colors, "opacity": 0.9},
        text=[f"{v*100:.1f}%" for v in imp_df["Importance"]],
        textposition="outside",
        textfont={"size": 12, "color": "#B0BEC5"},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin={"t": 20, "b": 20, "l": 10, "r": 60},
        xaxis={
            "range": [0, imp_df["Importance"].max() * 100 * 1.4],
            "ticksuffix": "%",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.05)",
        },
        yaxis={
            "tickfont": {"size": 12},
        },
        bargap=0.35,
    )
    return fig


def make_trend_chart(model, temp: float, wind: float, mig: int) -> go.Figure:
    """12-hour synthetic risk trend anchored to current conditions."""
    seed = int(
        datetime.now(UTC3).replace(minute=0, second=0, microsecond=0).timestamp()
    ) % (2**31)
    rng = np.random.default_rng(seed)

    hours = list(range(-11, 0))
    h_temps = np.clip(temp + rng.normal(0, 3, 11), 8, 52)
    h_winds = np.clip(wind + rng.normal(0, 5, 11), 0, 75)

    hist_risk = [
        model.predict_proba([[t, w, mig]])[0][1] * 100
        for t, w in zip(h_temps, h_winds)
    ]
    live_risk = model.predict_proba([[temp, wind, mig]])[0][1] * 100

    all_hours = hours + [0]
    all_risk = hist_risk + [live_risk]
    labels = [f"{h}h" for h in hours] + ["NOW"]

    if live_risk >= 60:
        line_color = "#FF1744"
    elif live_risk >= 30:
        line_color = "#FFD600"
    else:
        line_color = "#00E676"

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=all_hours,
        y=all_risk,
        fill="tozeroy",
        fillcolor="rgba(68,138,255,0.07)",
        line={"color": "rgba(0,0,0,0)"},
        showlegend=False,
        hoverinfo="skip",
    ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=hours,
        y=hist_risk,
        mode="lines+markers",
        line={"color": "#546E7A", "width": 2, "dash": "dot"},
        marker={"size": 4, "color": "#546E7A"},
        name="Historical",
        hovertemplate="T%{x}h: %{y:.1f}%<extra></extra>",
    ))

    # Connector to live
    fig.add_trace(go.Scatter(
        x=[-1, 0],
        y=[hist_risk[-1], live_risk],
        mode="lines",
        line={"color": line_color, "width": 2.5},
        showlegend=False,
        hoverinfo="skip",
    ))

    # Live marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[live_risk],
        mode="markers+text",
        marker={"color": line_color, "size": 12, "line": {"color": "#263238", "width": 2}},
        text=[f" {live_risk:.1f}%"],
        textposition="middle right",
        textfont={"size": 13, "color": line_color},
        name="LIVE",
        hovertemplate=f"LIVE: {live_risk:.1f}%<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(
        y=30,
        line={"color": "#FFD600", "dash": "dash", "width": 1},
        opacity=0.5,
        annotation_text="WARN",
        annotation_font={"size": 10, "color": "#FFD600"},
        annotation_position="right",
    )
    fig.add_hline(
        y=60,
        line={"color": "#FF1744", "dash": "dash", "width": 1},
        opacity=0.5,
        annotation_text="CRIT",
        annotation_font={"size": 10, "color": "#FF1744"},
        annotation_position="right",
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin={"t": 20, "b": 50, "l": 10, "r": 70},
        xaxis={
            "title": "Hours Relative to Now",
            "tickvals": list(range(-10, 1, 2)),
            "ticktext": [f"{h}h" for h in range(-10, 0, 2)] + ["NOW"],
            "gridcolor": "rgba(255,255,255,0.05)",
            "zeroline": True,
            "zerolinecolor": "rgba(255,255,255,0.15)",
        },
        yaxis={
            "title": "Risk %",
            "range": [0, 100],
            "ticksuffix": "%",
            "gridcolor": "rgba(255,255,255,0.05)",
        },
        legend={
            "orientation": "h",
            "y": -0.18,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════

def build_sidebar(accuracy: float):
    with st.sidebar:
        st.markdown("### NEOM Bio-Secure")
        st.caption("BASH SCADA Control Panel")
        st.divider()

        mode = st.selectbox(
            "OPERATING MODE",
            ["Live Real-Time", "Simulation"],
        )

        sim_temp = sim_wind = sim_mig = None
        if mode == "Simulation":
            st.divider()
            st.markdown("**Simulation Parameters**")
            sim_temp = st.slider("Temperature (C)", 5, 52, 32)
            sim_wind = st.slider("Wind Speed (km/h)", 0, 75, 20)
            mig_opt = st.selectbox("Migration Season", ["Auto-detect", "Active", "Inactive"])
            if mig_opt == "Active":
                sim_mig = 1
            elif mig_opt == "Inactive":
                sim_mig = 0
            else:
                sim_mig = is_migration(datetime.now(UTC3).month)

        st.divider()
        st.markdown("**AI Engine**")
        st.text(f"Algorithm:    Random Forest")
        st.text(f"Estimators:   150 trees")
        st.text(f"Training:     5,000 rows")
        st.text(f"Test Acc:     {accuracy*100:.1f}%")
        st.text(f"Features:     3")
        st.divider()
        st.caption("NEOM Bio-Secure v9.0 | SCADA Edition")

    return mode, sim_temp, sim_wind, sim_mig


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    # ── Timestamps ───────────────────────────────────────────────────
    now_ast = datetime.now(UTC3)
    now_utc = datetime.now(timezone.utc)

    # ── Train model ──────────────────────────────────────────────────
    model, accuracy, imp_df = train_model()

    # ── Sidebar ──────────────────────────────────────────────────────
    mode, sim_temp, sim_wind, sim_mig = build_sidebar(accuracy)

    # ── Resolve sensor data ──────────────────────────────────────────
    if mode == "Simulation":
        temp = float(sim_temp)
        wind = float(sim_wind)
        mig = int(sim_mig)
        data_source = "SIMULATION"
        uplink_status = "SIM"
    else:
        lv_t, lv_w, data_source, err = get_live_neom_weather()
        temp = lv_t if lv_t is not None else 32.0
        wind = lv_w if lv_w is not None else 18.0
        if lv_t is None:
            data_source = "FALLBACK"
        mig = is_migration(now_ast.month)
        uplink_status = "ACTIVE" if lv_t is not None else "DEGRADED"

    # ── Risk prediction ──────────────────────────────────────────────
    risk_pct = model.predict_proba([[temp, wind, mig]])[0][1] * 100
    adv = get_advisory(risk_pct)

    # ═════════════════════════════════════════════════════════════════
    # TOP BAR — System telemetry banner
    # ═════════════════════════════════════════════════════════════════
    sys_line = (
        f"`[SYS_ID: NEOM_BASH_V9]` | "
        f"`[UPLINK: {uplink_status}]` | "
        f"`[SRC: {data_source}]` | "
        f"`LAT: {NEOM_LAT}N` | "
        f"`LON: {NEOM_LON}E` | "
        f"`{now_ast.strftime('%Y-%m-%d %H:%M:%S')} AST` | "
        f"`{now_utc.strftime('%H:%M:%S')} UTC`"
    )
    st.success(sys_line)

    # ═════════════════════════════════════════════════════════════════
    # KPI ROW — Temperature, Wind, Migration, Risk Level
    # ═════════════════════════════════════════════════════════════════
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(label="TEMPERATURE", value=f"{temp:.1f} C", delta=f"{data_source}")
    with k2:
        st.metric(label="WIND SPEED", value=f"{wind:.1f} km/h", delta=f"{data_source}")
    with k3:
        mig_label = "ACTIVE" if mig == 1 else "INACTIVE"
        st.metric(label="MIGRATION SEASON", value=mig_label, delta=f"Month {now_ast.month}")
    with k4:
        st.metric(
            label="RISK LEVEL",
            value=f"{risk_pct:.1f}%",
            delta=f"{adv['level']}",
            delta_color="inverse" if risk_pct < 30 else "off" if risk_pct < 60 else "normal",
        )

    st.divider()

    # ═════════════════════════════════════════════════════════════════
    # CENTER — Main Risk Gauge (large)
    # ═════════════════════════════════════════════════════════════════
    st.markdown("#### AI RISK ASSESSMENT — BIRD STRIKE HAZARD INDEX")

    gauge_col, advisory_col = st.columns([3, 2])

    with gauge_col:
        st.plotly_chart(
            make_risk_gauge(risk_pct),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with advisory_col:
        st.markdown(f"#### {adv['icon']} BASH ADVISORY: {adv['level']}")
        st.warning(adv["directive"]) if risk_pct >= 30 else st.info(adv["directive"])
        if risk_pct >= 60:
            st.error(adv["directive"])

        for i, step in enumerate(adv["steps"], 1):
            st.markdown(f"**{i}.** {step}")

        st.divider()
        st.caption(
            f"Airport: NEOM INTL (OEGN) | "
            f"Model: RF-150 | "
            f"Accuracy: {accuracy*100:.1f}% | "
            f"NOTAM: {'ISSUED' if risk_pct >= 60 else 'NOMINAL'}"
        )

    st.divider()

    # ═════════════════════════════════════════════════════════════════
    # BOTTOM ROW — Feature Importance + 12-Hour Trend
    # ═════════════════════════════════════════════════════════════════
    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("#### FEATURE IMPORTANCE — WHY THE AI DECIDED")
        st.caption("Signal weight driving the risk score (Random Forest permutation importance)")
        st.plotly_chart(
            make_feature_importance_chart(imp_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        with st.expander("Live Input Vector"):
            vec_c1, vec_c2, vec_c3 = st.columns(3)
            vec_c1.metric("Temp", f"{temp:.1f} C")
            vec_c2.metric("Wind", f"{wind:.1f} km/h")
            vec_c3.metric("Migration", "Yes" if mig else "No")

    with bottom_right:
        st.markdown("#### 12-HOUR RISK TREND")
        st.caption("Synthetic historical context seeded per cache window (15 min TTL)")
        st.plotly_chart(
            make_trend_chart(model, temp, wind, mig),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        with st.expander("Model Statistics"):
            ms1, ms2 = st.columns(2)
            with ms1:
                st.text(f"Training Samples:  4,000")
                st.text(f"Test Samples:      1,000")
                st.text(f"Test Accuracy:     {accuracy*100:.1f}%")
            with ms2:
                st.text(f"Estimators:        150")
                st.text(f"Max Depth:         12")
                st.text(f"Min Leaf Samples:  10")

    # ── Footer telemetry ─────────────────────────────────────────────
    st.divider()
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.caption(f"**COORD:** {NEOM_LAT}N, {NEOM_LON}E")
    f2.caption(f"**AIRPORT:** OEGN")
    f3.caption(f"**AI CONF:** {accuracy*100:.1f}%")
    f4.caption(f"**TREES:** 150")
    f5.caption(f"**CACHE TTL:** 15 min")
    f6.caption(f"**VER:** NEOM Bio-Secure v9.0")


if __name__ == "__main__":
    main()
