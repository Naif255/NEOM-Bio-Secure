import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ========================================
# NEOM CONFIGURATION & BRANDING
# ========================================

st.set_page_config(
    page_title="NEOM Bio-Secure",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

NEOM_COLORS = {
    'primary': '#00D4AA',      # NEOM Teal
    'secondary': '#1A1A2E',    # Dark Navy
    'accent': '#16213E',       # Deep Blue
    'success': '#00FF88',      # Bright Green
    'warning': '#FFD700',      # Gold
    'danger': '#FF4444',       # Red
    'text': '#FFFFFF'          # White
}

# ========================================
# DATA GENERATION ENGINE (SIMULATION MODE)
# ========================================

@st.cache_data
def load_historical_data():
    """Generate Advanced NEOM Bird Strike Simulation Data"""
    try:
        # Simulation Parameters
        np.random.seed(42)
        rows = 5000
        dates = pd.date_range(start='2024-01-01', periods=rows, freq='H')
        
        # Generate Weather Data
        data = pd.DataFrame({
            'Date': dates,
            'Temperature': np.random.normal(30, 8, rows),  # Mean 30C
            'Wind_Speed': np.random.uniform(0, 60, rows),  # 0-60 km/h
            'Migration_Season': np.random.choice([0, 1], rows, p=[0.7, 0.3]) # 30% active season
        })
        
        # Clip realistic values
        data['Temperature'] = data['Temperature'].clip(5, 50)
        
        # Smart Risk Logic (Creating patterns for AI to learn)
        # Risk increases if: Migration is ON + Wind is LOW + Temp is MODERATE
        conditions = (
            (data['Migration_Season'] == 1) & 
            (data['Wind_Speed'] < 25) & 
            (data['Temperature'].between(20, 38))
        )
        
        # Assign Risk Event based on logic + some random noise
        data['Risk_Event'] = np.where(conditions, 1, 0)
        noise = np.random.choice([0, 1], rows, p=[0.95, 0.05]) # 5% randomness
        data['Risk_Event'] = data['Risk_Event'] | noise

        return data, True
        
    except Exception as e:
        st.error(f"❌ Error generating data: {str(e)}")
        return None, False

# ========================================
# MACHINE LEARNING MODEL
# ========================================

@st.cache_resource
def train_prediction_model(data):
    """Train RandomForest model for bird strike prediction"""
    
    # Features and target
    features = ['Temperature', 'Wind_Speed', 'Migration_Season']
    X = data[features].copy()
    y = data['Risk_Event'].copy()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, accuracy, importance_df

def predict_strike_risk(model, temperature, wind_speed, migration_season):
    """Predict bird strike probability"""
    features = np.array([[temperature, wind_speed, migration_season]])
    try:
        prob = model.predict_proba(features)[0][1] * 100  # Probability of risk event
        return min(100, max(0, prob))
    except:
        return 50.0  # Fallback

# ========================================
# ADVANCED UI COMPONENTS
# ========================================

def create_neom_gauge(risk_percentage):
    """Create professional NEOM-themed gauge chart"""
    
    if risk_percentage < 30:
        color, status = NEOM_COLORS['success'], "SAFE / آمن"
    elif risk_percentage < 60:
        color, status = NEOM_COLORS['warning'], "CAUTION / حذر"
    else:
        color, status = NEOM_COLORS['danger'], "DANGER / خطر"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        title={'text': f"Risk Probability<br><span style='font-size:0.8em;color:{color}'>{status}</span>"},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 255, 136, 0.1)"},
                {'range': [50, 100], 'color': "rgba(255, 68, 68, 0.1)"}
            ]
        }
    ))
    
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def create_historical_analysis(data):
    """Create advanced scatter plot for historical analysis"""
    
    fig = px.scatter(
        data.sample(500), # Sample for better visuals
        x='Temperature',
        y='Wind_Speed',
        color='Risk_Event',
        size='Migration_Season',
        color_discrete_map={0: NEOM_COLORS['success'], 1: NEOM_COLORS['danger']},
        title="<b>Historical Analysis: Weather vs Risk</b>",
        labels={'Risk_Event': 'Risk Level'}
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NEOM_COLORS['text']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_risk_recommendation(risk_pct):
    """Generate detailed risk recommendations"""
    if risk_pct >= 60:
        return {
            'level': '🚨 CRITICAL ALERT',
            'action': 'HALT OPERATIONS',
            'color': NEOM_COLORS['danger'],
            'details': 'High probability of bird strikes detected.',
            'recommendations': ['Suspend takeoffs', 'Deploy acoustic deterrents', 'Alert ground control']
        }
    elif risk_pct >= 30:
        return {
            'level': '⚠️ CAUTION',
            'action': 'ENHANCED MONITORING',
            'color': NEOM_COLORS['warning'],
            'details': 'Moderate bird activity expected.',
            'recommendations': ['Notify pilots', 'Increase visual scanning', 'Review weather updates']
        }
    else:
        return {
            'level': '🟢 ALL CLEAR',
            'action': 'NORMAL OPS',
            'color': NEOM_COLORS['success'],
            'details': 'Conditions are safe for flight.',
            'recommendations': ['Continue standard procedure', 'Log routine check']
        }

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # CSS Styling
    st.markdown(f"""
    <style>
    .stApp {{ background-color: #0E1117; color: white; }}
    .metric-container {{ background: rgba(26, 26, 46, 0.8); padding: 15px; border-radius: 10px; border: 1px solid {NEOM_COLORS['primary']}; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🦅 NEOM Bio-Secure")
    st.markdown("### AI-Powered Bird Strike Prediction Platform")
    st.divider()
    
    # Load Data & Model
    data, loaded = load_historical_data()
    
    if loaded:
        model, acc, importance, importance_df = train_prediction_model(data)
        
        # Sidebar
        st.sidebar.header("📡 Live Sensors")
        temp = st.sidebar.slider("Temperature (°C)", 10, 50, 30)
        wind = st.sidebar.slider("Wind Speed (km/h)", 0, 80, 15)
        mig = st.sidebar.selectbox("Migration Season?", ["No", "Yes"])
        mig_val = 1 if mig == "Yes" else 0
        
        # Prediction
        risk = predict_strike_risk(model, temp, wind, mig_val)
        rec = create_risk_recommendation(risk)
        
        # Dashboard Layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(create_neom_gauge(risk), use_container_width=True)
            st.markdown(f"""
            <div style="background-color: {rec['color']}20; border: 1px solid {rec['color']}; padding: 15px; border-radius: 10px;">
                <h3 style="color: {rec['color']}; margin:0;">{rec['level']}</h3>
                <p style="font-weight:bold;">{rec['action']}</p>
                <hr style="border-color: {rec['color']};">
                <ul style="text-align:left;">
                    {''.join([f'<li>{r}</li>' for r in rec['recommendations']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.plotly_chart(create_historical_analysis(data), use_container_width=True)
            
        # Analytics Footer
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{acc:.1%}")
        c2.metric("Total Records Analyzed", f"{len(data):,}")
        c3.metric("System Status", "ONLINE", delta="Active")

if __name__ == "__main__":
    main()
