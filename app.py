%%writefile app.py
#!/usr/bin/env python3
"""
NEOM Bio-Secure: AI-Powered Bird Strike Prediction System
Advanced Thermal Detection & Aviation Safety Platform
Designed for NEOM Smart City Aviation Operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ========================================
# NEOM CONFIGURATION & BRANDING
# ========================================

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
# DATA LOADING & PREPROCESSING
# ========================================

@st.cache_data
def load_historical_data():
    """Load NEOM bird strike historical data"""
    try:
        data = pd.read_csv('neom_bird_history.csv')
        st.success(f"✅ Historical data loaded: {len(data):,} records")
        return data, True
    except FileNotFoundError:
        st.error("❌ Error: neom_bird_history.csv not found!")
        return None, False
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
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
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
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
    
    return model, accuracy, importance_df, X_test, y_test

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
    
    if risk_percentage < 20:
        color, status = NEOM_COLORS['success'], "MINIMAL RISK"
        zone_color = "rgba(0, 255, 136, 0.3)"
    elif risk_percentage < 40:
        color, status = NEOM_COLORS['warning'], "LOW RISK"
        zone_color = "rgba(255, 215, 0, 0.3)"
    elif risk_percentage < 60:
        color, status = "#FF8C00", "MODERATE RISK"
        zone_color = "rgba(255, 140, 0, 0.3)"
    elif risk_percentage < 80:
        color, status = "#FF4500", "HIGH RISK"
        zone_color = "rgba(255, 69, 0, 0.3)"
    else:
        color, status = NEOM_COLORS['danger'], "CRITICAL RISK"
        zone_color = "rgba(255, 68, 68, 0.3)"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b style='color:{NEOM_COLORS['text']}'>BIRD STRIKE RISK ASSESSMENT</b><br>"
                   f"<span style='font-size:18px;color:{color}'>{status}</span>",
            'font': {'size': 20}
        },
        delta={'reference': 50, 'increasing': {'color': NEOM_COLORS['danger']}, 
               'decreasing': {'color': NEOM_COLORS['success']}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': NEOM_COLORS['text']},
            'bar': {'color': color, 'thickness': 0.4},
            'bgcolor': "rgba(26, 26, 46, 0.8)",
            'borderwidth': 3,
            'bordercolor': NEOM_COLORS['primary'],
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0, 255, 136, 0.2)'},
                {'range': [20, 40], 'color': 'rgba(255, 215, 0, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(255, 140, 0, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(255, 69, 0, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(255, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': NEOM_COLORS['text'], 'width': 6},
                'thickness': 0.8,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': NEOM_COLORS['text'], 'size': 16},
        height=450
    )
    
    return fig

def create_historical_analysis(data):
    """Create advanced scatter plot for historical analysis"""
    
    fig = px.scatter(
        data,
        x='Temperature',
        y='Wind_Speed',
        color='Risk_Event',
        size='Migration_Season',
        color_discrete_map={0: NEOM_COLORS['success'], 1: NEOM_COLORS['danger']},
        title="<b>Historical Analysis: Weather Patterns vs Bird Strike Events</b>",
        labels={
            'Temperature': 'Temperature (°C)',
            'Wind_Speed': 'Wind Speed (km/h)',
            'Risk_Event': 'Bird Strike Risk',
            'Migration_Season': 'Migration Activity'
        },
        hover_data=['Date']
    )
    
    # Update traces
    fig.for_each_trace(lambda t: t.update(
        name="No Risk Event" if t.name == "0" else "Risk Event Occurred"
    ))
    
    # Styling
    fig.update_layout(
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NEOM_COLORS['text'], size=12),
        height=450,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(26, 26, 46, 0.8)",
            bordercolor=NEOM_COLORS['primary'],
            borderwidth=1
        )
    )
    
    # Add risk zones
    fig.add_shape(
        type="rect", x0=28, y0=0, x1=38, y1=25,
        fillcolor="rgba(255, 215, 0, 0.1)", line=dict(color="rgba(255, 215, 0, 0.3)"),
        annotation_text="High Risk Zone", annotation_position="top right"
    )
    
    return fig

def create_risk_recommendation(risk_pct, temp, wind, migration):
    """Generate detailed risk recommendations"""
    
    if risk_pct >= 75:
        return {
            'level': '🚨 CRITICAL ALERT',
            'action': 'IMMEDIATE GROUND OPERATIONS',
            'color': NEOM_COLORS['danger'],
            'details': f'Risk: {risk_pct:.1f}% | Ideal conditions for massive bird activity detected',
            'recommendations': [
                '🛑 Suspend all takeoffs and landings',
                '📡 Deploy bird radar monitoring',
                '🚁 Deploy bird dispersal units',
                '📞 Alert all pilots and ground crews'
            ]
        }
    elif risk_pct >= 50:
        return {
            'level': '⚠️ HIGH CAUTION',
            'action': 'ENHANCED MONITORING PROTOCOL',
            'color': '#FF8C00',
            'details': f'Risk: {risk_pct:.1f}% | Elevated bird activity expected',
            'recommendations': [
                '👁️ Increase visual scanning frequency',
                '📊 Monitor thermal conditions closely',
                '🎯 Brief pilots on current conditions',
                '🔄 Update bird activity reports every 15min'
            ]
        }
    elif risk_pct >= 25:
        return {
            'level': '🟡 STANDARD CAUTION',
            'action': 'NORMAL OPERATIONS',
            'color': NEOM_COLORS['warning'],
            'details': f'Risk: {risk_pct:.1f}% | Moderate conditions with standard precautions',
            'recommendations': [
                '✅ Maintain standard bird strike protocols',
                '📋 Regular pilot weather briefings',
                '🔍 Continue routine bird observations',
                '📈 Monitor for changing conditions'
            ]
        }
    else:
        return {
            'level': '🟢 ALL CLEAR',
            'action': 'MINIMAL RISK CONDITIONS',
            'color': NEOM_COLORS['success'],
            'details': f'Risk: {risk_pct:.1f}% | Optimal conditions for safe operations',
            'recommendations': [
                '✈️ Proceed with normal flight operations',
                '🌟 Conditions not favorable for bird activity',
                '📝 Standard reporting procedures',
                '🎯 Maintain situational awareness'
            ]
        }

# ========================================
# MAIN STREAMLIT APPLICATION
# ========================================

def main():
    # Page configuration with NEOM branding
    st.set_page_config(
        page_title="NEOM Bio-Secure",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Advanced NEOM Dark Theme
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, #0B0B1F 0%, #1A1A2E 25%, #16213E 50%, #1A1A2E 75%, #0B0B1F 100%);
        background-attachment: fixed;
    }}
    
    .main-header {{
        background: linear-gradient(90deg, {NEOM_COLORS['primary']}, #007B7F);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.3);
    }}
    
    .metric-container {{
        background: rgba(26, 26, 46, 0.8);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid {NEOM_COLORS['primary']};
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 212, 170, 0.2);
    }}
    
    .alert-box {{
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        border: 2px solid;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }}
    
    .sidebar-info {{
        background: rgba(26, 26, 46, 0.9);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid {NEOM_COLORS['primary']};
        margin: 10px 0;
    }}
    
    .stMetric {{
        background: rgba(26, 26, 46, 0.6);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {NEOM_COLORS['primary']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🦅 NEOM Bio-Secure</h1>
        <h3 style="color: white; margin: 10px 0 0 0;">AI-Powered Bird Strike Prediction System</h3>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Advanced Aviation Safety Through Predictive Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("🔄 Initializing NEOM Bio-Secure AI System..."):
        data, data_loaded = load_historical_data()
        
        if data_loaded:
            model, accuracy, feature_importance, X_test, y_test = train_prediction_model(data)
            
            # System status
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: {NEOM_COLORS['primary']};">📊 DATASET</h3>
                    <h2 style="color: white;">{len(data):,}</h2>
                    <p style="color: rgba(255,255,255,0.7);">Historical Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                risk_events = data['Risk_Event'].sum()
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: {NEOM_COLORS['danger']};">⚠️ EVENTS</h3>
                    <h2 style="color: white;">{risk_events:,}</h2>
                    <p style="color: rgba(255,255,255,0.7);">Risk Events</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: {NEOM_COLORS['success']};">🎯 ACCURACY</h3>
                    <h2 style="color: white;">{accuracy:.1%}</h2>
                    <p style="color: rgba(255,255,255,0.7);">Model Performance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                migration_days = data['Migration_Season'].sum()
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: {NEOM_COLORS['warning']};">🦅 MIGRATION</h3>
                    <h2 style="color: white;">{migration_days:,}</h2>
                    <p style="color: rgba(255,255,255,0.7);">Active Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sidebar controls
            st.sidebar.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, {NEOM_COLORS['primary']}, #007B7F); border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🌡️ LIVE CONDITIONS</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">NEOM Aviation Control</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Input controls
            temperature = st.sidebar.slider(
                "🌡️ Temperature (°C)", 
                min_value=15.0, max_value=45.0, value=30.0, step=0.5,
                key="temp_slider"
            )
            
            wind_speed = st.sidebar.slider(
                "💨 Wind Speed (km/h)", 
                min_value=0.0, max_value=80.0, value=15.0, step=1.0,
                key="wind_slider"
            )
            
            migration_active = st.sidebar.radio(
                "🦅 Migration Season Active?",
                options=["No", "Yes"],
                key="migration_radio"
            )
            
            migration_binary = 1 if migration_active == "Yes" else 0
            
            # Current conditions display
            st.sidebar.markdown(f"""
            <div class="sidebar-info">
                <h4 style="color: {NEOM_COLORS['primary']};">📊 Current Inputs</h4>
                <p style="color: white;">🌡️ Temperature: <strong>{temperature}°C</strong></p>
                <p style="color: white;">💨 Wind Speed: <strong>{wind_speed} km/h</strong></p>
                <p style="color: white;">🦅 Migration: <strong>{migration_active}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Main dashboard
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                # Risk prediction
                current_risk = predict_strike_risk(model, temperature, wind_speed, migration_binary)
                gauge_fig = create_neom_gauge(current_risk)
                st.plotly_chart(gauge_fig, use_container_width=True, key="risk_gauge")
                
                # Risk recommendation
                recommendation = create_risk_recommendation(current_risk, temperature, wind_speed, migration_binary)
                
                st.markdown(f"""
                <div class="alert-box" style="border-color: {recommendation['color']}; background: rgba(26, 26, 46, 0.9);">
                    <h2 style="color: {recommendation['color']}; margin-bottom: 15px;">{recommendation['level']}</h2>
                    <h3 style="color: white; margin-bottom: 10px;">📋 {recommendation['action']}</h3>
                    <p style="color: rgba(255,255,255,0.8); margin-bottom: 15px;">{recommendation['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed recommendations
                st.markdown("#### 🎯 Recommended Actions:")
                for rec in recommendation['recommendations']:
                    st.markdown(f"• {rec}")
            
            with col_right:
                # Historical analysis
                historical_fig = create_historical_analysis(data)
                st.plotly_chart(historical_fig, use_container_width=True, key="historical_scatter")
            
            # Analytics section
            st.markdown("---")
            st.markdown("## 📊 ADVANCED ANALYTICS & MODEL INSIGHTS")
            
            col_analytics1, col_analytics2 = st.columns([1, 1])
            
            with col_analytics1:
                # Feature importance
                importance_fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="🔬 AI Model Feature Importance",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                importance_fig.update_layout(
                    plot_bgcolor="rgba(26, 26, 46, 0.8)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=NEOM_COLORS['text']),
                    height=350
                )
                st.plotly_chart(importance_fig, use_container_width=True, key="importance_chart")
            
            with col_analytics2:
                # Monthly risk distribution
                data['Month'] = pd.to_datetime(data['Date']).dt.month
                monthly_risk = data.groupby('Month')['Risk_Event'].agg(['count', 'sum']).reset_index()
                monthly_risk['Risk_Rate'] = (monthly_risk['sum'] / monthly_risk['count']) * 100
                
                monthly_fig = px.line(
                    monthly_risk,
                    x='Month',
                    y='Risk_Rate',
                    title="📈 Monthly Risk Event Rate",
                    markers=True,
                    color_discrete_sequence=[NEOM_COLORS['primary']]
                )
                monthly_fig.update_layout(
                    plot_bgcolor="rgba(26, 26, 46, 0.8)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=NEOM_COLORS['text']),
                    height=350
                )
                st.plotly_chart(monthly_fig, use_container_width=True, key="monthly_chart")
            
            # Data explorer
            with st.expander("🔍 **Historical Data Explorer**", expanded=False):
                st.dataframe(data.head(50), use_container_width=True)
                
                # Download processed data
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Dataset",
                    data=csv_data,
                    file_name="neom_bird_analysis_export.csv",
                    mime="text/csv"
                )
        
        else:
            st.error("❌ **System Error:** Unable to initialize NEOM Bio-Secure system")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.6);">
        <h4 style="color: {NEOM_COLORS['primary']};">NEOM Bio-Secure System v4.0</h4>
        <p>🚀 Powered by Advanced AI & Predictive Analytics | 🛡️ Protecting NEOM Airspace</p>
        <p>🌟 Smart City Aviation Safety Through Data Science Excellence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
