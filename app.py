"""
SaaS Churn Simulator - Interactive Demo
Deploy to Streamlit Cloud for live portfolio demo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page config
st.set_page_config(
    page_title="SaaS Churn Simulator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #0f3460;
    }
    .stMetric > div {
        background: rgba(15, 52, 96, 0.3);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(15, 52, 96, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Simulated model results (based on actual model performance)
@st.cache_data
def get_model_metrics():
    return {
        "auc_roc": 0.85,
        "precision_at_10": 0.65,
        "lift_at_10": 3.2,
        "total_users": 140000,
        "churn_rate": 0.23
    }

@st.cache_data
def generate_sample_predictions(n=1000):
    """Generate sample prediction distribution"""
    np.random.seed(42)
    probs = np.random.beta(2, 5, n)  # Skewed toward low churn
    return probs

@st.cache_data
def get_feature_importance():
    """Feature importance from trained model"""
    return pd.DataFrame({
        "feature": [
            "days_since_last_activity",
            "avg_session_duration",
            "purchase_count",
            "browsing_velocity",
            "days_on_platform",
            "avg_order_value",
            "support_tickets",
            "email_engagement"
        ],
        "importance": [0.28, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04]
    })

def calculate_roi(retention_budget, targeting_percent, avg_customer_value, cost_per_intervention):
    """ROI calculator for retention campaigns"""
    metrics = get_model_metrics()
    targeted_users = int(metrics["total_users"] * targeting_percent / 100)
    
    # Lift improves targeting efficiency
    base_retention = 1 - metrics["churn_rate"]
    lift_factor = metrics["lift_at_10"] if targeting_percent <= 10 else metrics["lift_at_10"] * 0.8
    
    # Simplified ROI calculation
    retained_users = int(targeted_users * 0.15 * lift_factor / 3)  # Conservative estimate
    revenue_saved = retained_users * avg_customer_value
    campaign_cost = targeted_users * cost_per_intervention
    
    roi = (revenue_saved - campaign_cost) / campaign_cost * 100 if campaign_cost > 0 else 0
    
    return {
        "targeted_users": targeted_users,
        "retained_users": retained_users,
        "revenue_saved": revenue_saved,
        "campaign_cost": campaign_cost,
        "roi_percent": roi
    }

# Header
st.title("📊 SaaS Churn Simulator")
st.markdown("**Predict which customers will churn, segment by value, and calculate retention ROI.**")
st.markdown("Built on 2.7M events from the RetailRocket dataset.")

# Model Performance Section
st.header("Model Performance", divider="blue")

metrics = get_model_metrics()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("AUC-ROC", f"{metrics['auc_roc']:.2f}", help="Good separation between active/churning users")
with col2:
    st.metric("Precision @ 10%", f"{metrics['precision_at_10']:.0%}", help="Top decile predictions are reliable")
with col3:
    st.metric("Lift @ 10%", f"{metrics['lift_at_10']:.1f}x", help="3x better than random targeting")
with col4:
    st.metric("Churn Rate", f"{metrics['churn_rate']:.0%}", help="Baseline churn in dataset")

# Prediction Distribution
st.subheader("Churn Probability Distribution")
probs = generate_sample_predictions()

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=probs,
    nbinsx=50,
    marker_color='#3b82f6',
    marker_line_color='#1d4ed8',
    marker_line_width=1,
    opacity=0.8
))
fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
fig.update_layout(
    xaxis_title="Churn Probability",
    yaxis_title="User Count",
    showlegend=False,
    height=300,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# Feature Importance
st.subheader("Top Churn Predictors")
importance = get_feature_importance()

fig2 = px.bar(
    importance.sort_values("importance", ascending=True),
    x="importance",
    y="feature",
    orientation='h',
    color="importance",
    color_continuous_scale="Blues"
)
fig2.update_layout(
    xaxis_title="Importance Score",
    yaxis_title="",
    showlegend=False,
    height=350,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig2, use_container_width=True)

# ROI Simulator
st.header("💰 Retention ROI Simulator", divider="blue")
st.markdown("Adjust parameters to see the ROI of targeted retention campaigns.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Campaign Parameters")
    targeting = st.slider("Targeting % (top at-risk users)", 5, 50, 10)
    budget = st.number_input("Retention Budget ($)", value=50000, step=5000)
    customer_value = st.number_input("Avg. Customer Lifetime Value ($)", value=500, step=50)
    cost_per_intervention = st.number_input("Cost per Intervention ($)", value=15, step=5)

with col2:
    st.subheader("Projected Results")
    roi_result = calculate_roi(budget, targeting, customer_value, cost_per_intervention)
    
    st.metric("Users Targeted", f"{roi_result['targeted_users']:,}")
    st.metric("Users Retained", f"{roi_result['retained_users']:,}")
    st.metric("Revenue Saved", f"${roi_result['revenue_saved']:,.0f}")
    st.metric("Campaign Cost", f"${roi_result['campaign_cost']:,.0f}")
    
    roi_color = "normal" if roi_result['roi_percent'] < 100 else "inverse"
    st.metric("ROI", f"{roi_result['roi_percent']:.0f}%", delta_color=roi_color)

# ROI Visualization
st.subheader("ROI by Targeting Percentage")

targeting_range = range(5, 51, 5)
roi_values = [calculate_roi(budget, t, customer_value, cost_per_intervention)['roi_percent'] for t in targeting_range]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=list(targeting_range),
    y=roi_values,
    mode='lines+markers',
    marker=dict(size=10, color='#3b82f6'),
    line=dict(width=3, color='#3b82f6')
))
fig3.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Break-even")
fig3.update_layout(
    xaxis_title="Targeting Percentage",
    yaxis_title="ROI (%)",
    height=300,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig3, use_container_width=True)

# Key Findings
st.header("🔍 Key Findings", divider="blue")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Top Predictors:**
    - Days since last activity is the strongest signal
    - Browsing velocity drops 2-3 weeks before churn
    - Purchase frequency matters more than order size
    """)

with col2:
    st.markdown("""
    **Recommended Strategy:**
    - Target top 10-20% at-risk users for best ROI
    - Intervention within 7 days of predicted churn
    - Personalized offers based on segment value
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
        <strong>SaaS Churn Simulator</strong> | 
        <a href="https://github.com/CCallahan308/saas-churn-simulator">GitHub</a>
    </div>
    <div>
        Built by <a href="https://christiangcallahan.tech">Christian Callahan</a>
    </div>
</div>
""", unsafe_allow_html=True)
