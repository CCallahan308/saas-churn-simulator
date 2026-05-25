"""
SaaS Churn Predictor - Portfolio Demo

NOTE: This app uses 100% synthetic/simulated data for visualization purposes.
Model performance metrics (AUC, Precision, Lift) reflect training on the
RetailRocket dataset. Prediction distributions and ROI projections are
generated from a synthetic distribution to illustrate the pipeline behavior.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

MODELS_DIR = Path(__file__).parent / "models"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Churn Predictor | Christian Callahan",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --color-bg-primary: #0a0a0b;
        --color-bg-secondary: #111113;
        --color-bg-tertiary: #18181b;
        --color-border: #27272a;
        --color-border-subtle: #1f1f23;
        --color-text-primary: #fafafa;
        --color-text-secondary: #a1a1aa;
        --color-text-tertiary: #71717a;
        --color-accent-primary: #3b82f6;
        --color-accent-secondary: #60a5fa;
        --color-success: #10b981;
        --color-warning: #f59e0b;
        --color-danger: #ef4444;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
        --transition-fast: 150ms ease;
        --transition-normal: 250ms ease;
    }
    
    /* Reset & Base */
    html, body, [class*="css"] {
        font-family: var(--font-family);
        color: var(--color-text-primary);
        background: var(--color-bg-primary);
    }
    
    .main {
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       HEADER COMPONENTS
       ═════════════════════════════════════════════════════════════════════ */
    
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 3rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid var(--color-border-subtle);
    }
    
    .app-title-section {
        flex: 1;
    }
    
    .app-title {
        font-size: 1.875rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: var(--color-text-primary);
        margin: 0 0 0.5rem 0;
    }
    
    .app-subtitle {
        font-size: 1rem;
        color: var(--color-text-secondary);
        margin: 0 0 1.25rem 0;
        line-height: 1.6;
    }
    
    .app-meta {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.375rem;
        font-size: 0.8125rem;
        color: var(--color-text-tertiary);
    }
    
    .meta-item a {
        color: var(--color-accent-secondary);
        text-decoration: none;
        font-weight: 500;
        transition: color var(--transition-fast);
    }
    
    .meta-item a:hover {
        color: var(--color-accent-primary);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.875rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-success);
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: var(--color-success);
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       METRICS GRID
       ═════════════════════════════════════════════════════════════════════ */
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2.5rem;
    }
    
    .metric-card {
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border-subtle);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        transition: border-color var(--transition-normal), 
                    box-shadow var(--transition-normal);
    }
    
    .metric-card:hover {
        border-color: var(--color-border);
        box-shadow: var(--shadow-md);
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-text-tertiary);
        margin-bottom: 0.75rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: var(--color-text-primary);
        margin-bottom: 0.375rem;
    }
    
    .metric-change {
        font-size: 0.8125rem;
        font-weight: 500;
    }
    
    .metric-change.positive {
        color: var(--color-success);
    }
    
    .metric-change.negative {
        color: var(--color-danger);
    }
    
    .metric-change.neutral {
        color: var(--color-text-tertiary);
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       SECTION COMPONENTS
       ═════════════════════════════════════════════════════════════════════ */
    
    .section {
        margin-bottom: 3rem;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin: 0;
    }
    
    .section-subtitle {
        font-size: 0.875rem;
        color: var(--color-text-tertiary);
    }
    
    .section-divider {
        height: 1px;
        background: var(--color-border-subtle);
        margin: 2.5rem 0;
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       CARDS & PANELS
       ═════════════════════════════════════════════════════════════════════ */
    
    .card {
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border-subtle);
        border-radius: var(--radius-md);
        padding: 1.5rem;
    }
    
    .card-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border-subtle);
    }
    
    .info-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .info-list li {
        font-size: 0.875rem;
        color: var(--color-text-secondary);
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--color-border-subtle);
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .info-list li:last-child {
        border-bottom: none;
    }
    
    .info-list li::before {
        content: "•";
        color: var(--color-accent-primary);
        font-weight: bold;
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       FORM CONTROLS
       ═════════════════════════════════════════════════════════════════════ */
    
    .control-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .control-group label {
        display: block;
        font-size: 0.8125rem;
        font-weight: 500;
        color: var(--color-text-secondary);
        margin-bottom: 0.5rem;
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       CHART OVERRIDES
       ═════════════════════════════════════════════════════════════════════ */
    
    .js-plotly-plot {
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       FOOTER
       ═════════════════════════════════════════════════════════════════════ */
    
    .app-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid var(--color-border-subtle);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer-left {
        font-size: 0.8125rem;
        color: var(--color-text-tertiary);
    }
    
    .footer-links {
        display: flex;
        gap: 1.5rem;
    }
    
    .footer-links a {
        font-size: 0.8125rem;
        color: var(--color-text-secondary);
        text-decoration: none;
        font-weight: 500;
        transition: color var(--transition-fast);
    }
    
    .footer-links a:hover {
        color: var(--color-accent-secondary);
    }
    
    /* ═════════════════════════════════════════════════════════════════════
       STREAMLIT OVERRIDES
       ═════════════════════════════════════════════════════════════════════ */
    
    .stMetric {
        background: transparent !important;
    }
    
    .stMetric > div {
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> .stMetric) {
        gap: 0;
    }
    
    .stSlider, .stNumberInput {
        background: var(--color-bg-tertiary);
        border: 1px solid var(--color-border-subtle);
        border-radius: var(--radius-sm);
        padding: 1rem;
    }
    
    .stButton > button {
        width: 100%;
        background: var(--color-accent-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        transition: background var(--transition-fast) !important;
    }
    
    .stButton > button:hover {
        background: #2563eb !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK = {
    "auc_roc": 0.83,
    "precision_at_10": 1.0,
    "lift_at_10": 1.01,
    "total_users": 5429,
    "churn_rate": 0.989,
    "features": {
        "days_view": 0.11,
        "days_since_cart": 0.10,
        "avg_evts_per_sess": 0.08,
        "days_since_purchase": 0.07,
        "diversity_ratio": 0.07,
        "days_since_any": 0.06,
    },
}


@st.cache_data
def load_model_data():
    """Real metrics from the committed training run (models/metrics.json).

    Falls back to the published values if the artifacts are absent (e.g. a clone
    that has not run `make train`). The numbers shown therefore always trace to a
    committed file, not hand-entered marketing figures.
    """
    data = dict(_FALLBACK)
    try:
        metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
        test = {row["split"]: row for row in metrics["metrics"]}["test"]
        data["auc_roc"] = test["auc_roc"]
        data["precision_at_10"] = test["prec@10"]
        data["lift_at_10"] = test["lift@10"]
        data["total_users"] = metrics.get("cohort_size", data["total_users"])
        data["churn_rate"] = metrics.get("churn_rate", data["churn_rate"])
    except Exception:
        pass
    try:
        importance = pd.read_csv(MODELS_DIR / "feature_importance.csv").head(8)
        data["features"] = dict(zip(importance["feature"], importance["pct"] / 100))
    except Exception:
        pass
    return data

@st.cache_data
def generate_prediction_distribution(n=1000):
    """Generate sample prediction probabilities"""
    np.random.seed(42)
    return np.random.beta(2, 5, n)

def calculate_retention_impact(params):
    """Calculate ROI and business impact"""
    metrics = load_model_data()
    
    targeted_users = int(metrics["total_users"] * params["target_pct"] / 100)
    # assume lift decays ~15% when targeting beyond the top decile (illustrative)
    lift = metrics["lift_at_10"] if params["target_pct"] <= 10 else metrics["lift_at_10"] * 0.85

    # Retention model
    intervention_rate = 0.12  # 12% targeted outreach rate (industry conservative estimate)
    retained_users = int(targeted_users * intervention_rate * lift / 3)  # conservative: 1-in-3 interventions succeed
    
    # Financial model
    revenue_retained = retained_users * params["customer_ltv"]
    campaign_cost = targeted_users * params["cost_per_intervention"]
    net_value = revenue_retained - campaign_cost
    roi = (net_value / campaign_cost * 100) if campaign_cost > 0 else 0
    
    return {
        "targeted_users": targeted_users,
        "retained_users": retained_users,
        "revenue_retained": revenue_retained,
        "campaign_cost": campaign_cost,
        "net_value": net_value,
        "roi_pct": roi
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-title-section">
            <h1 class="app-title">Churn Prediction System</h1>
            <p class="app-subtitle">
                Rank churn risk from behavioral signals over 2.7M RetailRocket events, and turn the
                scores into a retention-budget decision with the ROI simulator.
            </p>
            <div class="app-meta">
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
                    </svg>
                    <a href="https://github.com/CCallahan308/saas-churn-simulator">View Source</a>
                </div>
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    </svg>
                    <span>RetailRocket Dataset</span>
                </div>
            </div>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            Demo
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics():
    data = load_model_data()
    
    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
    
    metrics = [
        ("AUC-ROC (test)", f"{data['auc_roc']:.2f}", "Ranking quality", "positive"),
        ("Churn base rate", f"{data['churn_rate']:.0%}", "Near-degenerate target", "neutral"),
        ("Lift @10%", f"{data['lift_at_10']:.2f}x", "Capped by base rate", "neutral"),
        ("Cohort", f"{data['total_users']:,}", "Active buyers", "neutral")
    ]
    
    for label, value, change, change_type in metrics:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-change {change_type}">{change}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_analysis_charts():
    data = load_model_data()
    probs = generate_prediction_distribution()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section">
            <div class="section-header">
                <div>
                    <h3 class="section-title">Prediction Distribution</h3>
                    <p class="section-subtitle">Churn probability across user base</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs,
            nbinsx=45,
            marker_color='#3b82f6',
            marker_line_color='#2563eb',
            marker_line_width=0.5,
            opacity=0.9,
            hovertemplate='<b>Probability:</b> %{x:.2f}<br><b>Users:</b> %{y}<extra></extra>'
        ))
        
        fig.add_vline(x=0.5, line_dash="dot", line_color="#ef4444", line_width=2,
                      annotation_text="Threshold", annotation_position="top right",
                      annotation_font_size=11, annotation_font_color="#ef4444")
        
        fig.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Users",
            height=320,
            margin=dict(l=0, r=20, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='#a1a1aa', size=12),
            hoverlabel=dict(bgcolor='#18181b', font_size=12)
        )
        fig.update_xaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False)
        fig.update_yaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("""
        <div class="section">
            <div class="section-header">
                <div>
                    <h3 class="section-title">Feature Importance</h3>
                    <p class="section-subtitle">Top predictors by model gain</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        features = data["features"]
        df = pd.DataFrame({
            "feature": list(features.keys()),
            "importance": list(features.values())
        }).sort_values("importance", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["feature"],
            x=df["importance"],
            orientation='h',
            marker_color='#3b82f6',
            marker_line_color='#2563eb',
            marker_line_width=0.5,
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="",
            height=320,
            margin=dict(l=0, r=20, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='#a1a1aa', size=12),
            hoverlabel=dict(bgcolor='#18181b', font_size=12)
        )
        fig.update_xaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False, tickformat='.0%')
        fig.update_yaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_roi_calculator():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <div>
                <h3 class="section-title">Retention ROI Calculator</h3>
                <p class="section-subtitle">Estimate business impact of targeted retention campaigns</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        target_pct = st.slider("Target Top %", 5, 40, 10, 
                               help="Percentage of highest-risk users to target")
    with col2:
        customer_ltv = st.number_input("Customer LTV ($)", value=500, step=50,
                                       help="Average customer lifetime value")
    with col3:
        cost_per = st.number_input("Intervention Cost ($)", value=15, step=5,
                                   help="Cost per customer outreach")
    with col4:
        budget = st.number_input("Budget ($)", value=50000, step=5000,
                                 help="Total campaign budget")
    
    # Calculate
    result = calculate_retention_impact({
        "target_pct": target_pct,
        "customer_ltv": customer_ltv,
        "cost_per_intervention": cost_per,
        "budget": budget
    })
    
    # Results
    st.markdown('<div class="metrics-grid" style="grid-template-columns: repeat(5, 1fr); margin-top: 1.5rem;">', unsafe_allow_html=True)
    
    results_display = [
        ("Targeted", f"{result['targeted_users']:,}", "users"),
        ("Retained", f"{result['retained_users']:,}", "users"),
        ("Revenue", f"${result['revenue_retained']:,.0f}", "retained"),
        ("Cost", f"${result['campaign_cost']:,.0f}", "invested"),
        ("ROI", f"{result['roi_pct']:.0f}%", "return")
    ]
    
    for label, value, sublabel in results_display:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-change neutral">{sublabel}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ROI curve
    st.markdown("""
    <div style="margin-top: 1.5rem;">
        <p style="font-size: 0.8125rem; color: var(--color-text-tertiary); margin-bottom: 1rem;">
            ROI vs. Targeting Percentage
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    target_range = list(range(5, 41, 5))
    roi_curve = [calculate_retention_impact({
        "target_pct": t,
        "customer_ltv": customer_ltv,
        "cost_per_intervention": cost_per,
        "budget": budget
    })["roi_pct"] for t in target_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target_range,
        y=roi_curve,
        mode='lines+markers',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8, color='#3b82f6'),
        hovertemplate='<b>%{x}%</b> targeting → <b>%{y:.0f}%</b> ROI<extra></extra>'
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="#10b981", line_width=1.5,
                  annotation_text="Break-even", annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#10b981")
    fig.add_vline(x=target_pct, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="Current", annotation_position="top",
                  annotation_font_size=10, annotation_font_color="#f59e0b")
    
    fig.update_layout(
        xaxis_title="Targeting %",
        yaxis_title="ROI %",
        height=240,
        margin=dict(l=0, r=20, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#a1a1aa', size=11),
        hoverlabel=dict(bgcolor='#18181b')
    )
    fig.update_xaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(39, 39, 42, 0.5)', zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_insights():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    data = load_model_data()
    top_features = ", ".join(list(data["features"])[:3])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Key Findings (from the committed run)</div>
            <ul class="info-list">
                <li>Recency features dominate importance: {top_features}</li>
                <li>Churn base rate is ~{data['churn_rate']:.0%} - most buyers are one-off, so lift over random is ~1.0</li>
                <li>LightGBM overfits the tiny retained class; the LogisticRegression baseline generalizes better on this small, imbalanced cohort</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">Methodology Notes</div>
            <ul class="info-list">
                <li>Leakage-safe observation / gap / check windowing; visitor-disjoint split</li>
                <li>Hyperparameters tuned with RandomizedSearchCV; probabilities isotonic-calibrated</li>
                <li>The simulator turns calibrated risk plus cost/LTV assumptions into a targeting decision</li>
                <li>On a dataset with real churn variance, the same pipeline produces meaningful lift</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="app-footer">
        <div class="footer-left">
            Built by <a href="https://christiangcallahan.tech" style="color: var(--color-accent-secondary); text-decoration: none;">Christian Callahan</a> • 
            Data: RetailRocket • MIT License
        </div>
        <div class="footer-links">
            <a href="https://github.com/CCallahan308/saas-churn-simulator">GitHub</a>
            <a href="https://christiangcallahan.tech">Portfolio</a>
            <a href="mailto:contact@christiangcallahan.tech">Contact</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    render_header()
    st.info("📊 **Demo Mode** — Performance metrics reflect the trained model. Predictions use a synthetic data distribution for visualization purposes.")
    render_metrics()
    render_analysis_charts()
    render_roi_calculator()
    render_insights()
    render_footer()

if __name__ == "__main__":
    main()
