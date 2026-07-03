import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

# Inject Custom CSS with 8px grid spacing, CSS variables, and reduced motion settings
st.markdown("""
    <style>
    :root {
        --primary-accent: #2563EB;
        --secondary-accent: #0EA5E9;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-color: #0F172A;
    }
    
    /* Global reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-delay: 0s !important;
            animation-duration: 0s !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0s !important;
            scroll-behavior: auto !important;
        }
    }
    
    .dashboard-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        text-align: left;
        height: 100%;
    }
    
    .kpi-title {
        color: #64748B;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        color: var(--text-color);
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.25;
    }
    
    .kpi-subtitle {
        color: var(--secondary-accent);
        font-size: 0.8rem;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 32px;
        margin-bottom: 16px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    
    /* Action plan box styles */
    .action-box {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 4px solid var(--primary-accent);
        background-color: #F8FAFC;
    }
    .action-box.high {
        border-left-color: var(--danger);
    }
    .action-box.medium {
        border-left-color: var(--warning);
    }
    .action-box.low {
        border-left-color: var(--success);
    }
    .action-title {
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
    }
    .action-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<header>
    <h1 style="color:#0F172A; margin-bottom: 8px;">Churn Prediction</h1>
    <p style="color:#64748B; font-size:1.1rem; margin-top:0; margin-bottom:24px;">
        Identify customers at high risk of attrition and deploy proactive retention campaigns.
    </p>
</header>
""", unsafe_allow_html=True)

# Helper function to load prediction data
@st.cache_data
def load_churn_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pred_path = os.path.join(base_dir, "churn_prediction_true", "predictions", "customer_true_churn_predictions.csv")
    fallback_pred_paths = [
        pred_path,
        "churn_prediction_true/predictions/customer_true_churn_predictions.csv",
        "../churn_prediction_true/predictions/customer_true_churn_predictions.csv"
    ]
    
    # Try precomputed file first
    for path in fallback_pred_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, False  # Did not use fallback
            except Exception as e:
                pass
                
    # Fallback to model inference
    model_paths = [
        os.path.join(base_dir, "churn_prediction_true", "models", "lightgbm_model.pkl"),
        "churn_prediction_true/models/lightgbm_model.pkl",
        "../churn_prediction_true/models/lightgbm_model.pkl"
    ]
    dataset_paths = [
        os.path.join(base_dir, "churn_prediction_true", "datasets", "true_churn_dataset.csv"),
        "churn_prediction_true/datasets/true_churn_dataset.csv",
        "../churn_prediction_true/datasets/true_churn_dataset.csv"
    ]
    
    model_file = None
    for p in model_paths:
        if os.path.exists(p):
            model_file = p
            break
            
    dataset_file = None
    for p in dataset_paths:
        if os.path.exists(p):
            dataset_file = p
            break
            
    if not model_file or not dataset_file:
        st.error("Churn prediction predictions file, model, or dataset could not be found.")
        st.stop()
        
    try:
        import pickle
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        df = pd.read_csv(dataset_file)
        features = [col for col in df.columns if col not in ['customer_id', 'true_churn_flag']]
        X_input = df[features].values
        
        # Generate probabilities
        probs = model.predict_proba(X_input)[:, 1]
        
        pred_df = pd.DataFrame({
            'customer_id': df['customer_id'],
            'churn_probability': probs,
            'true_churn_flag': df['true_churn_flag'],
            'recency': df['recency'],
            'frequency': df['frequency'],
            'monetary': df['monetary'],
            'customer_loyalty_score': df['customer_loyalty_score']
        })
        pred_df['risk_category'] = np.where(
            pred_df['churn_probability'] < 0.30,
            'Low Risk',
            np.where(
                pred_df['churn_probability'] < 0.70,
                'Medium Risk',
                'High Risk'
            )
        )
        return pred_df, True
    except Exception as e:
        st.error(f"Error executing fallback LightGBM inference: {e}")
        st.stop()

# Load churn predictions
df, used_fallback = load_churn_data()

if used_fallback:
    st.sidebar.warning("⚠️ Using fallback LightGBM model inference.")
else:
    st.sidebar.success("✅ Loaded precomputed predictions.")

# -----------------
# 1. KPI CARDS SECTION
# -----------------
st.markdown("<section><div class='section-header'>Churn Risk Overview</div></section>", unsafe_allow_html=True)

total_customers = df['customer_id'].nunique()
high_risk_customers = (df['risk_category'] == 'High Risk').sum()
avg_churn_prob = df['churn_probability'].mean()

col_k1, col_k2, col_k3 = st.columns(3)

with col_k1:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Total Customers Evaluated</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-subtitle">Active cohort base</div>
        </article>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Total High-Risk Customers</div>
            <div class="kpi-value" style="color:var(--danger);">{high_risk_customers:,}</div>
            <div class="kpi-subtitle" style="color:var(--danger);">{high_risk_customers/total_customers*100:.1f}% of cohort</div>
        </article>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Avg. Churn Probability</div>
            <div class="kpi-value">{avg_churn_prob*100:.2f}%</div>
            <div class="kpi-subtitle">Global cohort probability</div>
        </article>
    """, unsafe_allow_html=True)

# Model Performance metrics
st.markdown("<section><div class='section-header'>Model Evaluation Performance</div></section>", unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown("""
        <article class="kpi-card">
            <div class="kpi-title">Model Accuracy</div>
            <div class="kpi-value">65.01%</div>
            <div class="kpi-subtitle">Overall correctness</div>
        </article>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
        <article class="kpi-card">
            <div class="kpi-title">Model Precision</div>
            <div class="kpi-value">59.94%</div>
            <div class="kpi-subtitle">Relevance of risk alerts</div>
        </article>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown("""
        <article class="kpi-card">
            <div class="kpi-title">Model Recall</div>
            <div class="kpi-value">62.31%</div>
            <div class="kpi-subtitle">Percentage of churners caught</div>
        </article>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown("""
        <article class="kpi-card">
            <div class="kpi-title">Model ROC AUC</div>
            <div class="kpi-value">71.77%</div>
            <div class="kpi-subtitle">Prediction strength</div>
        </article>
    """, unsafe_allow_html=True)

# -----------------
# 2. CHARTS SECTION
# -----------------
st.markdown("<section><div class='section-header'>Risk Segmentation & Feature Drivers</div></section>", unsafe_allow_html=True)

col_ch1, col_ch2 = st.columns(2)

with col_ch1:
    # Churn Risk Distribution Donut Chart
    risk_df = df['risk_category'].value_counts().reset_index()
    risk_df.columns = ['Risk Category', 'Count']
    
    fig_risk = px.pie(
        risk_df,
        names='Risk Category',
        values='Count',
        hole=0.4,
        color='Risk Category',
        color_discrete_map={
            'Low Risk': '#22C55E',
            'Medium Risk': '#F59E0B',
            'High Risk': '#EF4444'
        }
    )
    fig_risk.update_traces(
        textinfo='percent+label',
        pull=[0.02, 0.02, 0.02],
        textfont_size=12
    )
    fig_risk.update_layout(
        showlegend=False,
        margin=dict(t=24, b=24, l=16, r=16),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A'),
        height=320
    )
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:1.1rem; color:#0F172A; margin-top:0; margin-bottom:16px;'>Risk Cohort Share</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_ch2:
    # Feature Importances Horizontal Bar Chart
    importance_data = {
        'Feature': [
            'Quantity Per Order',
            'Average Purchase Value',
            'Recency',
            'Customer Tenure',
            'Monetary (Total Spend)',
            'Avg Days Between Purchases',
            'Customer Rank by Revenue',
            'Weekend Sales Ratio',
            'Customer Loyalty Score',
            'Purchase Frequency'
        ],
        'Importance': [513, 483, 426, 324, 294, 251, 162, 162, 161, 115],
        'Influence Type': [
            'Retention Driver (Loyalty)',
            'Retention Driver (Loyalty)',
            'Churn Driver (Risk)',
            'Retention Driver (Loyalty)',
            'Retention Driver (Loyalty)',
            'Churn Driver (Risk)',
            'Churn Driver (Risk)',
            'Retention Driver (Loyalty)',
            'Retention Driver (Loyalty)',
            'Retention Driver (Loyalty)'
        ]
    }
    imp_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(
        imp_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Influence Type',
        color_discrete_map={
            'Retention Driver (Loyalty)': '#22C55E',
            'Churn Driver (Risk)': '#EF4444'
        }
    )
    fig_imp.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(t=24, b=24, l=16, r=16),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A'),
        xaxis=dict(gridcolor='#F1F5F9', title='Gini Importance Score'),
        yaxis=dict(title=''),
        height=320
    )
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:1.1rem; color:#0F172A; margin-top:0; margin-bottom:16px;'>Top Churn & Retention Drivers</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# 3. CUSTOMER RISK TABLE
# -----------------
st.markdown("<section><div class='section-header'>High Churn Risk Customers</div></section>", unsafe_allow_html=True)

# Order table by churn probability descending
table_df = df.sort_values(by='churn_probability', ascending=False).copy()
table_df['churn_probability'] = table_df['churn_probability'].apply(lambda x: f"{x*100:.2f}%")
table_df['monetary'] = table_df['monetary'].apply(lambda x: f"${x:,.2f}")
table_df['customer_loyalty_score'] = table_df['customer_loyalty_score'].round(2)

table_df.columns = [
    'Customer ID', 'Churn Probability', 'True Churn Flag', 'Recency (Days)', 
    'Frequency (Orders)', 'Monetary Spend', 'Loyalty Score', 'Risk Category'
]

display_table = table_df[[
    'Customer ID', 'Churn Probability', 'Risk Category', 
    'Recency (Days)', 'Frequency (Orders)', 'Monetary Spend', 'Loyalty Score'
]]

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# 4. BUSINESS RECOMMENDATIONS PANEL
# -----------------
st.markdown("<section><div class='section-header'>Mitigation Strategy & Action Plan</div></section>", unsafe_allow_html=True)

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    st.markdown("""
        <section class="action-box high">
            <div class="action-title">
                <span>🔴 High-Risk Target</span>
                <span>(Prob &ge; 70%)</span>
            </div>
            <div class="action-desc">
                <strong>Target Cohort:</strong> Customers displaying critical churn indicators.<br>
                <strong>Actual Churn Rate:</strong> <span style="color:var(--danger); font-weight:700;">91.58%</span><br>
                <strong>Mitigation Strategy:</strong>
                <ul style="margin: 8px 0 0 0; padding-left: 16px;">
                    <li>Deploy aggressive win-back campaigns immediately.</li>
                    <li>Initiate direct phone or personal sales outreach.</li>
                    <li>Offer high-value compensation rewards or deep discount coupons.</li>
                </ul>
            </div>
        </section>
    """, unsafe_allow_html=True)

with col_rec2:
    st.markdown("""
        <section class="action-box medium">
            <div class="action-title">
                <span>🟡 Medium-Risk Target</span>
                <span>(30% &le; Prob &lt; 70%)</span>
            </div>
            <div class="action-desc">
                <strong>Target Cohort:</strong> Transitioning/drifting customers.<br>
                <strong>Actual Churn Rate:</strong> <span style="color:var(--warning); font-weight:700;">43.31%</span><br>
                <strong>Mitigation Strategy:</strong>
                <ul style="margin: 8px 0 0 0; padding-left: 16px;">
                    <li>Target with personalized re-engagement marketing newsletters.</li>
                    <li>Deploy feedback surveys to identify friction points.</li>
                    <li>Recommend popular accessory lines based on past purchase categories.</li>
                </ul>
            </div>
        </section>
    """, unsafe_allow_html=True)

with col_rec3:
    st.markdown("""
        <section class="action-box low">
            <div class="action-title">
                <span>🟢 Low-Risk Target</span>
                <span>(Prob &lt; 30%)</span>
            </div>
            <div class="action-desc">
                <strong>Target Cohort:</strong> Stable, highly active, loyal customers.<br>
                <strong>Actual Churn Rate:</strong> <span style="color:var(--success); font-weight:700;">3.72%</span><br>
                <strong>Mitigation Strategy:</strong>
                <ul style="margin: 8px 0 0 0; padding-left: 16px;">
                    <li>Maintain standard marketing schedules.</li>
                    <li>Send new product arrival alerts.</li>
                    <li><strong>No coupon incentives</strong> required; preserve margin by avoiding unnecessary discount offers.</li>
                </ul>
            </div>
        </section>
    """, unsafe_allow_html=True)
