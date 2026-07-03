import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime

from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RetailPulse | Alerts & Monitoring",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

require_auth()
render_sidebar()

# ==========================================
# THEME CONFIGURATION
# ==========================================
THEME = {
    "primary": "#2563EB",
    "secondary": "#0EA5E9",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text": "#0F172A",
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "abc_colors": {"A": "#2563EB", "B": "#F59E0B", "C": "#0EA5E9"},
    "risk_colors": {"Critical": "#EF4444", "High": "#F59E0B", "Medium": "#0EA5E9", "Low": "#22C55E"},
    "churn_colors": {"High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#22C55E"}
}

st.markdown("""
    <style>
    /* Card Styles */
    .alert-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
        height: 100%;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        text-align: left;
        height: 100%;
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-subtitle {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 32px;
        margin-bottom: 16px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_all_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 1. Churn Data
    churn_path = os.path.join(base_dir, "churn_prediction_true", "predictions", "customer_true_churn_predictions.csv")
    churn_df = pd.read_csv(churn_path) if os.path.exists(churn_path) else pd.DataFrame()
    
    # 2. Inventory Data
    inv_dir = os.path.join(base_dir, "inventory_optimization", "outputs")
    reorder_path = os.path.join(inv_dir, "reorder_recommendations.csv")
    master_path = os.path.join(inv_dir, "inventory_master.csv")
    reorder_df = pd.read_csv(reorder_path) if os.path.exists(reorder_path) else pd.DataFrame()
    master_df = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()
    
    # 3. Demand Data
    demand_path = os.path.join(base_dir, "Demand_Forecasting", "datasets", "weekly_forecast_dashboard_data.json")
    demand_data = {}
    if os.path.exists(demand_path):
        with open(demand_path, 'r') as f:
            demand_data = json.load(f)
            
    # Get last updated timestamp from files (use latest)
    paths = [churn_path, reorder_path, demand_path]
    valid_paths = [p for p in paths if os.path.exists(p)]
    
    last_updated = None
    if valid_paths:
        latest_time = max(os.path.getmtime(p) for p in valid_paths)
        last_updated = datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
            
    return churn_df, reorder_df, master_df, demand_data, last_updated

try:
    churn_df, reorder_df, master_df, demand_data, last_updated = load_all_data()
except Exception as e:
    st.error(f"Error loading dashboard data: {str(e)}")
    st.stop()

# ==========================================
# ALERT CALCULATIONS
# ==========================================

# 1. Churn Alerts
if not churn_df.empty:
    high_risk_churn_df = churn_df[churn_df['churn_probability'] > 0.70]
    total_high_risk_customers = len(high_risk_churn_df)
    revenue_at_risk = high_risk_churn_df['monetary'].sum()
else:
    total_high_risk_customers = 0
    revenue_at_risk = 0

# 2. Inventory Alerts
if not reorder_df.empty:
    critical_products_df = reorder_df[reorder_df['risk_category'] == 'Critical']
    critical_product_count = len(critical_products_df)
    immediate_reorder_count = len(reorder_df[reorder_df['reorder_urgency'] >= 0.80])
    total_reorders = len(reorder_df)
else:
    critical_product_count = 0
    immediate_reorder_count = 0
    total_reorders = 0

# 3. Demand Alerts
demand_surges = []
demand_drops = []
holiday_peaks = []
next_surge_week = "N/A"
expected_increase = 0
forecast_trend = "Stable"

if demand_data and 'future' in demand_data:
    future_df = pd.DataFrame(demand_data['future'])
    if not future_df.empty:
        # Determine trend slope
        x_idx = np.arange(len(future_df))
        y_val = future_df['forecast'].values
        slope, _ = np.polyfit(x_idx, y_val, 1)
        if slope > 5000:
            forecast_trend = "Strong Growth"
        elif slope > 1000:
            forecast_trend = "Moderate Growth"
        elif slope < -5000:
            forecast_trend = "Strong Decline"
        elif slope < -1000:
            forecast_trend = "Moderate Decline"
            
        for i, row in future_df.iterrows():
            if i == 0:
                wow_growth = row['growth_from_last_historical']
            else:
                prev_val = future_df.iloc[i-1]['forecast']
                wow_growth = (row['forecast'] - prev_val) / prev_val * 100
                
            if wow_growth > 15:
                demand_surges.append({'date': row['date'], 'growth': wow_growth, 'forecast': row['forecast']})
            elif wow_growth < -15:
                demand_drops.append({'date': row['date'], 'growth': wow_growth, 'forecast': row['forecast']})
                
            # Assume week of 12-19 to 12-25 is holiday peak
            if '12-19' in row['date'] or '12-20' in row['date'] or '12-21' in row['date'] or '12-22' in row['date']:
                holiday_peaks.append({'date': row['date'], 'forecast': row['forecast']})

        if demand_surges:
            next_surge = demand_surges[0]
            next_surge_week = next_surge['date']
            expected_increase = next_surge['growth']

# Summary Aggregations
critical_alerts = (1 if total_high_risk_customers > 100 else 0) + (1 if critical_product_count > 50 else 0) + len(demand_drops)
opportunity_alerts = len(demand_surges) + len(holiday_peaks)
warning_alerts = (1 if 0 < total_high_risk_customers <= 100 else 0) + (1 if 0 < critical_product_count <= 50 else 0) + (1 if immediate_reorder_count > 0 else 0)
total_active_alerts = critical_alerts + warning_alerts + opportunity_alerts

# ==========================================
# HEADER
# ==========================================
st.title("🚨 Alerts & Monitoring System")

st.markdown(f"<p style='color:{THEME['text']}; font-size:1.1rem; margin-top:-1rem;'>Centralized risk identification and opportunity monitoring.</p>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# ALERT KPI DASHBOARD (ENHANCEMENT 1)
# ==========================================
st.markdown("<div class='section-header' style='margin-top:0;'>Alert Status Overview</div>", unsafe_allow_html=True)

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid {THEME['primary']};">
            <div class="kpi-title">Active Alerts</div>
            <div class="kpi-value">{total_active_alerts}</div>
            <div class="kpi-subtitle">Total system notifications</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid {THEME['danger']};">
            <div class="kpi-title" style="color: {THEME['danger']};">Critical Alerts</div>
            <div class="kpi-value">{critical_alerts}</div>
            <div class="kpi-subtitle">Immediate action required</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid {THEME['warning']};">
            <div class="kpi-title" style="color: {THEME['warning']};">Warning Alerts</div>
            <div class="kpi-value">{warning_alerts}</div>
            <div class="kpi-subtitle">Monitor closely</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 5px solid {THEME['success']};">
            <div class="kpi-title" style="color: {THEME['success']};">Opportunity Alerts</div>
            <div class="kpi-value">{opportunity_alerts}</div>
            <div class="kpi-subtitle">Potential revenue upside</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# ALERT TYPE 4: EXECUTIVE ALERTS SUMMARY
# ==========================================
st.markdown("<div class='section-header' style='margin-top:0;'>Executive Priority Alerts</div>", unsafe_allow_html=True)

exec_col1, exec_col2, exec_col3 = st.columns(3)

with exec_col1:
    if critical_product_count > 0:
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['danger']};">
            <h4 style="margin-top:0; color:{THEME['danger']};">🔴 Critical Inventory Risk</h4>
            <p style="color:#475569; font-size:0.9rem;"><strong>Impact:</strong> {critical_product_count} critical products risk immediate stockout. Severe revenue impact.</p>
            <p style="color:#475569; font-size:0.9rem;"><strong>Action:</strong> Review reorder recommendations and expedite POs for Class A items.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['success']};">
            <h4 style="margin-top:0; color:{THEME['success']};">🟢 Inventory Stable</h4>
            <p style="color:#475569; font-size:0.9rem;">No critical inventory risks detected.</p>
        </div>
        """, unsafe_allow_html=True)

with exec_col2:
    if total_high_risk_customers > 0:
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['warning']};">
            <h4 style="margin-top:0; color:{THEME['warning']};">🟡 Elevated Churn Risk</h4>
            <p style="color:#475569; font-size:0.9rem;"><strong>Impact:</strong> ${revenue_at_risk:,.2f} revenue at risk from {total_high_risk_customers} customers likely to churn (>70% prob).</p>
            <p style="color:#475569; font-size:0.9rem;"><strong>Action:</strong> Deploy targeted retention and win-back campaigns immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['success']};">
            <h4 style="margin-top:0; color:{THEME['success']};">🟢 Churn Stable</h4>
            <p style="color:#475569; font-size:0.9rem;">No high-risk churn clusters detected.</p>
        </div>
        """, unsafe_allow_html=True)

with exec_col3:
    if demand_surges or holiday_peaks:
        peak_msg = f"Surge expected week of {next_surge_week} (+{expected_increase:.1f}%)." if demand_surges else "Upcoming holiday peaks detected."
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['success']};">
            <h4 style="margin-top:0; color:{THEME['success']};">🟢 Demand Growth Opportunity</h4>
            <p style="color:#475569; font-size:0.9rem;"><strong>Impact:</strong> {peak_msg}</p>
            <p style="color:#475569; font-size:0.9rem;"><strong>Action:</strong> Ensure supply chain readiness and scale up inventory buffers.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card" style="border-top: 4px solid {THEME['secondary']};">
            <h4 style="margin-top:0; color:{THEME['secondary']};">⚪ Normal Demand</h4>
            <p style="color:#475569; font-size:0.9rem;">Forecast remains steady with no extreme anomalies.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# ALERT TYPE 1: CHURN ALERTS
# ==========================================
st.markdown("<div class='section-header'>🚨 Churn Risk Alerts</div>", unsafe_allow_html=True)

ch_col1, ch_col2, ch_col3 = st.columns([1, 1, 2])
with ch_col1:
    st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:10px;">
            <div class="kpi-title">High Risk Customers</div>
            <div class="kpi-value" style="color:{THEME['danger']}">{total_high_risk_customers:,}</div>
            <div class="kpi-subtitle">>70% churn probability</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Revenue at Risk</div>
            <div class="kpi-value" style="color:{THEME['warning']}">${revenue_at_risk:,.0f}</div>
            <div class="kpi-subtitle">From high-risk accounts</div>
        </div>
    """, unsafe_allow_html=True)

with ch_col2:
    if not churn_df.empty:
        risk_counts = churn_df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        fig_churn = px.pie(
            risk_counts, names='Risk', values='Count', hole=0.5,
            color='Risk', color_discrete_map=THEME['churn_colors']
        )
        fig_churn.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220, showlegend=False)
        st.plotly_chart(fig_churn, use_container_width=True)

with ch_col3:
    st.markdown("**Top 20 High-Risk Customers**")
    if not churn_df.empty and total_high_risk_customers > 0:
        top_churn = high_risk_churn_df.sort_values(by='churn_probability', ascending=False).head(20).copy()
        top_churn['churn_probability'] = top_churn['churn_probability'].apply(lambda x: f"{x:.1%}")
        top_churn['monetary'] = top_churn['monetary'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(
            top_churn[['customer_id', 'churn_probability', 'monetary', 'recency']],
            hide_index=True, use_container_width=True, height=220
        )
    else:
        st.info("No high-risk customers identified.")

st.markdown("---")

# ==========================================
# ALERT TYPE 2: INVENTORY ALERTS
# ==========================================
st.markdown("<div class='section-header'>🚨 Inventory & Stockout Alerts</div>", unsafe_allow_html=True)

inv_col1, inv_col2, inv_col3 = st.columns([1, 1, 2])
with inv_col1:
    st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:10px;">
            <div class="kpi-title">Critical Risk Products</div>
            <div class="kpi-value" style="color:{THEME['danger']}">{critical_product_count:,}</div>
            <div class="kpi-subtitle">High urgency stockouts</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Immediate Reorders</div>
            <div class="kpi-value" style="color:{THEME['warning']}">{immediate_reorder_count:,}</div>
            <div class="kpi-subtitle">Urgency Index > 0.80</div>
        </div>
    """, unsafe_allow_html=True)

with inv_col2:
    if not master_df.empty:
        risk_dist = master_df['risk_category'].value_counts().reset_index()
        risk_dist.columns = ['Risk Category', 'Count']
        fig_inv = px.bar(
            risk_dist, x='Risk Category', y='Count', color='Risk Category',
            color_discrete_map=THEME['risk_colors'],
            category_orders={"Risk Category": ["Critical", "High", "Medium", "Low"]}
        )
        fig_inv.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220, showlegend=False)
        st.plotly_chart(fig_inv, use_container_width=True)

with inv_col3:
    st.markdown("**Top 20 Critical Products**")
    if not reorder_df.empty and critical_product_count > 0:
        top_inv = critical_products_df.sort_values(by='reorder_urgency', ascending=False).head(20).copy()
        top_inv['stockout_probability'] = top_inv['stockout_probability'].apply(lambda x: f"{x:.1%}")
        top_inv['reorder_urgency'] = top_inv['reorder_urgency'].apply(lambda x: f"{x:.2f}")
        st.dataframe(
            top_inv[['product_id', 'abc_class', 'stockout_probability', 'reorder_urgency']],
            hide_index=True, use_container_width=True, height=220
        )
    else:
        st.info("No critical products identified.")

st.markdown("---")

# ==========================================
# ALERT TYPE 3: DEMAND ALERTS
# ==========================================
st.markdown("<div class='section-header'>🚨 Demand Forecasting Alerts</div>", unsafe_allow_html=True)

dem_col1, dem_col2 = st.columns([1, 2])
with dem_col1:
    st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:15px;">
            <div class="kpi-title">Next Surge Week</div>
            <div class="kpi-value" style="color:{THEME['primary']}">{next_surge_week}</div>
            <div class="kpi-subtitle">Expected Increase: <strong>{expected_increase:+.1f}%</strong></div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Forecast Trend Status</div>
            <div class="kpi-value">{forecast_trend}</div>
            <div class="kpi-subtitle">Based on 8-week trajectory</div>
        </div>
    """, unsafe_allow_html=True)

with dem_col2:
    if demand_data and 'future' in demand_data:
        f_df = pd.DataFrame(demand_data['future'])
        fig_dem = go.Figure()
        
        fig_dem.add_trace(go.Scatter(
            x=f_df['date'], y=f_df['forecast'], mode='lines+markers', name='Forecast',
            line=dict(color=THEME['secondary'], width=3), marker=dict(size=8)
        ))
        
        # Add alert markers for surges
        if demand_surges:
            surge_dates = [s['date'] for s in demand_surges]
            surge_vals = [s['forecast'] for s in demand_surges]
            fig_dem.add_trace(go.Scatter(
                x=surge_dates, y=surge_vals, mode='markers', name='Demand Surge',
                marker=dict(color=THEME['success'], size=14, symbol='star')
            ))
            
        # Add alert markers for drops
        if demand_drops:
            drop_dates = [d['date'] for d in demand_drops]
            drop_vals = [d['forecast'] for d in demand_drops]
            fig_dem.add_trace(go.Scatter(
                x=drop_dates, y=drop_vals, mode='markers', name='Demand Drop',
                marker=dict(color=THEME['danger'], size=12, symbol='triangle-down')
            ))
            
        fig_dem.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=260,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dem, use_container_width=True)
    else:
        st.info("Demand forecast data not available.")

