import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RetailPulse | Inventory Optimization",
    page_icon="📦",
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
    "risk_colors": {"Critical": "#EF4444", "High": "#F59E0B", "Medium": "#0EA5E9", "Low": "#22C55E"}
}

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(base_dir, "inventory_optimization", "outputs")
    
    master_path = os.path.join(output_dir, "inventory_master.csv")
    reorder_path = os.path.join(output_dir, "reorder_recommendations.csv")
    kpi_path = os.path.join(output_dir, "inventory_kpi_summary.csv")
    
    master_df = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()
    reorder_df = pd.read_csv(reorder_path) if os.path.exists(reorder_path) else pd.DataFrame()
    
    kpi_dict = {}
    if os.path.exists(kpi_path):
        kpi_df = pd.read_csv(kpi_path)
        for _, row in kpi_df.iterrows():
            kpi_dict[row['metric']] = row['value']
            
    return master_df, reorder_df, kpi_dict

try:
    master_df, reorder_df, kpi_dict = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

if master_df.empty or reorder_df.empty or not kpi_dict:
    st.warning("Inventory Optimization data not found. Please ensure the pipeline has been executed.")
    st.stop()

# Parse KPIs
health_score = float(kpi_dict.get('inventory_health_score', 0))
total_products = int(float(kpi_dict.get('total_products', 0)))
critical_risk = int(float(kpi_dict.get('risk_category_counts.Critical', 0)))
high_risk = int(float(kpi_dict.get('risk_category_counts.High', 0)))
avg_eoq = master_df['eoq'].mean() if 'eoq' in master_df.columns else 0
total_reorder_recommendations = len(reorder_df)

# ==========================================
# HEADER
# ==========================================
st.title("📦 Inventory Optimization")
st.markdown(f"<p style='color:{THEME['text']}; font-size:1.1rem; margin-top:-1rem;'>Executive Dashboard for Stock Replenishment & Risk Mitigation</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SECTION 1 & 5: KPI CARDS & INVENTORY HEALTH
# ==========================================
if health_score >= 80:
    health_status = "Healthy"
    health_color = THEME['success']
elif health_score >= 60:
    health_status = "Attention Required"
    health_color = THEME['warning']
else:
    health_status = "Critical Inventory Health"
    health_color = THEME['danger']

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05); border-left:5px solid {health_color};'>
            <p style='color:#64748B; margin:0; font-size:0.9rem; font-weight:600;'>Inventory Health</p>
            <h2 style='color:{THEME['text']}; margin:5px 0; font-size:1.8rem;'>{health_score:.1f}/100</h2>
            <p style='color:{health_color}; margin:0; font-weight:bold;'>{health_status}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
            <p style='color:#64748B; margin:0; font-size:0.9rem; font-weight:600;'>Total Products</p>
            <h2 style='color:{THEME['text']}; margin:5px 0; font-size:1.8rem;'>{total_products:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05); border-left:5px solid {THEME['danger']};'>
            <p style='color:#64748B; margin:0; font-size:0.9rem; font-weight:600;'>Critical / High Risk</p>
            <h2 style='color:{THEME['text']}; margin:5px 0; font-size:1.8rem;'>{critical_risk:,} / {high_risk:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
            <p style='color:#64748B; margin:0; font-size:0.9rem; font-weight:600;'>Avg. EOQ</p>
            <h2 style='color:{THEME['text']}; margin:5px 0; font-size:1.8rem;'>{avg_eoq:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05); border-left:5px solid {THEME['primary']};'>
            <p style='color:#64748B; margin:0; font-size:0.9rem; font-weight:600;'>Reorder Alerts</p>
            <h2 style='color:{THEME['text']}; margin:5px 0; font-size:1.8rem;'>{total_reorder_recommendations:,}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# WIDGET & CHARTS ROW
# ==========================================
col_w1, col_w2 = st.columns([1, 2])

with col_w1:
    st.markdown("### 🚨 Top 10 Critical Risk Products")
    if not reorder_df.empty:
        critical_prods = reorder_df[reorder_df['risk_category'] == 'Critical'].copy()
        if not critical_prods.empty:
            critical_prods = critical_prods.sort_values(by='reorder_urgency', ascending=False).head(10)
            widget_df = critical_prods[['product_id', 'abc_class', 'stockout_probability', 'reorder_urgency']].copy()
            widget_df['stockout_probability'] = widget_df['stockout_probability'].apply(lambda x: f"{x:.1%}")
            widget_df['reorder_urgency'] = widget_df['reorder_urgency'].apply(lambda x: f"{x:.2f}")
            widget_df.columns = ['Product ID', 'ABC Class', 'Stockout Prob.', 'Urgency']
            st.dataframe(widget_df, use_container_width=True, hide_index=True)
        else:
            st.info("No critical risk products found.")

with col_w2:
    st.markdown("### 📊 Stockout Risk Distribution")
    risk_counts = master_df['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Product Count']
    fig_risk = px.bar(
        risk_counts,
        x='Risk Category',
        y='Product Count',
        color='Risk Category',
        color_discrete_map=THEME['risk_colors'],
        category_orders={"Risk Category": ["Critical", "High", "Medium", "Low"]}
    )
    fig_risk.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 1B: STOCKOUT RISK MATRIX
# ==========================================
st.markdown("### 🎯 Stockout Risk Matrix")

if not master_df.empty and all(col in master_df.columns for col in ['simulated_current_stock', 'stockout_probability', 'abc_class', 'product_id', 'safety_stock', 'reorder_point']):
    plot_df = master_df.copy()
    plot_df['Below Reorder Point'] = plot_df['simulated_current_stock'] < plot_df['reorder_point']
    plot_df['Marker Size'] = plot_df['Below Reorder Point'].apply(lambda x: 12 if x else 6)
    
    fig_matrix = px.scatter(
        plot_df,
        x='simulated_current_stock',
        y='stockout_probability',
        color='abc_class',
        color_discrete_map=THEME['abc_colors'],
        symbol='Below Reorder Point',
        symbol_map={True: 'diamond', False: 'circle'},
        size='Marker Size',
        hover_data={
            'product_id': True,
            'simulated_current_stock': ':.1f',
            'safety_stock': ':.1f',
            'reorder_point': ':.1f',
            'stockout_probability': ':.1%',
            'Below Reorder Point': False,
            'Marker Size': False,
            'abc_class': True
        },
        labels={
            'simulated_current_stock': 'Current Stock',
            'stockout_probability': 'Stockout Probability',
            'abc_class': 'ABC Class'
        },
        title='Current Stock vs Stockout Probability'
    )
    
    fig_matrix.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor='#F1F5F9'),
        yaxis=dict(gridcolor='#F1F5F9')
    )
    st.markdown('<div style="background-color:#FFFFFF; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="background-color:#FFFFFF; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.05);"><h3>Stockout Risk Matrix</h3><p>Not Available</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 2: ABC ANALYSIS
# ==========================================
st.markdown("### 🔠 ABC Classification Analysis")

abc_col1, abc_col2 = st.columns(2)

with abc_col1:
    abc_counts = master_df['abc_class'].value_counts().reset_index()
    abc_counts.columns = ['ABC Class', 'Product Count']
    fig_abc_count = px.pie(
        abc_counts, 
        names='ABC Class', 
        values='Product Count',
        title="Distribution by Product Count",
        color='ABC Class',
        color_discrete_map=THEME['abc_colors'],
        hole=0.4
    )
    fig_abc_count.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_abc_count, use_container_width=True)

with abc_col2:
    abc_rev = master_df.groupby('abc_class')['total_revenue'].sum().reset_index()
    abc_rev.columns = ['ABC Class', 'Total Revenue']
    fig_abc_rev = px.pie(
        abc_rev, 
        names='ABC Class', 
        values='Total Revenue',
        title="Distribution by Revenue Contribution",
        color='ABC Class',
        color_discrete_map=THEME['abc_colors'],
        hole=0.4
    )
    fig_abc_rev.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_abc_rev, use_container_width=True)

st.markdown("---")

# ==========================================
# SECTION 4: REORDER RECOMMENDATIONS
# ==========================================
st.markdown("### 📋 Reorder Recommendations")
st.markdown(f"<p style='color:{THEME['text']};'>Detailed replenishment suggestions based on current simulated stock levels and demand forecasting.</p>", unsafe_allow_html=True)

if not reorder_df.empty:
    display_df = reorder_df[['product_id', 'abc_class', 'simulated_current_stock', 'reorder_point', 'safety_stock', 'eoq', 'reorder_urgency', 'stockout_probability', 'risk_category']].copy()
    
    # Format columns
    display_df['simulated_current_stock'] = display_df['simulated_current_stock'].round(1)
    display_df['reorder_point'] = display_df['reorder_point'].round(1)
    display_df['safety_stock'] = display_df['safety_stock'].round(1)
    display_df['eoq'] = display_df['eoq'].round(0)
    display_df['reorder_urgency'] = display_df['reorder_urgency'].round(2)
    display_df['stockout_probability'] = display_df['stockout_probability'].apply(lambda x: f"{x:.1%}")
    
    display_df.columns = [
        'Product ID', 'ABC Class', 'Current Stock (Sim)', 'Reorder Point (ROP)', 
        'Safety Stock', 'EOQ', 'Urgency Index', 'Stockout Prob.', 'Risk'
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
else:
    st.info("No reorder recommendations available.")

st.markdown("---")

# ==========================================
# SECTION 6 & 7: BUSINESS INSIGHTS & SUMMARY
# ==========================================
st.markdown("### 💡 Business Insights & Project Summary")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown(f"""
    <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.05);'>
        <h4 style='color:{THEME['text']}; margin-top:0;'>Executive Recommendations</h4>
        <ul style='color:#475569; line-height:1.6;'>
            <li><strong>Immediate Action Required:</strong> Focus on Class A products showing 'Critical' risk. Stockouts in this category disproportionately impact revenue.</li>
            <li><strong>Safety Stock Review:</strong> Class B and C products with high stockout probability should have their safety stock buffers reassessed to prevent lost sales.</li>
            <li><strong>Ordering Efficiency:</strong> Utilize the recommended EOQ (Economic Order Quantity) to balance ordering costs and holding costs, particularly for stable Class A items.</li>
            <li><strong>Supplier Alignment:</strong> Lead times should be strictly monitored for critical items to prevent unexpected pipeline delays.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    st.markdown(f"""
    <div style='background-color:{THEME['card']}; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.05);'>
        <h4 style='color:{THEME['text']}; margin-top:0;'>Methodology & Readiness</h4>
        <ul style='color:#475569; line-height:1.6;'>
            <li><strong>ABC Analysis:</strong> Products classified by revenue contribution (A=80%, B=15%, C=5%).</li>
            <li><strong>EOQ Calculation:</strong> Balanced ordering vs. holding costs based on unit cost and annual demand.</li>
            <li><strong>Risk Scoring:</strong> Matrix combining probability of stockout and ABC class impact.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
