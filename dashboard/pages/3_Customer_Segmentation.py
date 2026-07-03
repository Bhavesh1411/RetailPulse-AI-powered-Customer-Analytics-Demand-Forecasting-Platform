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
    
    /* Recommendations styling */
    .rec-box {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 4px solid var(--primary-accent);
        background-color: #F8FAFC;
    }
    .rec-box.vip {
        border-left-color: var(--primary-accent);
    }
    .rec-box.loyalist {
        border-left-color: var(--secondary-accent);
    }
    .rec-box.atrisk {
        border-left-color: var(--danger);
    }
    .rec-title {
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 8px;
    }
    .rec-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<header>
    <h1 style="color:#0F172A; margin-bottom: 8px;">Customer Segmentation</h1>
    <p style="color:#64748B; font-size:1.1rem; margin-top:0; margin-bottom:24px;">
        Behavioral clusters and RFM analysis of the RetailPulse customer base.
    </p>
</header>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_segmentation_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = [
        os.path.join(base_dir, "customer_segmentation", "datasets", "customer_segments_kmeans_finalone.csv"),
        r"customer_segmentation/datasets/customer_segments_kmeans_finalone.csv",
        r"../customer_segmentation/datasets/customer_segments_kmeans_finalone.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, skip_blank_lines=True)
                df.columns = df.columns.str.strip()
                df = df.dropna(subset=['customer_id', 'cluster'])
                return df
            except Exception as e:
                pass
    st.error("Failed to load customer segmentation dataset. Please ensure the CSV exists.")
    st.stop()

# Load the data
df = load_segmentation_data()

# Apply business-friendly names
cluster_mapping = {
    'Cluster 0': 'VIP Customers',
    'Cluster 1': 'At-Risk Customers',
    'Cluster 2': 'Potential Loyalists'
}
df['cluster_name'] = df['cluster'].map(cluster_mapping).fillna(df['cluster'])

# -----------------
# 1. KPI CARDS SECTION
# -----------------
st.markdown("<section><div class='section-header'>Customer Base Overview</div></section>", unsafe_allow_html=True)

total_customers = df['customer_id'].nunique()
num_segments = df['cluster_name'].nunique()
high_value_customers = df[df['high_value_customer_flag'] == 1]['customer_id'].nunique()
avg_value = df['monetary'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-subtitle">Active customer base</div>
        </article>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Customer Segments</div>
            <div class="kpi-value">{num_segments}</div>
            <div class="kpi-subtitle">Cleaned KMeans clusters</div>
        </article>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">High-Value Customers</div>
            <div class="kpi-value">{high_value_customers:,}</div>
            <div class="kpi-subtitle" style="color:#22C55E;">{high_value_customers/total_customers*100:.1f}% of total base</div>
        </article>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <article class="kpi-card">
            <div class="kpi-title">Avg. Customer Value</div>
            <div class="kpi-value">${avg_value:,.2f}</div>
            <div class="kpi-subtitle">Lifetime monetary spend</div>
        </article>
    """, unsafe_allow_html=True)

# -----------------
# 2. INTERACTIVE CHARTS
# -----------------
st.markdown("<section><div class='section-header'>Segment Distribution & Financial Contribution</div></section>", unsafe_allow_html=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # Cluster Distribution Donut Chart
    dist_df = df['cluster_name'].value_counts().reset_index()
    dist_df.columns = ['Segment', 'Count']
    
    fig_dist = px.pie(
        dist_df,
        names='Segment',
        values='Count',
        hole=0.4,
        color='Segment',
        color_discrete_map={
            'VIP Customers': '#2563EB',
            'Potential Loyalists': '#0EA5E9',
            'At-Risk Customers': '#EF4444'
        }
    )
    fig_dist.update_traces(
        textinfo='percent+label', 
        pull=[0.02, 0.02, 0.02],
        textfont_size=12
    )
    fig_dist.update_layout(
        showlegend=False,
        margin=dict(t=24, b=24, l=16, r=16),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A'),
        height=320
    )
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:1.1rem; color:#0F172A; margin-top:0; margin-bottom:16px;'>Customer Share by Segment</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_chart2:
    # Revenue Contribution Bar Chart
    rev_df = df.groupby('cluster_name')['monetary'].sum().reset_index()
    rev_df.columns = ['Segment', 'Revenue']
    rev_df = rev_df.sort_values(by='Revenue', ascending=False)
    
    fig_rev = px.bar(
        rev_df,
        x='Segment',
        y='Revenue',
        color='Segment',
        color_discrete_map={
            'VIP Customers': '#2563EB',
            'Potential Loyalists': '#0EA5E9',
            'At-Risk Customers': '#EF4444'
        },
        text=rev_df['Revenue'].apply(lambda x: f"${x/1_000_000:.2f}M")
    )
    fig_rev.update_traces(
        textposition='outside',
        textfont_size=11
    )
    fig_rev.update_layout(
        showlegend=False,
        margin=dict(t=24, b=24, l=16, r=16),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0F172A'),
        yaxis=dict(gridcolor='#F1F5F9', title='Total Spend ($)'),
        xaxis=dict(title=''),
        height=320
    )
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:1.1rem; color:#0F172A; margin-top:0; margin-bottom:16px;'>Total Revenue Contribution</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_rev, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# 2B. BUSINESS ANALYTICS VISUALIZATIONS
# -----------------
st.markdown("<section><div class='section-header'>Business Analytics Visualizations</div></section>", unsafe_allow_html=True)

col_ba1, col_ba2 = st.columns(2)

with col_ba1:
    if 'monetary' in df.columns and 'cluster_name' in df.columns:
        rev_dist_df = df.groupby('cluster_name')['monetary'].sum().reset_index()
        rev_dist_df.columns = ['Segment', 'Revenue']
        fig_rev_donut = px.pie(
            rev_dist_df,
            names='Segment',
            values='Revenue',
            hole=0.4,
            color='Segment',
            color_discrete_map={
                'VIP Customers': '#2563EB',
                'Potential Loyalists': '#0EA5E9',
                'At-Risk Customers': '#EF4444'
            },
            title='Revenue Contribution by Segment'
        )
        # Update text to show value nicely
        fig_rev_donut.update_traces(
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Contribution: %{percent}',
            textfont_size=11
        )
        fig_rev_donut.update_layout(
            showlegend=False,
            margin=dict(t=40, b=24, l=16, r=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0F172A'),
            height=320
        )
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_rev_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dashboard-card"><h3>Revenue Contribution</h3><p>Not Available</p></div>', unsafe_allow_html=True)

with col_ba2:
    if all(col in df.columns for col in ['recency', 'frequency', 'monetary']):
        fig_rfm = px.density_heatmap(
            df,
            x='recency',
            y='frequency',
            z='monetary',
            histfunc='avg',
            title='Customer RFM Heatmap (Avg Monetary)',
            color_continuous_scale='Blues'
        )
        fig_rfm.update_layout(
            margin=dict(t=40, b=24, l=16, r=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0F172A'),
            height=320,
            xaxis_title="Recency (Days)",
            yaxis_title="Frequency (Orders)",
            coloraxis_colorbar_title="Avg Value"
        )
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_rfm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dashboard-card"><h3>Customer RFM Heatmap</h3><p>Not Available</p></div>', unsafe_allow_html=True)


# -----------------
# 3. SEGMENT INSIGHTS TABLE
# -----------------
st.markdown("<section><div class='section-header'>Segment Behavioral Profiling</div></section>", unsafe_allow_html=True)

summary_df = df.groupby('cluster_name').agg(
    customer_count=('customer_id', 'count'),
    total_revenue=('monetary', 'sum'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_basket=('average_purchase_value', 'mean'),
    avg_tenure=('customer_tenure', 'mean')
).reset_index()

total_cust = summary_df['customer_count'].sum()
total_rev = summary_df['total_revenue'].sum()

summary_df['pct_customers'] = summary_df['customer_count'] / total_cust * 100
summary_df['pct_revenue'] = summary_df['total_revenue'] / total_rev * 100

# Interpretations map
interpretations = {
    'VIP Customers': 'Highly active, high-value loyal champions with lowest recency. Represents the core revenue engine.',
    'At-Risk Customers': 'Lapsed, low-frequency occasional buyers. High risk of churn, requiring urgent reactivation.',
    'Potential Loyalists': 'Predictable weekend value shoppers. Moderate spend, high growth potential via targeted weekend flash promotions.'
}
summary_df['business_interpretation'] = summary_df['cluster_name'].map(interpretations)

# Reorder
segment_order = ['VIP Customers', 'Potential Loyalists', 'At-Risk Customers']
summary_df['cluster_name'] = pd.Categorical(summary_df['cluster_name'], categories=segment_order, ordered=True)
summary_df = summary_df.sort_values('cluster_name').reset_index(drop=True)

display_df = pd.DataFrame({
    'Segment Name': summary_df['cluster_name'],
    'Customers': summary_df['customer_count'].map('{:,}'.format),
    '% Customers': summary_df['pct_customers'].map('{:.1f}%'.format),
    'Total Spend': summary_df['total_revenue'].map('${:,.2f}'.format),
    '% Revenue': summary_df['pct_revenue'].map('{:.1f}%'.format),
    'Avg Recency (Days)': summary_df['avg_recency'].map('{:.1f}'.format),
    'Avg Frequency (Orders)': summary_df['avg_frequency'].map('{:.2f}'.format),
    'Avg Basket Size': summary_df['avg_basket'].map('${:,.2f}'.format),
    'Avg Tenure (Days)': summary_df['avg_tenure'].map('{:.1f}'.format),
    'Business Interpretation': summary_df['business_interpretation']
})

st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# 4. BUSINESS RECOMMENDATIONS PANEL
# -----------------
st.markdown("<section><div class='section-header'>Strategic Recommendations</div></section>", unsafe_allow_html=True)

tab_vip, tab_loyalist, tab_atrisk = st.tabs([
    "🏆 VIP Customers Strategy", 
    "🌅 Potential Loyalists Strategy", 
    "💤 At-Risk Customers Strategy"
])

with tab_vip:
    st.markdown("""
        <section class="rec-box vip">
            <div class="rec-title">📣 Marketing & Loyalty Programs</div>
            <div class="rec-desc">
                Deploy an <strong>exclusive VIP loyalty program</strong> (private sales events, tiered point multipliers, early access). 
                Implement personalized email campaigns with curated product recommendations based on past purchase categories.
            </div>
        </section>
        <section class="rec-box vip">
            <div class="rec-title">🔒 Retention & Concierge Service</div>
            <div class="rec-desc">
                Establish a <strong>white-glove concierge experience</strong> for the top 100 spenders. 
                Configure churn early-warning triggers: if a VIP has not purchased in 45+ days, auto-trigger a personal check-in or exclusive reward.
            </div>
        </section>
        <section class="rec-box vip">
            <div class="rec-title">💹 Revenue Growth & Upselling</div>
            <div class="rec-desc">
                Promote subscription plans or volume bundles to leverage their high frequency. 
                Set a free-shipping threshold above $500 to expand Average Order Value (AOV) from the current $403.38 average.
            </div>
        </section>
    """, unsafe_allow_html=True)

with tab_loyalist:
    st.markdown("""
        <section class="rec-box loyalist">
            <div class="rec-title">📣 Weekend Flash Promotions</div>
            <div class="rec-desc">
                Launch a recurring <strong>Weekend Flash Sale</strong>. Deploy targeted push notifications and email newsletters on Thursday evenings (6 PM - 8 PM) to capture planning-phase shoppers.
            </div>
        </section>
        <section class="rec-box loyalist">
            <div class="rec-title">⭐ Tailored Loyalty Incentives</div>
            <div class="rec-desc">
                Implement double points for weekend purchases. Reinforce weekend shopping behavior while working to shift high-potential accounts toward weekday purchases through weekday-only promotions.
            </div>
        </section>
        <section class="rec-box loyalist">
            <div class="rec-title">💹 AOV Expansion via Add-ons</div>
            <div class="rec-desc">
                Recommend discovery kits and leisure-themed product bundles at checkout. 
                Implement an intermediate free-shipping threshold ($300) to push their $249.14 basket average higher.
            </div>
        </section>
    """, unsafe_allow_html=True)

with tab_atrisk:
    st.markdown("""
        <section class="rec-box atrisk">
            <div class="rec-title">📣 Structured Win-Back Sequences</div>
            <div class="rec-desc">
                Automate <strong>reactivation email cycles</strong> at 90, 120, and 150 days of inactivity. 
                Escalate offer incentives dynamically (e.g., 10% discount → 20% discount → free gift with purchase).
            </div>
        </section>
        <section class="rec-box atrisk">
            <div class="rec-title">🔒 Onboarding & Trial Intervention</div>
            <div class="rec-desc">
                Deploy an educational post-purchase email series immediately after a customer's first purchase. 
                Provide usage tips and a time-limited incentive to secure the vital second purchase within 30 days.
            </div>
        </section>
        <section class="rec-box atrisk">
            <div class="rec-title">💹 Acquisition Waste Remediation</div>
            <div class="rec-desc">
                Filter lapsed accounts by historical order size. Prioritize win-back efforts on lapsed accounts whose single order exceeded $300 to recover Customer Acquisition Cost (CAC) efficiently.
            </div>
        </section>
    """, unsafe_allow_html=True)
