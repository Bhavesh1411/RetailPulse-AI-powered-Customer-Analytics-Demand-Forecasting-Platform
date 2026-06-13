import streamlit as st
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

st.title("Executive Overview")
st.markdown("Top-level KPIs and business metrics.")

# Placeholder KPI Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">$0.00</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Forecast Accuracy</div>
            <div class="metric-value">0%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">0</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">High Risk Customers</div>
            <div class="metric-value">0</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Products</div>
            <div class="metric-value">0</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Inventory Health Score</div>
            <div class="metric-value">0 / 100</div>
        </div>
    """, unsafe_allow_html=True)

st.info("Visualizations and actual data will be connected in future phases.")
