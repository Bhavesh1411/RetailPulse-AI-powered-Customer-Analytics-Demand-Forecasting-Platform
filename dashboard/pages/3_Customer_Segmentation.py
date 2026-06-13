import streamlit as st
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

st.title("Customer Segmentation")
st.markdown("Behavioral clusters and RFM analysis.")

st.subheader("Cluster Distribution")
st.info("[Placeholder] Donut chart or bar chart showing the breakdown of customers across segments.")

st.subheader("Segment Insights")
st.info("[Placeholder] Tables and radar charts illustrating the unique characteristics of each cluster (e.g., Champions, At-Risk).")

st.subheader("Revenue by Segment")
st.info("[Placeholder] Tree map or stacked bar showing revenue contribution by segment.")
