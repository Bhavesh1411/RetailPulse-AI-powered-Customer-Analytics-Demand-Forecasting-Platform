import streamlit as st
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

st.title("Churn Prediction")
st.markdown("Identify customers at high risk of attrition.")

st.subheader("Churn Risk Distribution")
st.info("[Placeholder] Histogram or gauge charts showing the probability scores across the customer base.")

st.subheader("High Risk Customers")
st.info("[Placeholder] Data table listing customers with probability > 80%, sortable by customer lifetime value.")

st.subheader("Model Metrics")
st.info("[Placeholder] Accuracy, Recall, and ROC-AUC scores of the underlying predictive model.")
