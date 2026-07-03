import streamlit as st
from auth.session_manager import init_session_state, is_authenticated
from components.sidebar import render_sidebar
from login import render_login_page

st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cards to ensure white background per theme
st.markdown("""
    <style>
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
    }
    .metric-title {
        color: #0F172A;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #2563EB;
        font-size: 2em;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize authentication variables
init_session_state()

# Streamlit Multi-Page routing entry point

if not is_authenticated():
    render_sidebar()
    render_login_page()
else:
    st.switch_page("pages/1_Executive_Overview.py")
