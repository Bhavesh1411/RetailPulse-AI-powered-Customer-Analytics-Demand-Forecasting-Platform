import streamlit as st
from auth.session_manager import logout_user, is_authenticated

def render_sidebar():
    """
    Renders the reusable sidebar component containing branding,
    logged-in username, navigation hints, and logout button.
    """
    with st.sidebar:
        st.markdown("## RetailPulse")
        st.markdown("**AI-powered Customer Analytics & Demand Forecasting Platform**")
        st.divider()

        if is_authenticated():
            username = st.session_state.get("username", "User")
            st.markdown(f"👤 **Logged in as:** `{username}`")
            st.divider()
            
            st.markdown("### Navigation")
            st.markdown("- Use the menu above to switch between modules.")
            
            st.divider()
            if st.button("Log Out", use_container_width=True, type="secondary"):
                logout_user()
                st.rerun()
        else:
            st.markdown("Please log in to access the dashboard modules.")
