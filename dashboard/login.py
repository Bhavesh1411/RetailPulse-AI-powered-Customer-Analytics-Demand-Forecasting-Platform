import streamlit as st
from auth.auth_manager import verify_credentials
from auth.session_manager import login_user

def render_login_page():
    """
    Renders the login UI and handles authentication.
    """
    st.title("RetailPulse")
    st.subheader("Login to access the dashboard")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In", type="primary")

        if submit:
            if verify_credentials(username, password):
                login_user(username)
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

    st.markdown("*(Hint: use `admin` / `retailpulse123`)*")
