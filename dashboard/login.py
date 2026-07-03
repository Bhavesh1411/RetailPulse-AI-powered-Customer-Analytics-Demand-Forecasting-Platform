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

        st.write(f"**Debug - Form Username Entered:** `{username}`")
        st.write(f"**Debug - Form Password Entered:** `{password}`")
        
        if submit:
            st.write("**Debug - Log In Button Clicked**")
            st.write(f"**Debug - Authenticated Before Login call:** `{st.session_state.get('authenticated')}`")
            res = verify_credentials(username, password)
            st.write(f"**Debug - verify_credentials result:** `{res}`")
            if res:
                login_user(username)
                st.write(f"**Debug - Authenticated After Login call:** `{st.session_state.get('authenticated')}`")
                st.write(f"**Debug - Username in session_state:** `{st.session_state.get('username')}`")
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

    st.markdown("*(Hint: use `admin` / `retailpulse123`)*")
