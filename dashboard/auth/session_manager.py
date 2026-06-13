"""
session_manager.py

Handles Streamlit session state for authentication.
Provides helper functions to check auth status, log in, and log out.
"""
import streamlit as st

def init_session_state():
    """Initializes authentication variables in session state."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

def login_user(username):
    """Sets session state variables for a logged-in user."""
    st.session_state["authenticated"] = True
    st.session_state["username"] = username

def logout_user():
    """Clears session state and logs the user out."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

def is_authenticated():
    """Returns True if the user is currently authenticated."""
    return st.session_state.get("authenticated", False)

def require_auth():
    """
    To be called at the top of protected pages.
    If the user is not authenticated, stops execution and prompts them to log in.
    """
    if not is_authenticated():
        st.warning("Please log in from the main page to access this dashboard.")
        st.stop()
