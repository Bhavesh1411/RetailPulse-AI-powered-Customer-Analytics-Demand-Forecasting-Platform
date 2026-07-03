"""
auth_manager.py

Handles credential verification. Designed with a pluggable interface so that
PostgreSQL can easily replace the hardcoded temporary local credentials
in the future without modifying the dashboard logic.
"""

import os
import getpass
import streamlit as st

def verify_credentials(username, password):
    """
    Verifies user credentials.
    Reads from Streamlit secrets or Environment Variables to avoid hardcoded credentials.
    TODO: Integrate PostgreSQL authentication here.
    """
    valid_username = None
    valid_password = None

    # 1. Check Streamlit secrets
    try:
        if "USERNAME" in st.secrets:
            valid_username = st.secrets["USERNAME"]
        if "PASSWORD" in st.secrets:
            valid_password = st.secrets["PASSWORD"]
    except Exception:
        pass

    # 2. Check environment variables
    # We only consider environment variables if PASSWORD is explicitly set in env.
    # This prevents the OS-default USERNAME (like 'LENOVO' on Windows) from triggering
    # when no custom credentials are provided.
    if not valid_password:
        if "PASSWORD" in os.environ:
            valid_password = os.environ["PASSWORD"]
            valid_username = os.environ.get("USERNAME", "admin")

    # 3. Fallback to default credentials
    if not valid_username:
        valid_username = "admin"
    if not valid_password:
        valid_password = "retailpulse123"


    if username == valid_username and password == valid_password:
        return True
    
    return False
