"""
auth_manager.py

Handles credential verification. Designed with a pluggable interface so that
PostgreSQL can easily replace the hardcoded temporary local credentials
in the future without modifying the dashboard logic.
"""

def verify_credentials(username, password):
    """
    Verifies user credentials.
    Currently uses temporary local credentials for development.
    TODO: Integrate PostgreSQL authentication here.
    """
    # Temporary local credentials
    VALID_USERNAME = "admin"
    VALID_PASSWORD = "retailpulse123"

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return True
    
    return False
