#!/usr/bin/env python3
"""
Minimal Python wrapper to run Streamlit dashboard.
This avoids macOS security restrictions on script execution and cwd access.
"""
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CRITICAL: Patch os.getcwd() BEFORE importing streamlit
# Streamlit tries to call os.getcwd() during import, which macOS blocks
_original_getcwd = os.getcwd
def _patched_getcwd():
    """Return script directory instead of trying to get actual cwd"""
    return SCRIPT_DIR
os.getcwd = _patched_getcwd

# Also patch pathlib.Path.cwd() which Streamlit also uses
try:
    from pathlib import Path
    _original_path_cwd = Path.cwd
    @classmethod
    def _patched_path_cwd(cls):
        return cls(SCRIPT_DIR)
    Path.cwd = _patched_path_cwd
except:
    pass

# Change to the script directory
try:
    os.chdir(SCRIPT_DIR)
except:
    pass  # If chdir fails, the patch above will handle it

# Add the script directory to Python path
sys.path.insert(0, SCRIPT_DIR)

# Set environment variables before importing Streamlit
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Now import and run Streamlit
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Set up arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(SCRIPT_DIR, "app_dashboard.py"),
        "--server.headless=true",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    # Run Streamlit
    import os
    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        sys.exit(stcli.main())

