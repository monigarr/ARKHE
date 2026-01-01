"""
Quick launcher for Streamlit demo.

This script sets up the path and launches the Streamlit app.

Usage:
    python src/apps/streamlit_demo/run.py
"""

import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    import subprocess
    import os
    
    # Get the app.py path
    app_path = Path(__file__).parent / "app.py"
    
    # Change to project root
    os.chdir(project_root)
    
    # Run streamlit
    subprocess.run(["streamlit", "run", str(app_path)])

