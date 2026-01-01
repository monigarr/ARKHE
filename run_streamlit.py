"""
Launcher script for Streamlit demo.

This script ensures proper path setup regardless of where it's run from.

Usage:
    python run_streamlit.py
    OR
    streamlit run run_streamlit.py
"""

import sys
from pathlib import Path

# Get project root (where this file is located)
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"

# Add src directory to path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Verify math_research can be imported
try:
    from math_research.sequences import CollatzSequence
    print("Path setup successful!")
except ImportError as e:
    print(f"Error: Could not import math_research")
    print(f"  Project root: {project_root}")
    print(f"  Src directory: {src_dir}")
    print(f"  Src exists: {src_dir.exists()}")
    print(f"  Math research exists: {(src_dir / 'math_research').exists()}")
    print(f"  Error: {e}")
    sys.exit(1)

# Now import streamlit and run the app
import streamlit.web.cli as stcli

# Run the streamlit app
if __name__ == "__main__":
    app_path = src_dir / "apps" / "streamlit_demo" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    # Set up sys.argv for streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
    ]
    
    stcli.main()

