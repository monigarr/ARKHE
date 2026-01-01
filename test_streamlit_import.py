"""
Test script to verify Streamlit app imports work correctly.

Run this from the project root to test if imports are working:
    python test_streamlit_import.py
"""

import sys
from pathlib import Path

# Add src to path (same as what app.py does)
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print(f"Project root: {project_root}")
print(f"Src directory: {src_dir}")
print(f"Src exists: {src_dir.exists()}")
print(f"Math research exists: {(src_dir / 'math_research').exists()}")
print(f"\nTesting imports...")

try:
    from math_research.sequences import CollatzSequence
    print("✓ math_research.sequences imported")
    
    from math_research.analysis import SequenceStatistics, SequenceVisualizer
    print("✓ math_research.analysis imported")
    
    from math_research.ml import MultiBaseEncoder
    print("✓ math_research.ml imported")
    
    print("\n✅ All imports successful! Streamlit app should work.")
    print("\nTo run Streamlit:")
    print("  streamlit run src/apps/streamlit_demo/app.py")
    print("  OR")
    print("  python run_streamlit.py")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print(f"\nDebug info:")
    print(f"  Sys.path entries: {[p for p in sys.path if 'src' in str(p) or 'ARKHE' in str(p)]}")
    sys.exit(1)

