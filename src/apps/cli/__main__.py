"""
CLI entry point for running as a module.

Usage:
    python -m src.apps.cli
"""

from src.apps.cli.main import main

if __name__ == "__main__":
    import sys
    sys.exit(main())

