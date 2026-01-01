"""
Setup script for development environment.

This script helps set up the development environment by creating
virtual environments, installing dependencies, and configuring tools.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: list[str], check: bool = True) -> int:
    """Run a shell command."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=check)
    return result.returncode


def main():
    """Main setup function."""
    print("Setting up ARKHĒ Framework development environment...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("\nCreating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment created.")
        print("\nPlease activate the virtual environment:")
        if os.name == "nt":  # Windows
            print("  venv\\Scripts\\activate")
        else:  # Unix/MacOS
            print("  source venv/bin/activate")
        print("\nThen run this script again to install dependencies.")
        sys.exit(0)
    
    # Determine pip executable
    if os.name == "nt":  # Windows
        pip = "venv/Scripts/pip.exe"
        python = "venv/Scripts/python.exe"
    else:  # Unix/MacOS
        pip = "venv/bin/pip"
        python = "venv/bin/python"
    
    # Upgrade pip
    print("\nUpgrading pip...")
    run_command([python, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("\nInstalling dependencies...")
    run_command([pip, "install", "-r", "requirements.txt"])
    
    # Install development dependencies
    print("\nInstalling development dependencies...")
    run_command([pip, "install", "-r", "requirements-dev.txt"])
    
    # Install package in development mode
    print("\nInstalling package in development mode...")
    run_command([pip, "install", "-e", "."])
    
    # Install pre-commit hooks
    print("\nInstalling pre-commit hooks...")
    run_command([python, "-m", "pre_commit", "install"])
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("  1. Activate the virtual environment (if not already)")
    print("  2. Run tests: pytest")
    print("  3. Run linting: flake8 src/")
    print("  4. Start developing!")


if __name__ == "__main__":
    main()

