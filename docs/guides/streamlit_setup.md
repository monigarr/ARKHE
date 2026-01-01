# Streamlit Demo Setup Guide

Quick guide for running the Streamlit demo application.

## Quick Start

### Method 1: Using the Launcher Script (Recommended)

From the project root directory:

```bash
python run_streamlit.py
```

This script automatically handles path setup.

### Method 2: Using Streamlit Directly

From the project root directory:

```bash
streamlit run src/apps/streamlit_demo/app.py
```

**Important**: Make sure you're in the project root (`I:\CursorProjects\Research\Mathematics\ARKHE`)

## Troubleshooting

### ModuleNotFoundError: No module named 'math_research'

This error occurs when Python can't find the `math_research` module. Solutions:

**Solution 1: Use the launcher script**
```bash
python run_streamlit.py
```

**Solution 2: Run from project root**
```bash
# Make sure you're in the project root
cd I:\CursorProjects\Research\Mathematics\ARKHE

# Then run
streamlit run src/apps/streamlit_demo/app.py
```

**Solution 3: Add src to PYTHONPATH**
```bash
# Windows (PowerShell)
$env:PYTHONPATH="I:\CursorProjects\Research\Mathematics\ARKHE\src"
streamlit run src/apps/streamlit_demo/app.py

# Windows (Command Prompt)
set PYTHONPATH=I:\CursorProjects\Research\Mathematics\ARKHE\src
streamlit run src/apps/streamlit_demo/app.py

# Linux/Mac
export PYTHONPATH="$PWD/src"
streamlit run src/apps/streamlit_demo/app.py
```

**Solution 4: Install in development mode**
```bash
pip install -e .
```

(Requires setup.py or pyproject.toml configured for editable install)

### Verify Installation

Check that the module can be imported:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from math_research.sequences import CollatzSequence; print('✓ Import successful!')"
```

### Verify Working Directory

Make sure you're in the correct directory:

```bash
# Windows (PowerShell)
pwd
# Should show: I:\CursorProjects\Research\Mathematics\ARKHE

# Check if src directory exists
Test-Path src/math_research
# Should return: True
```

## What the App Does

The Streamlit demo provides:

1. **Sequence Explorer**: Generate and visualize Collatz sequences
2. **Model Inference**: Use trained models to make predictions
3. **Statistical Analysis**: Analyze patterns across sequences
4. **Home**: Overview and quick navigation

## Browser Access

Once Streamlit starts, it will:
1. Open automatically in your default browser
2. Show the URL (usually `http://localhost:8501`)
3. Display any errors in the terminal and browser

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.

## Next Steps

- Check the [Getting Started Guide](getting_started.md) for more information
- See [Usage Examples](usage_examples.md) for code examples
- Review the [FAQ](faq.md) for common issues

