# Running the Streamlit Demo

## Quick Start

### Option 1: Use the Launcher Script (Easiest)

From the **project root** directory:

```bash
python run_streamlit.py
```

This automatically handles all path setup.

### Option 2: Run Streamlit Directly

From the **project root** directory:

```bash
streamlit run src/apps/streamlit_demo/app.py
```

**Important**: You must be in the project root (`I:\CursorProjects\Research\Mathematics\ARKHE`), not in the `src/apps/streamlit_demo/` folder.

## Troubleshooting "ModuleNotFoundError: No module named 'math_research'"

### The Problem

This error occurs when Python can't find the `math_research` module. This happens if:
- You're running from the wrong directory
- The path setup in the app fails
- Streamlit changes the working directory

### Solutions

**Solution 1: Use the launcher script (Recommended)**
```bash
python run_streamlit.py
```

**Solution 2: Verify your current directory**
```bash
# Windows (PowerShell)
Get-Location
# Should show: I:\CursorProjects\Research\Mathematics\ARKHE

# If not, navigate there:
cd I:\CursorProjects\Research\Mathematics\ARKHE
```

**Solution 3: Test imports first**
```bash
python test_streamlit_import.py
```

If this works, then Streamlit should work too.

**Solution 4: Set PYTHONPATH explicitly**
```bash
# Windows PowerShell
$env:PYTHONPATH="I:\CursorProjects\Research\Mathematics\ARKHE\src"
streamlit run src/apps/streamlit_demo/app.py

# Windows Command Prompt
set PYTHONPATH=I:\CursorProjects\Research\Mathematics\ARKHE\src
streamlit run src/apps/streamlit_demo/app.py
```

## What Changed

The `app.py` file has been updated with:
- Better path resolution using absolute paths
- Fallback path searching
- Helpful error messages if imports fail

## Still Having Issues?

1. **Test imports**: Run `python test_streamlit_import.py`
2. **Check directory**: Make sure you're in the project root
3. **Verify structure**: Ensure `src/math_research/` exists
4. **Check Streamlit**: Verify Streamlit is installed: `streamlit --version`

## Accessing the App

Once Streamlit starts successfully:
- Browser will open automatically
- URL: `http://localhost:8501`
- If browser doesn't open, copy the URL from the terminal

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.

