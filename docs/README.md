# ARKHE Framework Documentation

This directory contains the Sphinx-based API documentation for the ARKHE Framework.

## Building the Documentation

### Prerequisites

Install Sphinx and the theme:

```bash
pip install sphinx sphinx-rtd-theme
```

Or install from requirements-dev.txt:

```bash
pip install -r requirements-dev.txt
```

### Build Commands

**HTML documentation:**

```bash
# Using make (Linux/Mac)
cd docs
make html

# Using sphinx-build directly
sphinx-build -b html . _build/html

# Using make.bat (Windows)
cd docs
make.bat html
```

**Other formats:**

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Single HTML page
make singlehtml
```

### View the Documentation

After building, open `_build/html/index.html` in your browser.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `api/` - API reference documentation
- `guides/` - User guides (linked from main docs)

## Auto-generating API Docs

The documentation is automatically generated from docstrings using Sphinx's autodoc extension. To update:

1. Ensure all modules have proper docstrings
2. Run `make html` to rebuild
3. Commit the updated `_build/html/` directory (if hosting on GitHub Pages)

## Hosting

The documentation can be hosted on:

- **GitHub Pages**: Enable in repository settings, point to `docs/_build/html/`
- **Read the Docs**: Connect repository, auto-builds on commits
- **Local**: Serve `_build/html/` with any web server

## Continuous Integration

Add to CI/CD pipeline:

```yaml
- name: Build documentation
  run: |
    cd docs
    pip install sphinx sphinx-rtd-theme
    make html
```

---

**Last Updated:** 2025-01-09
