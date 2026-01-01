# Contributing to ARKHĒ FRAMEWORK

Thank you for your interest in contributing to the ARKHĒ Framework! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Project Structure](#project-structure)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, considerate, and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Familiarity with Python, NumPy, and optionally PyTorch

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/math-research-framework.git
cd math-research-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, well-documented code
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run linting
flake8 src/
black --check src/
isort --check src/
mypy src/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "Add feature: description of what was added"
```

Good commit message format:
```
Type: Brief description

Longer explanation if needed
- Bullet point 1
- Bullet point 2
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

## Coding Standards

### Code Style

We use the following tools for code quality:

- **black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run before committing:
```bash
black src/
isort src/
```

### Code Headers

Every Python file must include a comprehensive header:

```python
"""
Module: [module_name]
Package: [package_path]

Description:
    [Detailed description of module purpose, functionality, and usage]

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Author Research Interests:
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages (spoken in English communities)

Usage:
    [Code examples and usage patterns]

Dependencies:
    [List of required dependencies]

Version: [semantic version]
Last Modified: [ISO date format]
License: MIT

Notes:
    [Additional notes, warnings, or considerations]
"""
```

### Type Hints

All functions must include type hints:

```python
def function_name(param1: int, param2: str) -> Dict[str, Any]:
    """Function description."""
    pass
```

### Docstrings

Use Google-style or NumPy-style docstrings:

```python
def example_function(value: int, name: str) -> str:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        value: Description of value
        name: Description of name
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When value is invalid
    """
    pass
```

### Naming Conventions

- **Classes**: PascalCase (`CollatzSequence`)
- **Functions/Methods**: snake_case (`compute_long_step`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- **Private**: Leading underscore (`_internal_method`)

## Testing

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Place tests in `tests/` directory matching source structure
- Use descriptive test names

Example:
```python
def test_collatz_sequence_generation():
    """Test Collatz sequence generation."""
    seq = CollatzSequence(start=27)
    sequence = seq.generate(max_iterations=10)
    assert len(sequence) > 0
    assert sequence[0] == 27
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sequences.py

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run with verbose output
pytest -v
```

## Documentation

### Inline Documentation

- All public functions/methods need docstrings
- Add comments for complex algorithms
- Document non-obvious code sections

### External Documentation

- Update README.md for user-facing changes
- Update API documentation for API changes
- Add examples to notebooks for new features
- Update architecture docs for structural changes

### Documentation Templates

See `docs/templates/` for documentation templates:
- Software Architecture
- Design Documents
- UI/UX Design
- Tech Stack
- Milestones
- Requirements
- Goals

## Submitting Changes

### Pull Request Process

1. **Update your branch** with latest from main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Ensure all tests pass** and code quality checks pass

3. **Create Pull Request**:
   - Clear title and description
   - Reference related issues
   - Include example usage if adding features
   - Add screenshots for UI changes

4. **PR Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated and passing
   - [ ] Documentation updated
   - [ ] Type hints added
   - [ ] No linting errors
   - [ ] Commit messages are clear

### PR Review

- Address reviewer feedback promptly
- Be open to suggestions
- Discuss significant changes before implementing
- Keep PRs focused and reasonably sized

## Project Structure

```
math-research-framework/
├── src/
│   └── math_research/
│       ├── sequences/      # Sequence generation
│       ├── analysis/       # Analysis tools
│       ├── ml/             # Machine learning
│       └── utils/          # Utilities
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance tests
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
└── scripts/                # Utility scripts
```

### Adding New Features

1. **New Sequence Type**:
   - Create class inheriting from `BaseSequence`
   - Implement `step()` method
   - Add tests
   - Register in registry (optional)
   - Document usage

2. **New Analysis Tool**:
   - Add to `analysis/` module
   - Follow existing patterns
   - Add tests and examples
   - Update documentation

3. **New ML Component**:
   - Add to `ml/` module
   - Follow existing architecture
   - Include training/evaluation code
   - Add to documentation

## Areas for Contribution

### High Priority

- Additional sequence types
- More analysis metrics
- Enhanced visualizations
- Performance optimizations
- Documentation improvements

### Medium Priority

- Example notebooks
- UI applications (Streamlit/Gradio)
- CLI tools
- Additional ML architectures
- Benchmarking tools

### Ideas Welcome

- New research directions
- Integration with other tools
- Community suggestions
- Bug fixes
- Documentation enhancements

## Questions?

- Open an issue for questions or discussions
- Check existing issues and PRs
- Review documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ARKHĒ Framework! 🎉

