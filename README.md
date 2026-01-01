# ARKHĒ FRAMEWORK

**A research framework for discovering structure, invariants, and emergence in mathematical sequences and symbolic systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/monigarr/ARKHE.git)

## Overview

The **ARKHĒ FRAMEWORK** is a comprehensive, enterprise-level Python framework designed to support mathematical sequence research and machine learning experimentation. Inspired by Collatz conjecture research, the framework provides tools for:

- **Sequence Generation**: Extensible framework for generating and analyzing mathematical sequences
- **Machine Learning**: Transformer-based models for sequence prediction
- **Analysis Tools**: Statistical analysis, visualization, and pattern detection
- **Interactive Interfaces**: CLI, Streamlit web app, and Jupyter notebooks
- **Research Support**: Comprehensive documentation and example notebooks

## Features

### Core Capabilities

- 🔢 **Mathematical Sequences**: Base framework for sequence generation with Collatz implementation and long-step optimization
- 🤖 **ML Models**: Complete transformer architecture (`CollatzTransformer`) for sequence prediction
- 📊 **Analysis Tools**: Statistical analysis, visualization, and pattern detection
- 🔬 **Pattern Detection**: Algorithms for identifying patterns in sequences
- 📈 **Experiment Tracking**: Integration with wandb and MLflow
- 🔧 **Extensible**: Easy to add custom sequence types via registry system

### User Interfaces

- 💻 **CLI Application**: Full-featured command-line interface with 4 commands
  - `generate`: Generate Collatz sequences (TXT, JSON, CSV)
  - `train`: Train transformer models
  - `evaluate`: Evaluate trained models
  - `analyze`: Batch sequence analysis
  **CLI Example**
  
---

- 🌐 **Streamlit Dashboard**: Interactive web application with 5 pages
  - Sequence Explorer with real-time visualization
  - Model Inference interface
  - Statistical Analysis dashboard
  - Interactive charts and data export
  ![ARKHĒ Streamlit Screenshot](images/ARKHE_Screenshot_streamlit_dashboard.png)
  ![ARKHĒ CLI Screenshot](images/ARKHE_Screenshare_Streamlit_Dashboard.mp4)

---

- 📓 **Jupyter Notebooks**: 3 complete notebooks with working examples
  ![ARKHĒ CLI Screenshot](images/ARKHE_Screenshare_Streamlit_Dashboard.mp4)

---

### Enterprise Quality

- ✅ **Type Hints**: Full type annotation coverage
- ✅ **Documentation**: 12+ comprehensive guides (~80KB+)
- ✅ **Testing**: 40+ tests across 7 test files
- ✅ **Code Quality**: Linting, formatting, and quality checks
- ✅ **Maintainability**: Clean architecture and design patterns

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/monigarr/ARKHE.git
cd ARKHE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
python -c "import sys; sys.path.insert(0, 'src'); from math_research import CollatzSequence; print('✓ ARKHE installed successfully!')"
```

### Basic Usage

#### Python API

```python
from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceStatistics, SequenceVisualizer

# Generate a Collatz sequence
seq = CollatzSequence(start=27)
sequence = seq.generate()

# Analyze the sequence
stats = SequenceStatistics(sequence)
print(stats.summary())

# Visualize
visualizer = SequenceVisualizer()
fig, ax = visualizer.plot_sequence(sequence, title="Collatz Sequence Starting at 27")
```

#### Command Line Interface

```bash
# Generate a sequence
python -m src.apps.cli generate --start 27 --output sequence.txt --show-stats

# Train a model
python -m src.apps.cli train --num-samples 10000 --epochs 10

# Evaluate a model
python -m src.apps.cli evaluate --checkpoint checkpoints/best_model.pt --test-size 1000

# Analyze sequences
python -m src.apps.cli analyze --start 1 --end 100 --output analysis.json
```

#### Streamlit Web Application

```bash
# Launch interactive web interface
streamlit run src/apps/streamlit_demo/app.py

# OR use the launcher script
python run_streamlit.py
```

Then open your browser to `http://localhost:8501`

#### Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook src/notebooks/
```

Try the notebooks:
- `01_collatz_basics.ipynb` - Basic sequence operations
- `02_sequence_analysis.ipynb` - Complete statistical analysis (18 cells): multiple sequences, visualizations, pattern analysis, and comparisons
- `03_transformer_training.ipynb` - Complete training pipeline

## Project Structure

```
ARKHE/
├── src/
│   ├── math_research/          # Main package
│   │   ├── sequences/          # Sequence generation (base, collatz, registry)
│   │   ├── analysis/           # Analysis tools (statistics, visualization, patterns)
│   │   ├── ml/                 # Machine learning
│   │   │   ├── models/         # Transformer models
│   │   │   ├── encoding/       # Data encoding (multi-base)
│   │   │   ├── training/       # Training pipeline (trainer, data_loader, experiment_tracker)
│   │   │   └── evaluation/     # Metrics and error analysis
│   │   └── utils/              # Utilities (config, logging, validators)
│   ├── apps/
│   │   ├── cli/                # Command-line interface
│   │   │   ├── main.py         # CLI entry point
│   │   │   └── commands/       # Command implementations
│   │   └── streamlit_demo/     # Streamlit web application
│   │       └── app.py          # Main Streamlit app
│   └── notebooks/              # Jupyter notebooks
│       ├── 01_collatz_basics.ipynb
│       ├── 02_sequence_analysis.ipynb
│       └── 03_transformer_training.ipynb
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (40+ tests)
│   └── integration/            # Integration tests
├── docs/                       # Documentation
│   ├── guides/                 # User guides
│   ├── api/                    # API documentation
│   ├── architecture/           # Architecture docs
│   └── examples/               # Example scripts
├── configs/                    # Configuration files
├── scripts/                    # Utility scripts
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── models/                 # Saved models
├── checkpoints/                # Model checkpoints (created during training)
├── README.md                   # This file
├── CHANGELOG.md                # Version history
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project configuration
```

## Documentation

### Getting Started
- **[Getting Started Guide](docs/guides/getting_started.md)** - Installation and quick start
- **[Usage Examples](docs/guides/usage_examples.md)** - Comprehensive code examples
- **[Training Guide](docs/guides/training_guide.md)** - Complete ML training workflow
- **[FAQ](docs/guides/faq.md)** - Common questions and troubleshooting
- **[Streamlit Setup](docs/guides/streamlit_setup.md)** - Streamlit-specific help

### API Reference
- **[API Documentation Index](docs/api/README.md)** - Complete API reference

### Examples
- **[Quick Start Script](docs/examples/quick_start.py)** - Runnable example script

## Key Components

### Sequence Framework

```python
from math_research.sequences import BaseSequence, CollatzSequence

# Use built-in Collatz sequence
seq = CollatzSequence(start=27)
sequence = seq.generate()

# Long step optimization
long_step = seq.compute_long_step(27)
print(f"k={long_step['k']}, k'={long_step['k_prime']}, result={long_step['result']}")

# Generate with long steps
long_steps = seq.generate_with_long_steps()
```

### Machine Learning Pipeline

```python
from math_research.ml import (
    CollatzTransformer,
    MultiBaseEncoder,
    CollatzDataset,
    Trainer,
    compute_accuracy,
)

# Create dataset
dataset = CollatzDataset(start_range=(1, 10000), num_samples=10000, base=24)

# Create model
model = CollatzTransformer(vocab_size=24, d_model=512, nhead=8, num_layers=6)

# Train
trainer = Trainer(model, train_loader, val_loader)
history = trainer.train(num_epochs=20)

# Evaluate
accuracy = compute_accuracy(predictions, targets)
```

See [Training Guide](docs/guides/training_guide.md) for complete examples.

### Analysis Tools

```python
from math_research.analysis import SequenceStatistics, SequenceVisualizer

stats = SequenceStatistics(sequence)
summary = stats.summary()  # Comprehensive statistics

visualizer = SequenceVisualizer()
fig, ax = visualizer.plot_sequence(sequence, show_peaks=True)
fig, ax = visualizer.plot_log_sequence(sequence)
fig, ax = visualizer.plot_histogram(sequence)
```

## Requirements

### Core Dependencies
- Python 3.8 or higher
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Pandas >= 2.0.0

### ML Dependencies
- PyTorch >= 2.0.0
- tqdm >= 4.65.0

### Visualization
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0 (optional)

### Interfaces
- Streamlit >= 1.50.0 (for web app)
- Jupyter (for notebooks)

### Utilities
- PyYAML (for configuration)
- pytest (for testing)

See [requirements.txt](requirements.txt) for complete list.

## Hardware Recommendations

**Minimum:**
- CPU: Multi-core processor
- RAM: 8 GB
- Storage: 10 GB

**Recommended (for ML training):**
- CPU: High-performance multi-core (Intel i7/i9, AMD Ryzen 7/9)
- RAM: 16+ GB
- GPU: NVIDIA GPU with CUDA support (8+ GB VRAM recommended)

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/math_research

# Run linting
flake8 src/
```

### Test Suite

The project includes comprehensive test coverage:
- **40+ test functions** across 7 test files
- Unit tests for all major components
- Integration tests for complete pipelines
- Parametrized tests for multiple configurations

Run tests:
```bash
pytest tests/ -v
```

### Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the project.

## Usage Examples

### CLI Examples

```bash
# Generate sequence and save as JSON
python -m src.apps.cli generate --start 27 --format json --output sequence.json

# Train model with config file
python -m src.apps.cli train --config configs/training/collatz_transformer.yaml

# Evaluate with custom test range
python -m src.apps.cli evaluate --checkpoint checkpoints/best_model.pt --test-range 10000 20000

# Batch analysis with step size
python -m src.apps.cli analyze --start 1 --end 1000 --step 10 --output batch_analysis.json
```

### Python API Examples

```python
# Custom sequence class
from math_research.sequences.base import BaseSequence

class MySequence(BaseSequence):
    def step(self, n: int) -> int:
        return n * 2 + 1

seq = MySequence(start=5)
sequence = seq.generate(max_iterations=10)
```

See [Usage Examples Guide](docs/guides/usage_examples.md) for more.

## Research Inspiration

This framework is inspired by research on training transformers to predict Collatz sequences:

> "Transformers know more than they can tell: Learning the Collatz sequence"

Key insights:
- Transformers can learn complex arithmetic functions with proper encoding
- Base 24 and 32 encoding yield optimal performance
- Models learn specific patterns (k, k' values) rather than universal algorithms
- Error patterns are explainable, not random hallucinations

## Statistics

- **38 Python source files** in `src/`
- **7 test files** with **40+ test functions**
- **3 Jupyter notebooks** with complete examples
- **12+ documentation files** (~80KB+)
- **4 CLI commands** fully functional
- **5 Streamlit pages** interactive
- **Comprehensive test coverage** for all major components

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**MoniGarr**
- Email: monigarr@MoniGarr.com
- Website: MoniGarr.com

**Research Interests:**
- AI/ML Research and Development
- Extended Reality (XR) Applications
- 3D Graphics and Visualization
- Robotics and Autonomous Systems
- Computer Vision
- Navigation Systems
- Natural Language Processing (NLP)
- Low Resource Languages (spoken in English communities)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{arkhe_framework,
  title = {ARKHĒ FRAMEWORK: Mathematical Sequence Research and ML Framework},
  author = {MoniGarr},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/monigarr/ARKHE.git},
  note = {Framework for Collatz sequence research and transformer model training}
}
```

## Acknowledgments

- Inspired by Collatz conjecture research and transformer-based sequence learning
- Built with excellent open-source tools (PyTorch, NumPy, Streamlit, Jupyter, etc.)
- Design principles influenced by research on interpretable ML

## Roadmap

- [ ] Additional sequence types (Fibonacci, Prime sequences, etc.)
- [ ] Enhanced visualization tools (3D plots, animations)
- [ ] Advanced ML architectures (attention variants, hybrid models)
- [ ] Distributed training support
- [ ] Real-time training monitoring
- [ ] Model comparison and benchmarking tools
- [ ] Sequence animation and playback
- [ ] REST API for model serving

## Support

For questions, issues, or contributions:
- Check the [FAQ](docs/guides/faq.md) first
- Review [documentation](docs/guides/)
- Open an issue on GitHub
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.

---

**Note:** This is a research framework designed for mathematical exploration and ML experimentation. It prioritizes interpretability and research insights over production deployment optimization.
