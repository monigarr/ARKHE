# Getting Started with ARKHE Framework

Welcome to the ARKHE Framework! This guide will help you get up and running quickly.

## What is ARKHE?

ARKHE (ARKHĒ) is an enterprise-grade Python framework for mathematical sequence research and machine learning experimentation. It provides tools for:

- Generating and analyzing mathematical sequences (Collatz, etc.)
- Training transformer models to understand sequence patterns
- Statistical analysis and pattern detection
- Interactive exploration via CLI and web interfaces

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install specific dependencies
pip install numpy scipy pandas matplotlib torch streamlit tqdm pyyaml
```

### Verify Installation

```python
python -c "from math_research import CollatzSequence; print('ARKHE installed successfully!')"
```

## Quick Start Examples

### 1. Generate a Collatz Sequence

```python
from math_research.sequences import CollatzSequence

# Create a sequence starting from 27
seq = CollatzSequence(start=27)
sequence = seq.generate()

print(f"Sequence length: {len(sequence)} steps")
print(f"Max value: {seq.get_max_value()}")
print(f"Sequence: {sequence}")
```

### 2. Analyze Sequence Statistics

```python
from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceStatistics

seq = CollatzSequence(start=27)
sequence = seq.generate()

stats = SequenceStatistics(sequence)
summary = stats.summary()

print("Sequence Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

### 3. Visualize Sequences

```python
from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceVisualizer
import matplotlib.pyplot as plt

seq = CollatzSequence(start=27)
sequence = seq.generate()

visualizer = SequenceVisualizer()
fig, ax = visualizer.plot_sequence(sequence, title="Collatz Sequence from 27")
plt.show()
```

### 4. Train a Transformer Model

See the [Training Guide](training_guide.md) for detailed instructions, or try the Jupyter notebook:

```bash
# Run the training notebook
jupyter notebook src/notebooks/03_transformer_training.ipynb
```

## Using the CLI

### Generate Sequences

```bash
# Generate a sequence starting from 27
python -m src.apps.cli generate --start 27

# Save to file with statistics
python -m src.apps.cli generate --start 27 --output sequence.txt --show-stats

# Generate as JSON
python -m src.apps.cli generate --start 27 --format json --output sequence.json
```

### Train Models

```bash
# Train with default settings
python -m src.apps.cli train --num-samples 10000 --epochs 10

# Train with configuration file
python -m src.apps.cli train --config configs/training/collatz_transformer.yaml
```

### Evaluate Models

```bash
# Evaluate a trained model
python -m src.apps.cli evaluate --checkpoint checkpoints/best_model.pt --test-size 1000
```

### Analyze Patterns

```bash
# Analyze range of sequences
python -m src.apps.cli analyze --start 1 --end 100 --output analysis.json
```

## Using the Streamlit Demo

```bash
# Launch interactive web interface
streamlit run src/apps/streamlit_demo/app.py
```

Then open your browser to `http://localhost:8501` to:
- Generate and visualize sequences interactively
- Train and monitor models
- Analyze patterns across multiple sequences

## Next Steps

1. **Explore Examples**: Check out the Jupyter notebooks in `src/notebooks/`
2. **Read Documentation**: See `docs/guides/` for detailed guides
3. **API Reference**: See `docs/api/` for complete API documentation
4. **Training Guide**: See `docs/guides/training_guide.md` for ML workflows

## Getting Help

- Check the [FAQ](faq.md)
- Review the [API Documentation](../api/)
- See example notebooks in `src/notebooks/`
- Run `python -m src.apps.cli --help` for CLI help

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to the project.

