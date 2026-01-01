# Frequently Asked Questions (FAQ)

Common questions and answers about the ARKHE Framework.

## General Questions

### What is ARKHE?

ARKHE (ARKHĒ) is an enterprise-grade Python framework for mathematical sequence research and machine learning experimentation. It provides tools for exploring mathematical sequences (like Collatz), performing statistical analysis, and training transformer models.

### What is the Collatz Conjecture?

The Collatz conjecture states that starting from any positive integer and repeatedly applying:
- If n is even: n → n/2
- If n is odd: n → 3n+1

You will eventually reach 1. This has been verified for numbers up to 2^68 but remains unproven.

### Why use a transformer model for sequences?

Transformers excel at learning patterns in sequences. By encoding integers as digit sequences, the model can learn the mathematical relationships in Collatz transformations.

## Installation

### How do I install ARKHE?

```bash
pip install -r requirements.txt
```

Or install dependencies individually:
```bash
pip install numpy scipy pandas matplotlib torch streamlit tqdm pyyaml
```

### Do I need CUDA/GPU?

No, the framework works on CPU. GPU (CUDA) is optional but recommended for faster training.

### What Python version is required?

Python 3.8 or higher is required.

## Usage

### How do I generate a Collatz sequence?

**Using Python:**
```python
from math_research.sequences import CollatzSequence

seq = CollatzSequence(start=27)
sequence = seq.generate()
print(sequence)
```

**Using CLI:**
```bash
python -m src.apps.cli generate --start 27
```

### How do I train a model?

**Using CLI:**
```bash
python -m src.apps.cli train --num-samples 10000 --epochs 10
```

**Using Python:**
See `src/notebooks/03_transformer_training.ipynb` or the [Training Guide](training_guide.md).

### How do I use a trained model?

```python
import torch
from math_research.ml import CollatzTransformer

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt")
model = CollatzTransformer(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
# (See usage examples for full code)
```

**Using CLI:**
```bash
python -m src.apps.cli evaluate --checkpoint checkpoints/best_model.pt
```

## Configuration

### What encoding base should I use?

Bases 24 and 32 are recommended based on research. Base 24 is a good default:
- Good balance between sequence length and vocabulary size
- Efficient for transformer models

### How many epochs should I train?

Start with 10-20 epochs. Monitor validation loss and use early stopping if validation loss stops improving.

### What batch size should I use?

Start with 32. Increase if you have GPU memory available, decrease if you run out of memory.

### What learning rate should I use?

Start with 0.0001. Use a learning rate scheduler to adjust automatically during training.

## Troubleshooting

### "Out of memory" error during training

Solutions:
1. Reduce batch size (e.g., from 32 to 16)
2. Reduce model size (smaller `d_model` or `num_layers`)
3. Reduce `max_length` in dataset
4. Use gradient accumulation

### Model not learning (loss not decreasing)

Check:
1. Learning rate might be too high or too low
2. Check that data is loading correctly
3. Verify model architecture is correct
4. Check that gradients are flowing (not NaN)

### Import errors

Make sure:
1. You're running from the project root directory
2. All dependencies are installed: `pip install -r requirements.txt`
3. Python path includes `src/` directory

### CUDA/GPU not detected

1. Check CUDA installation: `torch.cuda.is_available()`
2. Verify PyTorch CUDA version matches your CUDA installation
3. Framework works fine on CPU, just slower

### Validation loss higher than training loss

This is normal early in training. If it persists:
1. Model might be underfitting (needs more capacity)
2. Training set might be harder than validation set
3. Consider regularization techniques

## Advanced Topics

### How do I create a custom sequence type?

```python
from math_research.sequences.base import BaseSequence

class MySequence(BaseSequence):
    def step(self, n: int) -> int:
        # Define your sequence rule
        return n * 2 + 1

seq = MySequence(start=5)
sequence = seq.generate(max_iterations=10)
```

### How do I use a different encoding base?

Change the `base` parameter when creating datasets:

```python
from math_research.ml import CollatzDataset

dataset = CollatzDataset(
    start_range=(1, 10000),
    num_samples=10000,
    base=32,  # Changed from 24 to 32
    max_length=50,
)
```

Make sure `vocab_size` in model matches the base!

### How do I extend to predict full sequences?

Currently the model predicts single steps. To predict full sequences:
1. Modify the model to generate autoregressively
2. Update the training loop to use teacher forcing
3. Adjust the loss function for sequence generation

This is an advanced topic - see transformer sequence generation literature.

### How do I use experiment tracking (wandb/mlflow)?

```python
from math_research.ml.training import ExperimentTracker

tracker = ExperimentTracker(
    backend="wandb",  # or "mlflow"
    project_name="my-project",
    experiment_name="experiment-1",
)

tracker.log_metric("loss", 0.5, step=1)
tracker.log_params({"lr": 0.0001})
tracker.finish()
```

## Performance

### How long does training take?

Depends on:
- Dataset size: 10K samples ≈ 5-10 min, 100K ≈ 1-2 hours (CPU)
- Model size: Larger models take longer
- Hardware: GPU is 10-100x faster than CPU

### How much memory does training require?

Rough estimates:
- Model: 50-500 MB depending on size
- Dataset: 100-1000 MB depending on size
- Training: 2-4x dataset size for gradients/optimizer states

### Can I speed up training?

1. Use GPU (CUDA)
2. Increase batch size (if memory allows)
3. Reduce sequence length
4. Use mixed precision training (advanced)

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### How do I report bugs?

Open an issue on the project repository with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (Python version, OS, etc.)

### How do I request features?

Open an issue with:
- Feature description
- Use case
- Proposed implementation (if you have ideas)

## Getting Help

### Where can I find more help?

1. Check the [Usage Examples](usage_examples.md)
2. Review the [Training Guide](training_guide.md)
3. See [API Documentation](../api/)
4. Explore Jupyter notebooks in `src/notebooks/`
5. Check CLI help: `python -m src.apps.cli --help`

### Is there a community/forum?

Currently documentation and GitHub issues are the main support channels.

---

Have a question not answered here? Feel free to open an issue or contribute to the documentation!

