# Usage Examples

This guide provides comprehensive examples for using the ARKHE Framework.

## Table of Contents

1. [Sequence Generation](#sequence-generation)
2. [Statistical Analysis](#statistical-analysis)
3. [Machine Learning](#machine-learning)
4. [Visualization](#visualization)
5. [Advanced Examples](#advanced-examples)

## Sequence Generation

### Basic Sequence Generation

```python
from math_research.sequences import CollatzSequence

# Generate a standard Collatz sequence
seq = CollatzSequence(start=27)
sequence = seq.generate()
print(f"Sequence: {sequence}")
print(f"Length: {len(sequence)} steps")
```

### Long Step Optimization

```python
# Use long step optimization for efficiency
seq = CollatzSequence(start=27, use_long_step=True)

# Generate using long steps
long_steps = seq.generate_with_long_steps()
print(f"Number of long steps: {len(long_steps)}")

# Compute long step for a specific value
step_info = seq.compute_long_step(27)
print(f"k={step_info['k']}, k'={step_info['k_prime']}, result={step_info['result']}")
```

### Batch Sequence Generation

```python
# Generate sequences for multiple starting values
results = []
for start in [27, 127, 255, 511, 1023]:
    seq = CollatzSequence(start=start)
    sequence = seq.generate()
    results.append({
        'start': start,
        'length': len(sequence),
        'max_value': seq.get_max_value(),
    })

for r in results:
    print(f"Start: {r['start']}, Length: {r['length']}, Max: {r['max_value']}")
```

## Statistical Analysis

### Basic Statistics

```python
from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceStatistics

seq = CollatzSequence(start=27)
sequence = seq.generate()

stats = SequenceStatistics(sequence)

# Individual statistics
print(f"Length: {stats.length()}")
print(f"Max value: {stats.max_value()}")
print(f"Min value: {stats.min_value()}")
print(f"Mean: {stats.mean():.2f}")
print(f"Std deviation: {stats.std():.2f}")

# Get complete summary
summary = stats.summary()
print("\nComplete Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```

### Pattern Detection

```python
from math_research.analysis import PatternDetector

seq = CollatzSequence(start=27)
sequence = seq.generate()

detector = PatternDetector()
patterns = detector.detect(sequence)

print("Detected patterns:")
for pattern_type, details in patterns.items():
    print(f"  {pattern_type}: {details}")
```

### Comparative Analysis

```python
# Compare multiple sequences
sequences_data = []

for start in [27, 127, 255]:
    seq = CollatzSequence(start=start)
    sequence = seq.generate()
    stats = SequenceStatistics(sequence)
    
    sequences_data.append({
        'start': start,
        'sequence': sequence,
        'stats': stats.summary(),
    })

# Compare statistics
for data in sequences_data:
    print(f"\nStart value: {data['start']}")
    print(f"  Length: {data['stats']['length']}")
    print(f"  Max value: {data['stats']['max_value']}")
    print(f"  Mean: {data['stats']['mean']:.2f}")
```

## Machine Learning

### Creating a Dataset

```python
from math_research.ml import CollatzDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = CollatzDataset(
    start_range=(1, 10000),
    num_samples=10000,
    base=24,
    max_length=50,
    seed=42,
)

# Create data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Inspect a batch
for inputs, targets in loader:
    print(f"Batch shape - Inputs: {inputs.shape}, Targets: {targets.shape}")
    break
```

### Model Training

```python
from math_research.ml import CollatzTransformer, Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Create dataset
dataset = CollatzDataset(
    start_range=(1, 10000),
    num_samples=10000,
    base=24,
    max_length=50,
)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = CollatzTransformer(
    vocab_size=24,
    d_model=512,
    nhead=8,
    num_layers=6,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_dir="./checkpoints",
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

history = trainer.train(
    num_epochs=10,
    optimizer=optimizer,
    criterion=criterion,
    save_best=True,
)

print("Training complete!")
print(f"Best validation loss: {trainer.best_val_loss:.4f}")
```

### Model Evaluation

```python
from math_research.ml import compute_accuracy, compute_exact_match, compute_collatz_metrics
import torch

# Evaluate on validation set
model.eval()
all_predictions = []
all_targets = []
all_inputs = []

with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1)
        all_predictions.append(predictions)
        all_targets.append(targets)
        all_inputs.append(inputs)

predictions = torch.cat(all_predictions, dim=0)
targets = torch.cat(all_targets, dim=0)
inputs = torch.cat(all_inputs, dim=0)

# Compute metrics
accuracy = compute_accuracy(predictions, targets)
exact_match = compute_exact_match(predictions, targets)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Exact Match: {exact_match:.4f} ({exact_match*100:.2f}%)")

# Collatz-specific metrics
input_values = [27, 127, 255]  # Example inputs
collatz_metrics = compute_collatz_metrics(
    predictions[:len(input_values)],
    targets[:len(input_values)],
    input_values,
    base=24,
)

print("\nCollatz Metrics:")
for key, value in collatz_metrics.items():
    print(f"  {key}: {value}")
```

### Loading and Using Trained Models

```python
import torch
from math_research.ml import CollatzTransformer, MultiBaseEncoder

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")

# Recreate model
model = CollatzTransformer(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
encoder = MultiBaseEncoder(base=24, max_length=50)
test_value = 27

encoded_input = encoder.encode(test_value)
input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0)

# Pad to max_length
if len(input_tensor[0]) < 50:
    padding = torch.zeros(50 - len(input_tensor[0]), dtype=torch.long)
    input_tensor = torch.cat([input_tensor, padding.unsqueeze(0)], dim=1)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=-1)[0]

print(f"Input: {test_value}")
print(f"Prediction: {prediction.numpy()}")
```

## Visualization

### Basic Plotting

```python
from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceVisualizer
import matplotlib.pyplot as plt

seq = CollatzSequence(start=27)
sequence = seq.generate()

visualizer = SequenceVisualizer()

# Line plot
fig, ax = visualizer.plot_sequence(
    sequence,
    title="Collatz Sequence from 27",
    show_peaks=True,
)
plt.show()

# Log scale plot
fig, ax = visualizer.plot_log_sequence(
    sequence,
    title="Collatz Sequence (Log Scale)",
)
plt.show()

# Histogram
fig, ax = visualizer.plot_histogram(
    sequence,
    bins=50,
    title="Value Distribution",
)
plt.show()
```

### Comparison Visualization

```python
# Compare multiple sequences
sequences_to_compare = []

for start in [27, 127, 255]:
    seq = CollatzSequence(start=start)
    sequence = seq.generate()
    sequences_to_compare.append((sequence, f"Start: {start}"))

visualizer = SequenceVisualizer()
fig, ax = visualizer.plot_comparison(
    sequences_to_compare,
    title="Sequence Comparison",
)
plt.show()
```

## Advanced Examples

### Custom Sequence Class

```python
from math_research.sequences.base import BaseSequence

class FibonacciSequence(BaseSequence):
    """Custom Fibonacci sequence implementation."""
    
    def step(self, n: int) -> int:
        if len(self.history) < 2:
            return n
        return self.history[-1] + self.history[-2]

# Use custom sequence
fib = FibonacciSequence(start=1)
fib_sequence = fib.generate(max_iterations=10)
print(f"Fibonacci: {fib_sequence}")
```

### Batch Processing with CLI

```python
import subprocess
import json

# Generate sequences for a range
results = []
for start in range(1, 101):
    result = subprocess.run(
        ["python", "-m", "src.apps.cli", "generate", "--start", str(start), "--format", "json"],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    results.append(data)

# Analyze results
lengths = [r['length'] for r in results]
print(f"Average length: {sum(lengths) / len(lengths):.2f}")
print(f"Max length: {max(lengths)}")
```

### Experiment Tracking

```python
from math_research.ml.training import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    backend="wandb",  # or "mlflow", "none"
    project_name="collatz-research",
    experiment_name="transformer-base24",
)

# Track training
for epoch in range(10):
    train_loss = 0.5  # Your actual loss
    val_loss = 0.6     # Your actual validation loss
    
    tracker.log_metric("train_loss", train_loss, step=epoch)
    tracker.log_metric("val_loss", val_loss, step=epoch)

# Log hyperparameters
tracker.log_params({
    "learning_rate": 0.0001,
    "batch_size": 32,
    "d_model": 512,
})

tracker.finish()
```

### Configuration Management

```python
from math_research.utils import load_config, Config

# Load configuration file
config = load_config("configs/default.yaml")

# Access nested values
learning_rate = config.get("training", {}).get("learning_rate", 0.0001)

# Or use Config class for dot notation
cfg = Config("configs/default.yaml")
learning_rate = cfg.training.learning_rate  # If nested structure supports it
```

## Next Steps

- See [Training Guide](training_guide.md) for detailed ML workflows
- Check [API Documentation](../api/) for complete API reference
- Explore Jupyter notebooks in `src/notebooks/` for interactive examples

