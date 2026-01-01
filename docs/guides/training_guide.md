# Machine Learning Training Guide

Complete guide for training transformer models on Collatz sequences.

## Overview

The ARKHE Framework provides a complete pipeline for training transformer models to predict Collatz sequence patterns. This guide covers everything from data preparation to model deployment.

## Table of Contents

1. [Understanding the Task](#understanding-the-task)
2. [Data Preparation](#data-preparation)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Best Practices](#best-practices)

## Understanding the Task

### What We're Predicting

The model learns to predict Collatz "long steps" - given an odd integer, predict the result after applying the optimized Collatz transformation:

- **Input**: An odd integer (e.g., 27)
- **Output**: The result after applying long step transformation
- **Encoding**: Integers are encoded in a specific base (typically 24) as digit sequences

### Long Step Optimization

Instead of predicting individual steps:
- n → n/2 (if even)
- n → 3n+1 (if odd)

We predict optimized long steps that combine multiple operations for efficiency.

## Data Preparation

### Creating a Dataset

```python
from math_research.ml import CollatzDataset
from torch.utils.data import DataLoader, random_split

# Create dataset
dataset = CollatzDataset(
    start_range=(1, 10000),      # Range of input values
    num_samples=50000,            # Number of samples
    base=24,                      # Encoding base
    max_length=50,                # Maximum sequence length
    seed=42,                      # For reproducibility
)

print(f"Dataset size: {len(dataset)}")
```

### Train/Validation Split

```python
# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # Set to 0 for Windows compatibility
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)
```

### Inspecting Data

```python
# Check a sample
input_sample, target_sample = dataset[0]
print(f"Input shape: {input_sample.shape}")
print(f"Target shape: {target_sample.shape}")
print(f"Input (first 10): {input_sample[:10]}")
print(f"Target (first 10): {target_sample[:10]}")
```

## Model Architecture

### Creating a Model

```python
from math_research.ml import CollatzTransformer

model = CollatzTransformer(
    vocab_size=24,      # Must match encoding base
    d_model=512,        # Model dimension
    nhead=8,            # Attention heads
    num_layers=6,       # Transformer layers
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Model Components

The `CollatzTransformer` includes:
- **Embedding Layer**: Converts token indices to embeddings
- **Positional Encoding**: Adds position information
- **Transformer Encoder**: Multi-head self-attention layers
- **Output Head**: Predicts next token probabilities

### Choosing Architecture

| Component | Recommended Values | Notes |
|-----------|-------------------|-------|
| `vocab_size` | 24 or 32 | Match encoding base |
| `d_model` | 256-512 | Larger = more capacity |
| `nhead` | 4-8 | Must divide `d_model` |
| `num_layers` | 4-6 | More layers = deeper model |

## Training Pipeline

### Basic Training

```python
from math_research.ml import Trainer
import torch
import torch.nn as nn

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    checkpoint_dir="./checkpoints",
    log_interval=100,
)

# Setup optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

# Train
history = trainer.train(
    num_epochs=20,
    optimizer=optimizer,
    criterion=criterion,
    save_every=5,        # Save checkpoint every 5 epochs
    save_best=True,      # Save best model based on validation loss
)
```

### Training with Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,          # Reduce LR by half
    patience=3,          # Wait 3 epochs without improvement
    verbose=True,
)

history = trainer.train(
    num_epochs=20,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    save_best=True,
)
```

### Monitoring Training

```python
import matplotlib.pyplot as plt

# Plot training history
epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.show()
```

## Evaluation

### Computing Metrics

```python
from math_research.ml import (
    compute_accuracy,
    compute_exact_match,
    compute_collatz_metrics,
)

model.eval()
all_predictions = []
all_targets = []
all_inputs = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1)
        
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_inputs.append(inputs.cpu())

predictions = torch.cat(all_predictions, dim=0)
targets = torch.cat(all_targets, dim=0)

# Compute metrics
accuracy = compute_accuracy(predictions, targets)
exact_match = compute_exact_match(predictions, targets)

print(f"Accuracy: {accuracy:.4f}")
print(f"Exact Match: {exact_match:.4f}")
```

### Collatz-Specific Metrics

```python
# Get input values (simplified - would decode properly in practice)
input_values = [27, 127, 255, 511, 1023]

collatz_metrics = compute_collatz_metrics(
    predictions[:len(input_values)],
    targets[:len(input_values)],
    input_values,
    base=24,
)

print("Collatz Metrics:")
for key, value in collatz_metrics.items():
    print(f"  {key}: {value}")
```

## Hyperparameter Tuning

### Key Hyperparameters

```python
# Configuration dictionary
config = {
    # Data hyperparameters
    "data_range": (1, 10000),
    "num_samples": 50000,
    "base": 24,                    # Try: 16, 24, 32
    "max_length": 50,
    
    # Model hyperparameters
    "vocab_size": 24,              # Must match base
    "d_model": 512,                # Try: 256, 512, 1024
    "nhead": 8,                    # Try: 4, 8, 16
    "num_layers": 6,               # Try: 4, 6, 8
    
    # Training hyperparameters
    "batch_size": 32,              # Try: 16, 32, 64
    "learning_rate": 0.0001,       # Try: 0.0001, 0.001, 0.00001
    "num_epochs": 20,
}
```

### Experiment Tracking

```python
from math_research.ml.training import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    backend="wandb",
    project_name="collatz-transformer",
    experiment_name=f"base{config['base']}_d{config['d_model']}",
)

# Log hyperparameters
tracker.log_params(config)

# Log metrics during training
# (Integrate into training loop)

tracker.finish()
```

## Best Practices

### 1. Reproducibility

Always set random seeds:

```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### 2. Regular Checkpointing

Save checkpoints frequently:

```python
trainer.train(
    num_epochs=100,
    save_every=10,      # Save every 10 epochs
    save_best=True,     # Always save best model
)
```

### 3. Validation Set

Always use a validation set to monitor overfitting:

```python
# Check for overfitting
if val_loss > train_loss * 1.2:
    print("Warning: Possible overfitting!")
```

### 4. Learning Rate

Start with a conservative learning rate and use scheduling:

```python
# Start small
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Use scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, patience=3)
```

### 5. Early Stopping

Implement early stopping to prevent overfitting:

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # ... training ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model size
2. **NaN Loss**: Check learning rate, may be too high
3. **No Learning**: Check data loading, learning rate may be too low
4. **Overfitting**: Add more data, use dropout, or simplify model

### Debugging

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
    else:
        print(f"{name}: No gradient!")

# Check data shapes
for inputs, targets in train_loader:
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    break
```

## Next Steps

- See [Usage Examples](usage_examples.md) for more code examples
- Check the training notebook: `src/notebooks/03_transformer_training.ipynb`
- Review [API Documentation](../api/) for detailed API reference

