"""
Quick Start Examples for ARKHE Framework

This file contains simple, ready-to-run examples demonstrating
various features of the ARKHE Framework.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Example 1: Basic Sequence Generation
print("=" * 60)
print("Example 1: Basic Collatz Sequence Generation")
print("=" * 60)

from math_research.sequences import CollatzSequence

seq = CollatzSequence(start=27)
sequence = seq.generate()

print(f"Sequence starting from 27:")
print(f"  Length: {len(sequence)} steps")
print(f"  Max value: {seq.get_max_value()}")
print(f"  First 10 values: {sequence[:10]}")
print(f"  Last 5 values: {sequence[-5:]}")


# Example 2: Sequence Statistics
print("\n" + "=" * 60)
print("Example 2: Statistical Analysis")
print("=" * 60)

from math_research.analysis import SequenceStatistics

stats = SequenceStatistics(sequence)
summary = stats.summary()

print("Sequence Statistics:")
for key, value in list(summary.items())[:5]:  # Show first 5
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")


# Example 3: Long Step Computation
print("\n" + "=" * 60)
print("Example 3: Long Step Optimization")
print("=" * 60)

test_values = [27, 127, 255]

for val in test_values:
    step_info = seq.compute_long_step(val)
    print(f"Long step for {val}:")
    print(f"  k={step_info['k']}, k'={step_info['k_prime']}")
    print(f"  Result: {step_info['result']}")


# Example 4: Encoding
print("\n" + "=" * 60)
print("Example 4: Multi-Base Encoding")
print("=" * 60)

from math_research.ml import MultiBaseEncoder

encoder = MultiBaseEncoder(base=24)

test_number = 12345
encoded = encoder.encode(test_number)
decoded = encoder.decode(encoded)

print(f"Number: {test_number}")
print(f"Encoded (base 24): {encoded}")
print(f"Decoded: {decoded}")
print(f"Match: {decoded == test_number}")


# Example 5: Creating a Dataset
print("\n" + "=" * 60)
print("Example 5: Creating Training Dataset")
print("=" * 60)

from math_research.ml import CollatzDataset

dataset = CollatzDataset(
    start_range=(1, 1000),
    num_samples=100,
    base=24,
    max_length=20,
    seed=42,
)

print(f"Dataset created: {len(dataset)} samples")
input_sample, target_sample = dataset[0]
print(f"Sample shapes: Input={input_sample.shape}, Target={target_sample.shape}")


# Example 6: Model Creation
print("\n" + "=" * 60)
print("Example 6: Creating Transformer Model")
print("=" * 60)

from math_research.ml import CollatzTransformer

model = CollatzTransformer(
    vocab_size=24,
    d_model=128,  # Small for quick example
    nhead=4,
    num_layers=2,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters")

# Test forward pass
import torch
test_input = torch.randint(0, 24, (2, 10))
output = model(test_input)
print(f"Test input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")


print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
print("\nNext steps:")
print("  - See docs/guides/usage_examples.md for more examples")
print("  - Check src/notebooks/ for interactive notebooks")
print("  - Try the CLI: python -m src.apps.cli --help")
print("  - Launch Streamlit: streamlit run src/apps/streamlit_demo/app.py")

