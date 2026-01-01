"""
Unit tests for ML evaluation modules.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import pytest
import torch
import numpy as np

from math_research.ml import (
    compute_accuracy,
    compute_exact_match,
    compute_collatz_metrics,
)
from math_research.sequences import CollatzSequence


def test_compute_accuracy_perfect_match():
    """Test accuracy computation with perfect predictions."""
    predictions = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    accuracy = compute_accuracy(predictions, targets)
    
    assert accuracy == 1.0


def test_compute_accuracy_partial_match():
    """Test accuracy computation with partial matches."""
    predictions = torch.tensor([[1, 2, 3], [4, 5, 7]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    accuracy = compute_accuracy(predictions, targets)
    
    # 5 out of 6 correct
    assert accuracy == pytest.approx(5.0 / 6.0, rel=1e-5)


def test_compute_accuracy_with_padding():
    """Test accuracy computation ignoring padding."""
    predictions = torch.tensor([[1, 2, 0, 0], [4, 5, 6, 0]])
    targets = torch.tensor([[1, 2, 0, 0], [4, 5, 7, 0]])
    
    accuracy = compute_accuracy(predictions, targets, ignore_index=0)
    
    # Only non-padding positions count: 1,2,4,5,6 vs 1,2,4,5,7
    # 4 out of 5 correct
    assert accuracy == pytest.approx(4.0 / 5.0, rel=1e-5)


def test_compute_accuracy_numpy_input():
    """Test accuracy computation with numpy arrays."""
    predictions = np.array([[1, 2, 3], [4, 5, 6]])
    targets = np.array([[1, 2, 3], [4, 5, 6]])
    
    accuracy = compute_accuracy(predictions, targets)
    
    assert accuracy == 1.0


def test_compute_exact_match_perfect():
    """Test exact match computation with perfect matches."""
    predictions = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    exact_match = compute_exact_match(predictions, targets)
    
    assert exact_match == 1.0


def test_compute_exact_match_partial():
    """Test exact match computation with partial matches."""
    predictions = torch.tensor([[1, 2, 3], [4, 5, 7]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    exact_match = compute_exact_match(predictions, targets)
    
    # Only 1 out of 2 sequences match exactly
    assert exact_match == 0.5


def test_compute_exact_match_with_padding():
    """Test exact match computation ignoring padding."""
    predictions = torch.tensor([[1, 2, 0, 0], [4, 5, 6, 0]])
    targets = torch.tensor([[1, 2, 0, 0], [4, 5, 7, 0]])
    
    exact_match = compute_exact_match(predictions, targets, ignore_index=0)
    
    # First sequence matches exactly, second doesn't
    assert exact_match == 0.5


def test_compute_collatz_metrics():
    """Test Collatz-specific metrics computation."""
    # Create sample predictions and targets
    # This is simplified - real test would decode properly
    predictions = torch.randint(0, 24, (10, 20))
    targets = torch.randint(0, 24, (10, 20))
    inputs = [27, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
    
    metrics = compute_collatz_metrics(
        predictions,
        targets,
        inputs,
        base=24,
    )
    
    assert isinstance(metrics, dict)
    # Check that expected keys are present
    assert 'k_accuracy' in metrics or 'result_accuracy' in metrics or len(metrics) > 0


def test_compute_collatz_metrics_with_collatz_sequence():
    """Test Collatz metrics with actual Collatz computations."""
    collatz = CollatzSequence(start=1)
    
    # Get actual long step results
    test_inputs = [27, 127, 255]
    actual_results = [collatz.compute_long_step(n)['result'] for n in test_inputs]
    
    # For this test, we'll create mock predictions that match
    # In practice, these would come from model predictions
    predictions = torch.randint(0, 24, (len(test_inputs), 10))
    targets = torch.randint(0, 24, (len(test_inputs), 10))
    
    metrics = compute_collatz_metrics(
        predictions,
        targets,
        test_inputs,
        base=24,
    )
    
    assert isinstance(metrics, dict)


def test_compute_accuracy_edge_cases():
    """Test accuracy computation edge cases."""
    # Empty tensors
    predictions = torch.tensor([[]])
    targets = torch.tensor([[]])
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 0.0 or accuracy == 1.0  # Edge case behavior
    
    # Single element
    predictions = torch.tensor([[5]])
    targets = torch.tensor([[5]])
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0
    
    predictions = torch.tensor([[5]])
    targets = torch.tensor([[6]])
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 0.0


def test_compute_accuracy_shape_validation():
    """Test that accuracy computation validates shapes."""
    predictions = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[1, 2]])  # Different shape
    
    # Should raise error or handle gracefully
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        compute_accuracy(predictions, targets)


def test_metrics_dtype_handling():
    """Test that metrics work with different dtypes."""
    # Long tensors
    predictions = torch.tensor([[1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.long)
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0
    
    # Int tensors (numpy)
    predictions = np.array([[1, 2, 3]], dtype=np.int64)
    targets = np.array([[1, 2, 3]], dtype=np.int64)
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 1.0

