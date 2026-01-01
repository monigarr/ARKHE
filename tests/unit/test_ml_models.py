"""
Unit tests for ML model modules.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import pytest
import torch
from math_research.ml import CollatzTransformer


def test_collatz_transformer_creation():
    """Test CollatzTransformer model creation."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=128,
        nhead=4,
        num_layers=2,
    )
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_collatz_transformer_forward_pass():
    """Test model forward pass."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=128,
        nhead=4,
        num_layers=2,
    )
    
    batch_size = 4
    seq_len = 10
    input_tensor = torch.randint(0, 24, (batch_size, seq_len))
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, seq_len, 24)
    assert not torch.isnan(output).any()


def test_collatz_transformer_parameters():
    """Test model parameter count."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=128,
        nhead=4,
        num_layers=2,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params == total_params  # All params should be trainable
    assert trainable_params > 1000  # Reasonable minimum


def test_collatz_transformer_device():
    """Test model device handling."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    # Test CPU
    model_cpu = model.cpu()
    assert next(model_cpu.parameters()).device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        assert next(model_cuda.parameters()).device.type == 'cuda'


def test_collatz_transformer_gradient_flow():
    """Test that gradients flow through the model."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    input_tensor = torch.randint(0, 24, (2, 5))
    target = torch.randint(0, 24, (2, 5))
    
    output = model(input_tensor)
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, 24),
        target.view(-1),
        ignore_index=0
    )
    
    loss.backward()
    
    # Check that gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_collatz_transformer_different_configs():
    """Test model with different configurations."""
    configs = [
        {"vocab_size": 10, "d_model": 64, "nhead": 2, "num_layers": 1},
        {"vocab_size": 24, "d_model": 128, "nhead": 4, "num_layers": 2},
        {"vocab_size": 32, "d_model": 256, "nhead": 8, "num_layers": 3},
    ]
    
    for config in configs:
        model = CollatzTransformer(**config)
        input_tensor = torch.randint(0, config["vocab_size"], (2, 10))
        output = model(input_tensor)
        
        assert output.shape == (2, 10, config["vocab_size"])


def test_collatz_transformer_eval_mode():
    """Test model in evaluation mode."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    model.eval()
    assert not model.training
    
    with torch.no_grad():
        input_tensor = torch.randint(0, 24, (2, 5))
        output = model(input_tensor)
        assert output.requires_grad == False


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [5, 10, 20])
def test_collatz_transformer_variable_shapes(batch_size, seq_len):
    """Test model with variable batch sizes and sequence lengths."""
    model = CollatzTransformer(
        vocab_size=24,
        d_model=64,
        nhead=2,
        num_layers=1,
    )
    
    input_tensor = torch.randint(0, 24, (batch_size, seq_len))
    output = model(input_tensor)
    
    assert output.shape == (batch_size, seq_len, 24)

