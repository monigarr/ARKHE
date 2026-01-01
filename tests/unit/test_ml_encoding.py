"""
Unit tests for ML encoding modules.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com
"""

import pytest
from math_research.ml.encoding import MultiBaseEncoder


def test_multi_base_encoder():
    """Test multi-base encoding."""
    encoder = MultiBaseEncoder(base=24)
    
    test_value = 12345
    encoded = encoder.encode(test_value)
    decoded = encoder.decode(encoded)
    
    assert decoded == test_value


def test_multi_base_encoder_batch():
    """Test batch encoding/decoding."""
    encoder = MultiBaseEncoder(base=10)
    
    values = [123, 456, 789]
    encoded_batch = encoder.encode_batch(values)
    decoded_batch = encoder.decode_batch(encoded_batch)
    
    assert decoded_batch == values


def test_encoder_different_bases():
    """Test encoding with different bases."""
    for base in [2, 10, 16, 24, 32]:
        encoder = MultiBaseEncoder(base=base)
        test_value = 255
        encoded = encoder.encode(test_value)
        decoded = encoder.decode(encoded)
        assert decoded == test_value

