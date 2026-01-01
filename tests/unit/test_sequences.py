"""
Unit tests for sequence modules.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com
"""

import pytest
from math_research.sequences import CollatzSequence, BaseSequence, SequenceRegistry


def test_collatz_sequence_basic():
    """Test basic Collatz sequence generation."""
    seq = CollatzSequence(start=27)
    sequence = seq.generate(max_iterations=100)
    
    assert len(sequence) > 0
    assert sequence[0] == 27
    assert sequence[-1] == 1


def test_collatz_long_step():
    """Test Collatz long step computation."""
    seq = CollatzSequence(start=1)
    step_info = seq.compute_long_step(27)
    
    assert 'result' in step_info
    assert 'k' in step_info
    assert 'k_prime' in step_info
    assert step_info['k'] >= 0
    assert step_info['k_prime'] >= 0


def test_sequence_registry():
    """Test sequence registry."""
    # Clear registry for clean test
    SequenceRegistry.clear()
    
    # Register a test sequence class
    class TestSequence(BaseSequence):
        def step(self, n: int) -> int:
            return n + 1
    
    SequenceRegistry.register("test", TestSequence)
    
    # Retrieve registered sequence
    retrieved = SequenceRegistry.get("test")
    assert retrieved == TestSequence
    
    # Test listing
    assert "test" in SequenceRegistry.list_all()
    
    # Cleanup
    SequenceRegistry.unregister("test")

