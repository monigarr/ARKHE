"""
Unit tests for analysis modules.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com
"""

import pytest
from math_research.analysis import SequenceStatistics, SequenceVisualizer


def test_sequence_statistics():
    """Test sequence statistics computation."""
    sequence = [27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214, 107, 322, 161]
    stats = SequenceStatistics(sequence)
    
    assert stats.length() == len(sequence)
    assert stats.max_value() == max(sequence)
    assert stats.min_value() == min(sequence)
    assert stats.mean() > 0


def test_sequence_visualizer():
    """Test sequence visualizer creation."""
    visualizer = SequenceVisualizer()
    assert visualizer is not None


def test_statistics_summary():
    """Test statistics summary generation."""
    sequence = [27, 82, 41, 124, 62, 31]
    stats = SequenceStatistics(sequence)
    summary = stats.summary()
    
    assert 'length' in summary
    assert 'max_value' in summary
    assert 'min_value' in summary
    assert 'mean' in summary

