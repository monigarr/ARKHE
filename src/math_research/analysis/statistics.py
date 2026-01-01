"""
Module: statistics
Package: math_research.analysis

Description:
    Statistical analysis tools for mathematical sequences. Provides functions
    to compute various statistics including length, max value, peaks, cycles,
    and distribution properties.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Author Research Interests:
    - AI/ML Research and Development
    - Extended Reality (XR) Applications
    - 3D Graphics and Visualization
    - Robotics and Autonomous Systems
    - Computer Vision
    - Navigation Systems
    - Natural Language Processing (NLP)
    - Low Resource Languages (spoken in English communities)

Usage:
    >>> from math_research.analysis import SequenceStatistics
    >>> sequence = [27, 82, 41, 124, 62, 31, ...]
    >>> stats = SequenceStatistics(sequence)
    >>> print(stats.length())
    >>> print(stats.max_value())
    >>> summary = stats.summary()

Dependencies:
    - numpy>=1.24.0
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    All statistics are computed lazily and cached for performance.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


class SequenceStatistics:
    """
    Statistical analysis of mathematical sequences.
    
    Provides comprehensive statistical analysis including basic statistics,
    peak detection, cycle detection, and distribution properties.
    """
    
    def __init__(self, sequence: List[int]):
        """
        Initialize statistics calculator.
        
        Args:
            sequence: List of integers representing the sequence
        """
        if not sequence:
            raise ValueError("sequence cannot be empty")
        
        self.sequence = sequence
        self._array = np.array(sequence, dtype=np.int64)
        self._cache: Dict[str, any] = {}
    
    def length(self) -> int:
        """
        Get the length of the sequence.
        
        Returns:
            Number of elements in the sequence
        """
        return len(self.sequence)
    
    def max_value(self) -> int:
        """
        Get the maximum value in the sequence.
        
        Returns:
            Maximum value
        """
        if 'max_value' not in self._cache:
            self._cache['max_value'] = int(np.max(self._array))
        return self._cache['max_value']
    
    def min_value(self) -> int:
        """
        Get the minimum value in the sequence.
        
        Returns:
            Minimum value
        """
        if 'min_value' not in self._cache:
            self._cache['min_value'] = int(np.min(self._array))
        return self._cache['min_value']
    
    def mean(self) -> float:
        """
        Get the mean (average) value in the sequence.
        
        Returns:
            Mean value
        """
        if 'mean' not in self._cache:
            self._cache['mean'] = float(np.mean(self._array))
        return self._cache['mean']
    
    def median(self) -> float:
        """
        Get the median value in the sequence.
        
        Returns:
            Median value
        """
        if 'median' not in self._cache:
            self._cache['median'] = float(np.median(self._array))
        return self._cache['median']
    
    def std_dev(self) -> float:
        """
        Get the standard deviation of the sequence.
        
        Returns:
            Standard deviation
        """
        if 'std_dev' not in self._cache:
            self._cache['std_dev'] = float(np.std(self._array))
        return self._cache['std_dev']
    
    def peak_position(self) -> Tuple[int, int]:
        """
        Get the peak value and its position in the sequence.
        
        Returns:
            Tuple of (peak_value, position_index)
        """
        if 'peak' not in self._cache:
            peak_idx = np.argmax(self._array)
            self._cache['peak'] = (int(self._array[peak_idx]), int(peak_idx))
        return self._cache['peak']
    
    def find_peaks(self, min_height: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Find all peaks in the sequence.
        
        A peak is defined as a value greater than both neighbors.
        
        Args:
            min_height: Optional minimum height for a peak to be included
            
        Returns:
            List of tuples (value, index) for each peak
        """
        peaks = []
        
        for i in range(1, len(self.sequence) - 1):
            if (self.sequence[i] > self.sequence[i-1] and 
                self.sequence[i] > self.sequence[i+1]):
                if min_height is None or self.sequence[i] >= min_height:
                    peaks.append((self.sequence[i], i))
        
        return peaks
    
    def detect_cycle(self, max_cycle_length: int = 100) -> Optional[List[int]]:
        """
        Detect if the sequence contains a cycle.
        
        Args:
            max_cycle_length: Maximum cycle length to search for
            
        Returns:
            Cycle as a list of values if found, None otherwise
        """
        seq_len = len(self.sequence)
        if seq_len < 2:
            return None
        
        # Look for repeating patterns
        for cycle_len in range(1, min(max_cycle_length, seq_len // 2) + 1):
            # Check if last cycle_len elements repeat
            pattern = self.sequence[-cycle_len:]
            if seq_len >= cycle_len * 2:
                prev_pattern = self.sequence[-cycle_len*2:-cycle_len]
                if pattern == prev_pattern:
                    return pattern
        
        return None
    
    def value_counts(self) -> Dict[int, int]:
        """
        Count occurrences of each value in the sequence.
        
        Returns:
            Dictionary mapping value to count
        """
        unique, counts = np.unique(self._array, return_counts=True)
        return {int(val): int(count) for val, count in zip(unique, counts)}
    
    def summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of sequence statistics.
        
        Returns:
            Dictionary containing all computed statistics
        """
        peak_value, peak_idx = self.peak_position()
        
        return {
            'length': self.length(),
            'max_value': self.max_value(),
            'min_value': self.min_value(),
            'mean': self.mean(),
            'median': self.median(),
            'std_dev': self.std_dev(),
            'peak_value': peak_value,
            'peak_position': peak_idx,
            'num_peaks': len(self.find_peaks()),
            'has_cycle': self.detect_cycle() is not None,
            'cycle': self.detect_cycle(),
            'unique_values': len(self.value_counts()),
        }

