"""
Module: patterns
Package: math_research.analysis

Description:
    Pattern detection algorithms for mathematical sequences. Provides functions
    to identify repeating patterns, trends, and structural properties in sequences.

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
    >>> from math_research.analysis.patterns import PatternDetector
    >>> detector = PatternDetector(sequence)
    >>> patterns = detector.find_repeating_patterns()

Dependencies:
    - numpy>=1.24.0
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Pattern detection algorithms are designed to be extensible for different
    sequence types and pattern definitions.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


class PatternDetector:
    """
    Pattern detection for mathematical sequences.
    
    Provides various algorithms to detect repeating patterns, trends,
    and structural properties.
    """
    
    def __init__(self, sequence: List[int]):
        """
        Initialize pattern detector.
        
        Args:
            sequence: Sequence to analyze
        """
        if not sequence:
            raise ValueError("sequence cannot be empty")
        
        self.sequence = sequence
        self._array = np.array(sequence, dtype=np.int64)
    
    def find_repeating_patterns(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """
        Find repeating patterns in the sequence.
        
        Args:
            min_pattern_length: Minimum length of pattern to search
            max_pattern_length: Maximum length of pattern (default: len/4)
            
        Returns:
            List of dictionaries with pattern information
        """
        if max_pattern_length is None:
            max_pattern_length = len(self.sequence) // 4
        
        patterns = []
        seq_len = len(self.sequence)
        
        for pattern_len in range(min_pattern_length, min(max_pattern_length, seq_len // 2) + 1):
            # Check for pattern at the end
            pattern = tuple(self.sequence[-pattern_len:])
            
            # Count occurrences
            count = 0
            for i in range(seq_len - pattern_len + 1):
                candidate = tuple(self.sequence[i:i+pattern_len])
                if candidate == pattern:
                    count += 1
            
            if count >= 2:  # Pattern appears at least twice
                patterns.append({
                    'pattern': pattern,
                    'length': pattern_len,
                    'occurrences': count,
                    'first_position': self.sequence.index(pattern[0]),
                    'last_position': seq_len - pattern_len,
                })
        
        # Sort by length (longest first), then by occurrences
        patterns.sort(key=lambda x: (x['length'], x['occurrences']), reverse=True)
        return patterns
    
    def detect_binary_patterns(self, max_length: int = 20) -> Dict[str, List[int]]:
        """
        Detect patterns in binary representation (for Collatz sequences).
        
        Groups sequence values by their binary suffix patterns.
        
        Args:
            max_length: Maximum binary suffix length to analyze
            
        Returns:
            Dictionary mapping binary suffix to list of values with that suffix
        """
        patterns: Dict[str, List[int]] = {}
        
        for value in self.sequence:
            binary = bin(value)[2:]  # Remove '0b' prefix
            
            for length in range(1, min(max_length, len(binary)) + 1):
                suffix = binary[-length:]
                if suffix not in patterns:
                    patterns[suffix] = []
                patterns[suffix].append(value)
        
        return patterns
    
    def find_trends(self, window_size: int = 10) -> List[str]:
        """
        Detect trends (increasing, decreasing, stable) in sequence.
        
        Args:
            window_size: Size of sliding window for trend detection
            
        Returns:
            List of trend labels for each window
        """
        if len(self.sequence) < window_size:
            return []
        
        trends = []
        
        for i in range(len(self.sequence) - window_size + 1):
            window = self._array[i:i+window_size]
            
            # Compute linear trend
            x = np.arange(len(window))
            coeffs = np.polyfit(x, window, 1)
            slope = coeffs[0]
            
            if slope > 0.1:
                trends.append("increasing")
            elif slope < -0.1:
                trends.append("decreasing")
            else:
                trends.append("stable")
        
        return trends
    
    def detect_periodicity(self, max_period: Optional[int] = None) -> Optional[int]:
        """
        Detect if sequence has a periodic component.
        
        Args:
            max_period: Maximum period to search for (default: len/4)
            
        Returns:
            Period if found, None otherwise
        """
        if max_period is None:
            max_period = len(self.sequence) // 4
        
        seq_len = len(self.sequence)
        
        for period in range(1, min(max_period, seq_len // 2) + 1):
            # Check if sequence repeats with this period
            matches = True
            for i in range(period, seq_len - period):
                if self.sequence[i] != self.sequence[i + period]:
                    matches = False
                    break
            
            if matches:
                return period
        
        return None

