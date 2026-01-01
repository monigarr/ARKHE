"""
Module: base_encoding
Package: math_research.ml.encoding

Description:
    Base class for integer encoding strategies. Provides abstract interface
    for encoding integers into sequences suitable for transformer models.

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
    >>> from math_research.ml.encoding import BaseEncoder
    >>> class MyEncoder(BaseEncoder):
    ...     def encode(self, n: int) -> List[int]:
    ...         return [int(d) for d in str(n)]
    >>> encoder = MyEncoder()
    >>> encoded = encoder.encode(12345)

Dependencies:
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This is an abstract base class. Subclasses must implement encode() and decode().
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEncoder(ABC):
    """
    Abstract base class for integer encoding strategies.
    
    Provides common interface for encoding integers into digit sequences
    and decoding sequences back to integers.
    """
    
    def __init__(self, max_length: int = 50):
        """
        Initialize encoder.
        
        Args:
            max_length: Maximum sequence length for encoded values
        """
        self.max_length = max_length
    
    @abstractmethod
    def encode(self, n: int) -> List[int]:
        """
        Encode an integer into a sequence of digits.
        
        Args:
            n: Integer to encode
            
        Returns:
            List of digit values (typically 0 to base-1)
        """
        pass
    
    @abstractmethod
    def decode(self, sequence: List[int]) -> int:
        """
        Decode a sequence of digits back to an integer.
        
        Args:
            sequence: List of digit values
            
        Returns:
            Decoded integer value
        """
        pass
    
    def encode_batch(self, numbers: List[int]) -> List[List[int]]:
        """
        Encode a batch of integers.
        
        Args:
            numbers: List of integers to encode
            
        Returns:
            List of encoded sequences
        """
        return [self.encode(n) for n in numbers]
    
    def decode_batch(self, sequences: List[List[int]]) -> List[int]:
        """
        Decode a batch of sequences.
        
        Args:
            sequences: List of encoded sequences
            
        Returns:
            List of decoded integers
        """
        return [self.decode(seq) for seq in sequences]

