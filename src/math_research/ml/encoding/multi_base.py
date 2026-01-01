"""
Module: multi_base
Package: math_research.ml.encoding

Description:
    Multi-base encoding for integers. Converts integers to sequences of digits
    in a specified base (2-57 as used in Collatz transformer research).
    Optimized for transformer training on mathematical sequences.

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
    >>> from math_research.ml.encoding import MultiBaseEncoder
    >>> encoder = MultiBaseEncoder(base=24)
    >>> encoded = encoder.encode(12345)  # Returns list of digits in base 24
    >>> decoded = encoder.decode(encoded)  # Returns 12345

Dependencies:
    - math_research.ml.encoding.base_encoding
    - math_research.utils.validators

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Based on research showing optimal performance with bases 24 and 32 for
    Collatz sequence prediction. Supports bases 2-64 by default.
"""

from typing import List
from math_research.ml.encoding.base_encoding import BaseEncoder
from math_research.utils.validators import validate_base, validate_positive_int


class MultiBaseEncoder(BaseEncoder):
    """
    Multi-base encoder for converting integers to digit sequences.
    
    Converts integers to sequences of digits in a specified base, suitable
    for transformer model input/output.
    """
    
    def __init__(self, base: int = 10, max_length: int = 50):
        """
        Initialize multi-base encoder.
        
        Args:
            base: Base for encoding (2-64)
            max_length: Maximum sequence length
        """
        super().__init__(max_length=max_length)
        self.base = validate_base(base, min_base=2, max_base=64)
        self._digit_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"
    
    def encode(self, n: int) -> List[int]:
        """
        Encode an integer to a list of digits in the specified base.
        
        Args:
            n: Integer to encode (must be non-negative)
            
        Returns:
            List of digits (most significant first)
        """
        n = validate_positive_int(n, "n")
        
        if n == 0:
            return [0]
        
        digits = []
        num = n
        
        while num > 0 and len(digits) < self.max_length:
            digits.append(num % self.base)
            num //= self.base
        
        # Reverse to get most significant digit first
        digits.reverse()
        
        return digits
    
    def decode(self, sequence: List[int]) -> int:
        """
        Decode a sequence of digits back to an integer.
        
        Args:
            sequence: List of digits in the specified base
            
        Returns:
            Decoded integer value
        """
        if not sequence:
            return 0
        
        # Validate digits are in valid range
        for digit in sequence:
            if digit < 0 or digit >= self.base:
                raise ValueError(
                    f"Digit {digit} is invalid for base {self.base} "
                    f"(must be 0-{self.base-1})"
                )
        
        result = 0
        for digit in sequence:
            result = result * self.base + digit
        
        return result
    
    def encode_string(self, n: int) -> str:
        """
        Encode an integer to a string representation in the specified base.
        
        Args:
            n: Integer to encode
            
        Returns:
            String representation of the number in the specified base
        """
        digits = self.encode(n)
        return ''.join(self._digit_chars[d] for d in digits)
    
    def decode_string(self, s: str) -> int:
        """
        Decode a string representation back to an integer.
        
        Args:
            s: String representation in the specified base
            
        Returns:
            Decoded integer value
        """
        digit_map = {char: i for i, char in enumerate(self._digit_chars[:self.base])}
        sequence = [digit_map[char] for char in s]
        return self.decode(sequence)

