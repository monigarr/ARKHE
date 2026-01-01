"""
Module: validators
Package: math_research.utils

Description:
    Validation utilities for input validation and type checking. Provides
    common validation functions used throughout the framework.

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
    >>> from math_research.utils.validators import validate_positive_int, validate_base
    >>> validate_positive_int(5)  # Returns 5
    >>> validate_base(24)  # Returns 24
    >>> validate_positive_int(-1)  # Raises ValueError

Dependencies:
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    All validation functions raise ValueError with descriptive messages on failure.
"""

from typing import Any


def validate_positive_int(value: Any, name: str = "value") -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If value is not a positive integer
        TypeError: If value cannot be converted to int
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from e
    
    if int_value <= 0:
        raise ValueError(f"{name} must be positive, got {int_value}")
    
    return int_value


def validate_non_negative_int(value: Any, name: str = "value") -> int:
    """
    Validate that a value is a non-negative integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If value is negative
        TypeError: If value cannot be converted to int
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from e
    
    if int_value < 0:
        raise ValueError(f"{name} must be non-negative, got {int_value}")
    
    return int_value


def validate_base(base: Any, min_base: int = 2, max_base: int = 64) -> int:
    """
    Validate that a value is a valid number base.
    
    Args:
        base: Base value to validate
        min_base: Minimum allowed base (default: 2)
        max_base: Maximum allowed base (default: 64)
        
    Returns:
        Validated base value
        
    Raises:
        ValueError: If base is outside valid range
        TypeError: If base cannot be converted to int
    """
    try:
        base_value = int(base)
    except (TypeError, ValueError) as e:
        raise TypeError(f"base must be an integer, got {type(base).__name__}") from e
    
    if base_value < min_base or base_value > max_base:
        raise ValueError(f"base must be between {min_base} and {max_base}, got {base_value}")
    
    return base_value


def validate_odd_int(value: Any, name: str = "value") -> int:
    """
    Validate that a value is an odd positive integer.
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        
    Returns:
        Validated odd integer value
        
    Raises:
        ValueError: If value is not odd or not positive
        TypeError: If value cannot be converted to int
    """
    int_value = validate_positive_int(value, name)
    
    if int_value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {int_value}")
    
    return int_value


def validate_range(
    value: Any,
    min_val: float = float('-inf'),
    max_val: float = float('inf'),
    name: str = "value",
) -> float:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter (for error messages)
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If value is outside the range
        TypeError: If value cannot be converted to float
    """
    try:
        float_value = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be a number, got {type(value).__name__}") from e
    
    if float_value < min_val or float_value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {float_value}")
    
    return float_value

