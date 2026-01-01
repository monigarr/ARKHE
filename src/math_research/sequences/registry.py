"""
Module: registry
Package: math_research.sequences

Description:
    Registry pattern implementation for sequence types. Allows dynamic registration
    and retrieval of sequence classes by name, enabling extensibility without
    modifying core framework code.

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
    >>> from math_research.sequences import SequenceRegistry, BaseSequence
    >>> 
    >>> class MySequence(BaseSequence):
    ...     def step(self, n: int) -> int:
    ...         return n * 2
    >>> 
    >>> SequenceRegistry.register("my_sequence", MySequence)
    >>> seq_class = SequenceRegistry.get("my_sequence")
    >>> seq = seq_class(start=5)

Dependencies:
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This registry uses a singleton pattern to maintain a global registry of sequence types.
"""

from typing import Dict, Type, Optional
from math_research.sequences.base import BaseSequence


class SequenceRegistry:
    """
    Registry for sequence types.
    
    This class provides a centralized registry for sequence classes, allowing
    dynamic registration and retrieval by name. This enables extensibility
    without modifying core framework code.
    """
    
    _registry: Dict[str, Type[BaseSequence]] = {}
    
    @classmethod
    def register(cls, name: str, sequence_class: Type[BaseSequence]) -> None:
        """
        Register a sequence class with the given name.
        
        Args:
            name: Unique identifier for the sequence type
            sequence_class: Class that inherits from BaseSequence
            
        Raises:
            ValueError: If sequence_class does not inherit from BaseSequence
            ValueError: If name is already registered
        """
        if not issubclass(sequence_class, BaseSequence):
            raise ValueError(
                f"sequence_class must inherit from BaseSequence, got {sequence_class}"
            )
        
        if name in cls._registry:
            raise ValueError(f"Sequence type '{name}' is already registered")
        
        cls._registry[name] = sequence_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseSequence]]:
        """
        Retrieve a sequence class by name.
        
        Args:
            name: Name of the registered sequence type
            
        Returns:
            Sequence class if found, None otherwise
        """
        return cls._registry.get(name)
    
    @classmethod
    def list_all(cls) -> list[str]:
        """
        Get a list of all registered sequence type names.
        
        Returns:
            List of registered sequence type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a sequence class.
        
        Args:
            name: Name of the sequence type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a sequence type is registered.
        
        Args:
            name: Name of the sequence type to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered sequence types."""
        cls._registry.clear()

