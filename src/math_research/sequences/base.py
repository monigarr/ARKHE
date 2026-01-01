"""
Module: base
Package: math_research.sequences

Description:
    Base class for mathematical sequences. Provides a common interface and shared
    functionality for all sequence types in the framework. Subclasses should implement
    the step() method to define sequence generation logic.

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
    >>> class MySequence(BaseSequence):
    ...     def step(self, n: int) -> int:
    ...         return n * 2 + 1
    >>> seq = MySequence(start=5)
    >>> sequence = seq.generate(max_iterations=10)

Dependencies:
    - numpy>=1.24.0
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This is an abstract base class. The step() method must be implemented by subclasses.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any
import numpy as np


class BaseSequence(ABC):
    """
    Abstract base class for mathematical sequences.
    
    This class provides a common interface for generating and analyzing mathematical
    sequences. Subclasses should implement the step() method to define the sequence
    generation rule.
    
    Attributes:
        start: The starting value of the sequence
        current: The current value in the sequence
        history: List of all values generated so far
        max_iterations: Maximum number of iterations allowed
        stop_condition: Optional callable that returns True when sequence should stop
    """
    
    def __init__(
        self,
        start: int,
        max_iterations: int = 1000000,
        stop_condition: Optional[Callable[[int, List[int]], bool]] = None,
    ):
        """
        Initialize a sequence generator.
        
        Args:
            start: Starting value for the sequence
            max_iterations: Maximum number of iterations before stopping
            stop_condition: Optional function(current_value, history) -> bool
                           Returns True when sequence should stop early
        """
        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer")
        
        self.start = start
        self.current = start
        self.history: List[int] = [start]
        self.max_iterations = max_iterations
        self.stop_condition = stop_condition
    
    @abstractmethod
    def step(self, n: int) -> int:
        """
        Compute the next value in the sequence.
        
        This method must be implemented by subclasses to define the sequence rule.
        
        Args:
            n: Current value in the sequence
            
        Returns:
            Next value in the sequence
        """
        pass
    
    def generate(
        self,
        max_iterations: Optional[int] = None,
        reset: bool = False,
    ) -> List[int]:
        """
        Generate the sequence up to max_iterations or until stop condition.
        
        Args:
            max_iterations: Override the instance max_iterations if provided
            reset: If True, reset to start value before generating
            
        Returns:
            List of sequence values generated
        """
        if reset:
            self.current = self.start
            self.history = [self.start]
        
        iterations = max_iterations if max_iterations is not None else self.max_iterations
        
        for _ in range(iterations):
            # Check stop condition
            if self.stop_condition and self.stop_condition(self.current, self.history):
                break
            
            # Compute next value
            next_value = self.step(self.current)
            
            # Update state
            self.current = next_value
            self.history.append(next_value)
        
        return self.history.copy()
    
    def get_current(self) -> int:
        """Get the current value in the sequence."""
        return self.current
    
    def get_history(self) -> List[int]:
        """Get the full history of generated values."""
        return self.history.copy()
    
    def reset(self) -> None:
        """Reset the sequence to the starting value."""
        self.current = self.start
        self.history = [self.start]
    
    def __iter__(self):
        """Make the sequence iterable."""
        self.reset()
        return self
    
    def __next__(self) -> int:
        """Get the next value in the sequence."""
        if len(self.history) > self.max_iterations:
            raise StopIteration
        
        if self.stop_condition and self.stop_condition(self.current, self.history):
            raise StopIteration
        
        next_value = self.step(self.current)
        self.current = next_value
        self.history.append(next_value)
        return next_value
    
    def __repr__(self) -> str:
        """String representation of the sequence."""
        return f"{self.__class__.__name__}(start={self.start}, current={self.current})"

