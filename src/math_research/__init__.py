"""
Module: __init__
Package: math_research

Description:
    ARKHĒ Framework - Enterprise-grade Python framework for mathematical sequence
    research and machine learning experimentation. This package provides tools for exploring
    mathematical sequences (such as Collatz), performing statistical analysis, and training
    transformer models to understand sequence patterns.

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
    >>> from math_research.sequences import CollatzSequence
    >>> seq = CollatzSequence(start=27)
    >>> sequence = seq.generate(max_iterations=100)
    >>> print(sequence)

Dependencies:
    - numpy>=1.24.0
    - scipy>=1.10.0
    - pandas>=2.0.0

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This is the main package for the ARKHĒ Framework.
"""

__version__ = "0.1.0"
__author__ = "MoniGarr"
__email__ = "monigarr@MoniGarr.com"
__website__ = "MoniGarr.com"

from math_research.sequences.base import BaseSequence
from math_research.sequences.registry import SequenceRegistry
from math_research.sequences.collatz import CollatzSequence

__all__ = [
    "BaseSequence",
    "SequenceRegistry",
    "CollatzSequence",
]

