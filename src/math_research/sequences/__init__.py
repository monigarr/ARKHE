"""
Module: __init__
Package: math_research.sequences

Description:
    Sequence generation and analysis module. Provides base classes and implementations
    for mathematical sequences including Collatz sequences and extensible framework
    for custom sequence types.

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
    >>> from math_research.sequences import BaseSequence, CollatzSequence
    >>> seq = CollatzSequence(start=27)
    >>> sequence = seq.generate()

Dependencies:
    - numpy>=1.24.0

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    All sequence classes should inherit from BaseSequence for consistency.
"""

from math_research.sequences.base import BaseSequence
from math_research.sequences.registry import SequenceRegistry
from math_research.sequences.collatz import CollatzSequence

__all__ = [
    "BaseSequence",
    "SequenceRegistry",
    "CollatzSequence",
]

