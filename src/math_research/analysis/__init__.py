"""
Module: __init__
Package: math_research.analysis

Description:
    Analysis tools for mathematical sequences. Provides statistical analysis,
    visualization capabilities, and pattern detection algorithms.

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
    >>> from math_research.analysis import SequenceStatistics, SequenceVisualizer
    >>> stats = SequenceStatistics(sequence)
    >>> print(stats.summary())

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

from math_research.analysis.statistics import SequenceStatistics
from math_research.analysis.visualization import SequenceVisualizer

__all__ = [
    "SequenceStatistics",
    "SequenceVisualizer",
]

