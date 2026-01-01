"""
Module: __init__
Package: math_research.utils

Description:
    Utility functions and classes for the ARKHĒ framework.
    Includes logging, configuration management, and validation utilities.

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
    >>> from math_research.utils import get_logger, load_config
    >>> logger = get_logger(__name__)
    >>> config = load_config("configs/default.yaml")

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

from math_research.utils.logging import get_logger, setup_logging
from math_research.utils.config import load_config, Config

__all__ = [
    "get_logger",
    "setup_logging",
    "load_config",
    "Config",
]

