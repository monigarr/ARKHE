"""
Module: __init__
Package: math_research.ml.encoding

Description:
    Encoding strategies for representing integers in various bases.
    Supports multi-base encoding for transformer training on mathematical sequences.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

from math_research.ml.encoding.multi_base import MultiBaseEncoder
from math_research.ml.encoding.base_encoding import BaseEncoder

__all__ = [
    "MultiBaseEncoder",
    "BaseEncoder",
]

