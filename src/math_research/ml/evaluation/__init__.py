"""
Module: __init__
Package: math_research.ml.evaluation

Description:
    Evaluation metrics and error analysis for transformer models.
    Provides accuracy, exact match, and Collatz-specific metrics.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

from math_research.ml.evaluation.metrics import (
    compute_accuracy,
    compute_exact_match,
    compute_collatz_metrics,
)
from math_research.ml.evaluation.error_analysis import ErrorAnalyzer

__all__ = [
    "compute_accuracy",
    "compute_exact_match",
    "compute_collatz_metrics",
    "ErrorAnalyzer",
]

