"""
Module: __init__
Package: math_research.ml.training

Description:
    Training pipelines and utilities for transformer models.
    Includes trainer classes, data loaders, and experiment tracking.

Author: MoniGarr
Author Email: monigarr@MoniGarr.com
Author Website: MoniGarr.com

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

from math_research.ml.training.trainer import Trainer
from math_research.ml.training.data_loader import CollatzDataset
from math_research.ml.training.experiment_tracker import ExperimentTracker

__all__ = [
    "Trainer",
    "CollatzDataset",
    "ExperimentTracker",
]

