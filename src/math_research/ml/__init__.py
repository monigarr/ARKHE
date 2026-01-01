"""
Module: __init__
Package: math_research.ml

Description:
    Machine learning components for sequence prediction and analysis.
    Includes transformer models, training pipelines, evaluation tools, and
    encoding strategies for mathematical sequences.

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
    >>> from math_research.ml import CollatzTransformer, MultiBaseEncoder
    >>> from math_research.ml import Trainer, CollatzDataset
    >>> from math_research.ml import compute_accuracy, compute_collatz_metrics
    >>>
    >>> # Create encoder and model
    >>> encoder = MultiBaseEncoder(base=24)
    >>> model = CollatzTransformer(vocab_size=24, d_model=512)
    >>>
    >>> # Create dataset and trainer
    >>> dataset = CollatzDataset(start_range=(1, 1000), num_samples=1000, base=24)
    >>> trainer = Trainer(model, train_loader, val_loader)
    >>>
    >>> # Evaluate
    >>> accuracy = compute_accuracy(predictions, targets)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT
"""

# Models
from math_research.ml.models import CollatzTransformer

# Encoding
from math_research.ml.encoding import MultiBaseEncoder, BaseEncoder

# Training
from math_research.ml.training import Trainer, CollatzDataset, ExperimentTracker

# Evaluation
from math_research.ml.evaluation import (
    compute_accuracy,
    compute_exact_match,
    compute_collatz_metrics,
    ErrorAnalyzer,
)

__all__ = [
    # Models
    "CollatzTransformer",
    # Encoding
    "MultiBaseEncoder",
    "BaseEncoder",
    # Training
    "Trainer",
    "CollatzDataset",
    "ExperimentTracker",
    # Evaluation
    "compute_accuracy",
    "compute_exact_match",
    "compute_collatz_metrics",
    "ErrorAnalyzer",
]

