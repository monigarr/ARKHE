"""
Module: experiment_tracker
Package: math_research.ml.training

Description:
    Experiment tracking utilities for ML training. Provides integration
    with experiment tracking systems like wandb and MLflow.

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
    >>> from math_research.ml.training.experiment_tracker import ExperimentTracker
    >>> tracker = ExperimentTracker(backend="wandb", project_name="collatz")
    >>> tracker.log_metric("loss", 0.5, step=100)

Dependencies:
    - wandb>=0.15.0 (optional)
    - mlflow>=2.0.0 (optional)
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This module provides a unified interface for experiment tracking.
    Backends can be selected at runtime.
"""

from typing import Dict, Any, Optional
from math_research.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    
    Supports multiple backends (wandb, mlflow) with a common API.
    """
    
    def __init__(
        self,
        backend: str = "wandb",
        project_name: str = "math-research",
        experiment_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize experiment tracker.
        
        Args:
            backend: Tracking backend ('wandb', 'mlflow', or 'none')
            project_name: Project name for tracking
            experiment_name: Experiment name (auto-generated if None)
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend.lower()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.tracker = None
        
        if self.backend == "wandb":
            try:
                import wandb
                self.tracker = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    **kwargs
                )
                logger.info("Initialized wandb tracker")
            except ImportError:
                logger.warning("wandb not available, tracking disabled")
                self.backend = "none"
        
        elif self.backend == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(project_name)
                if experiment_name:
                    mlflow.set_tag("mlflow.runName", experiment_name)
                self.tracker = mlflow
                logger.info("Initialized mlflow tracker")
            except ImportError:
                logger.warning("mlflow not available, tracking disabled")
                self.backend = "none"
        
        elif self.backend == "none":
            logger.info("Experiment tracking disabled")
        
        else:
            logger.warning(f"Unknown backend '{backend}', tracking disabled")
            self.backend = "none"
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (optional)
        """
        if self.backend == "wandb" and self.tracker:
            self.tracker.log({name: value}, step=step)
        elif self.backend == "mlflow" and self.tracker:
            self.tracker.log_metric(name, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step number (optional)
        """
        if self.backend == "wandb" and self.tracker:
            self.tracker.log(metrics, step=step)
        elif self.backend == "mlflow" and self.tracker:
            for name, value in metrics.items():
                self.tracker.log_metric(name, value, step=step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameter names to values
        """
        if self.backend == "wandb" and self.tracker:
            self.tracker.config.update(params)
        elif self.backend == "mlflow" and self.tracker:
            self.tracker.log_params(params)
    
    def finish(self) -> None:
        """Finish tracking and close connections."""
        if self.backend == "wandb" and self.tracker:
            self.tracker.finish()
        elif self.backend == "mlflow" and self.tracker:
            # MLflow doesn't need explicit finish
            pass

