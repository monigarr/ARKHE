"""
Module: base_model
Package: math_research.ml.models

Description:
    Base classes for transformer models used in mathematical sequence prediction.
    Provides common architecture patterns and interfaces.

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
    This module provides base classes that should be inherited by specific
    model implementations. See CollatzTransformer for an example.

Dependencies:
    - torch>=2.0.0
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    This is a base class module. Specific model implementations should
    inherit from these classes.
"""

from typing import Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSequenceModel(nn.Module, ABC):
    """
    Abstract base class for sequence prediction models.
    
    Provides common interface and utilities for transformer-based
    sequence prediction models.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 128,
    ):
        """
        Initialize base sequence model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
    
    @abstractmethod
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            tgt: Target sequence tensor [batch_size, tgt_len] (optional)
            
        Returns:
            Model output tensor
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

