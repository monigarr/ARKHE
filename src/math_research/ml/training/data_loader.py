"""
Module: data_loader
Package: math_research.ml.training

Description:
    Data loading utilities for training transformer models on Collatz sequences.
    Provides dataset classes and data loading functions for generating training
    and test sets of Collatz long steps.

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
    >>> from math_research.ml.training.data_loader import CollatzDataset
    >>> from math_research.sequences import CollatzSequence
    >>> 
    >>> dataset = CollatzDataset(
    ...     start_range=(1, 10**6),
    ...     num_samples=10000,
    ...     base=24
    ... )
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

Dependencies:
    - torch>=2.0.0
    - math_research.sequences.collatz
    - math_research.ml.encoding.multi_base
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Dataset generates Collatz long step pairs on-the-fly for memory efficiency.
"""

import random
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
from math_research.sequences.collatz import CollatzSequence
from math_research.ml.encoding.multi_base import MultiBaseEncoder


class CollatzDataset(Dataset):
    """
    Dataset for Collatz long step prediction.
    
    Generates pairs of (input_odd_integer, long_step_result) encoded in
    the specified base.
    """
    
    def __init__(
        self,
        start_range: Tuple[int, int],
        num_samples: int,
        base: int = 24,
        max_length: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Initialize Collatz dataset.
        
        Args:
            start_range: Tuple (min, max) for generating odd starting values
            num_samples: Number of samples to generate
            base: Base for encoding (2-64)
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        self.start_range = start_range
        self.num_samples = num_samples
        self.base = base
        self.max_length = max_length
        self.encoder = MultiBaseEncoder(base=base, max_length=max_length)
        
        # Generate sample pairs
        self.samples: List[Tuple[int, int]] = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[int, int]]:
        """
        Generate training samples (input, output) pairs.
        
        Returns:
            List of (input_odd_int, long_step_result) tuples
        """
        samples = []
        collatz = CollatzSequence(start=1)
        
        min_val, max_val = self.start_range
        
        # Generate odd integers in range
        odd_numbers = [n for n in range(min_val, max_val + 1) if n % 2 == 1]
        
        for _ in range(self.num_samples):
            # Select random odd number
            if odd_numbers:
                input_val = random.choice(odd_numbers)
            else:
                # Fallback: generate odd number in range
                input_val = random.randrange(min_val | 1, max_val + 1, 2)
            
            # Compute long step
            step_info = collatz.compute_long_step(input_val)
            output_val = step_info['result']
            
            samples.append((input_val, output_val))
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_sequence, target_sequence) as tensors
        """
        input_val, target_val = self.samples[idx]
        
        # Encode to sequences
        input_seq = self.encoder.encode(input_val)
        target_seq = self.encoder.encode(target_val)
        
        # Pad sequences to max_length
        input_seq = self._pad_sequence(input_seq, self.max_length)
        target_seq = self._pad_sequence(target_seq, self.max_length)
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
    
    def _pad_sequence(self, sequence: List[int], max_length: int, pad_value: int = 0) -> List[int]:
        """
        Pad sequence to max_length.
        
        Args:
            sequence: Sequence to pad
            max_length: Target length
            pad_value: Padding value
            
        Returns:
            Padded sequence
        """
        if len(sequence) >= max_length:
            return sequence[:max_length]
        
        return sequence + [pad_value] * (max_length - len(sequence))

