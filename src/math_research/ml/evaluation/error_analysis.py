"""
Module: error_analysis
Package: math_research.ml.evaluation

Description:
    Error analysis utilities for understanding model failures.
    Provides tools to analyze prediction errors and identify patterns.

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
    >>> from math_research.ml.evaluation.error_analysis import ErrorAnalyzer
    >>> analyzer = ErrorAnalyzer()
    >>> error_patterns = analyzer.analyze_errors(predictions, targets, inputs)

Dependencies:
    - numpy>=1.24.0
    - typing (standard library)

Version: 0.1.0
Last Modified: 2025-01-09
License: MIT

Notes:
    Error analysis helps identify systematic failures and model limitations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from math_research.ml.encoding.multi_base import MultiBaseEncoder


class ErrorAnalyzer:
    """
    Error analysis for model predictions.
    
    Provides tools to analyze prediction errors and identify patterns
    in model failures.
    """
    
    def __init__(self, base: int = 24):
        """
        Initialize error analyzer.
        
        Args:
            base: Base used for encoding
        """
        self.base = base
        self.encoder = MultiBaseEncoder(base=base)
    
    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors and identify patterns.
        
        Args:
            predictions: Predicted sequences [batch_size, seq_len]
            targets: Target sequences [batch_size, seq_len]
            inputs: Input values [batch_size]
            
        Returns:
            Dictionary with error analysis results
        """
        errors = []
        correct = []
        
        for i in range(len(predictions)):
            pred_seq = predictions[i].tolist()
            target_seq = targets[i].tolist()
            
            # Remove padding (0)
            pred_seq = [d for d in pred_seq if d != 0]
            target_seq = [d for d in target_seq if d != 0]
            
            try:
                pred_value = self.encoder.decode(pred_seq)
                target_value = self.encoder.decode(target_seq)
                
                if pred_value != target_value:
                    errors.append({
                        'input': int(inputs[i]),
                        'predicted': pred_value,
                        'target': target_value,
                        'error': abs(pred_value - target_value),
                    })
                else:
                    correct.append(int(inputs[i]))
            except (ValueError, IndexError):
                continue
        
        return {
            'num_errors': len(errors),
            'num_correct': len(correct),
            'error_rate': len(errors) / (len(errors) + len(correct)) if (len(errors) + len(correct)) > 0 else 0.0,
            'errors': errors[:100],  # Limit to first 100 for summary
            'correct_examples': correct[:100],
        }

