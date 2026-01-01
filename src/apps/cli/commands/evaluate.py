"""
Evaluate command - Evaluate trained models.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from math_research.ml import (
    CollatzTransformer,
    CollatzDataset,
    compute_accuracy,
    compute_exact_match,
    compute_collatz_metrics,
)
from math_research.utils import get_logger

logger = get_logger(__name__)


def add_arguments(parser):
    """Add arguments for the evaluate command."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    
    parser.add_argument(
        "--test-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=[10000, 20000],
        help="Range of input values for test data (default: 10000 20000)",
    )
    
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of test samples (default: 1000)",
    )
    
    parser.add_argument(
        "--base",
        type=int,
        default=24,
        help="Base for encoding (must match training base, default: 24)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for evaluation results (JSON). If not specified, prints to stdout",
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for evaluation (default: auto)",
    )


def execute(args) -> int:
    """Execute the evaluate command."""
    try:
        # Load checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
            return 1
        
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Determine device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        logger.info(f"Using device: {device}")
        
        # Reconstruct model (need to know architecture)
        # For now, we'll use default architecture or try to infer from checkpoint
        vocab_size = args.base  # Assume base is vocab_size
        if "model_config" in checkpoint:
            config = checkpoint["model_config"]
            vocab_size = config.get("vocab_size", vocab_size)
            d_model = config.get("d_model", 512)
            nhead = config.get("nhead", 8)
            num_layers = config.get("num_layers", 6)
        else:
            # Use defaults
            d_model = 512
            nhead = 8
            num_layers = 6
        
        model = CollatzTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        
        # Create test dataset
        logger.info("Creating test dataset...")
        test_dataset = CollatzDataset(
            start_range=tuple(args.test_range),
            num_samples=args.test_size,
            base=args.base,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        # Evaluate
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Get predictions
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_inputs.append(inputs.cpu())
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        inputs = torch.cat(all_inputs, dim=0)
        
        # Compute metrics
        logger.info("Computing metrics...")
        accuracy = compute_accuracy(predictions, targets)
        exact_match = compute_exact_match(predictions, targets)
        
        # Get input values for Collatz-specific metrics
        input_values = []
        for input_seq in inputs:
            # Decode first non-padding value (simplified - assumes single number)
            # This is a simplified approach
            input_values.append(input_seq[0].item() if input_seq[0].item() > 0 else None)
        
        collatz_metrics = compute_collatz_metrics(
            predictions,
            targets,
            input_values,
            base=args.base,
        )
        
        # Prepare results
        results = {
            "checkpoint": str(args.checkpoint),
            "test_samples": len(test_dataset),
            "test_range": args.test_range,
            "metrics": {
                "accuracy": float(accuracy),
                "exact_match": float(exact_match),
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in collatz_metrics.items()},
            },
        }
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2))
        
        logger.info(f"Evaluation complete: Accuracy={accuracy:.4f}, Exact Match={exact_match:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

