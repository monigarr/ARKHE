"""
Analyze command - Analyze sequence patterns and statistics.

Author: MoniGarr
Email: monigarr@MoniGarr.com
Website: MoniGarr.com
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from math_research.sequences import CollatzSequence
from math_research.analysis import SequenceStatistics
from math_research.utils import get_logger

logger = get_logger(__name__)


def add_arguments(parser):
    """Add arguments for the analyze command."""
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting value for analysis range",
    )
    
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Ending value for analysis range",
    )
    
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for range (default: 1)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (JSON). If not specified, prints to stdout",
    )
    
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        default=True,
        help="Compute statistical summaries (default: True)",
    )
    
    parser.add_argument(
        "--compute-patterns",
        action="store_true",
        help="Detect patterns in sequences",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000000,
        help="Maximum iterations per sequence (default: 1000000)",
    )


def execute(args) -> int:
    """Execute the analyze command."""
    try:
        # Validate range
        if args.start <= 0 or args.end <= 0:
            print(f"Error: Start and end values must be positive", file=sys.stderr)
            return 1
        
        if args.start > args.end:
            print(f"Error: Start value ({args.start}) must be <= end value ({args.end})", file=sys.stderr)
            return 1
        
        # Generate sequences for analysis
        logger.info(f"Analyzing Collatz sequences from {args.start} to {args.end}...")
        
        all_sequences = []
        all_lengths = []
        all_max_values = []
        
        values_to_analyze = range(args.start, args.end + 1, args.step)
        total = len(list(values_to_analyze))
        
        for i, start_val in enumerate(values_to_analyze, 1):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{total}...")
            
            seq = CollatzSequence(
                start=start_val,
                max_iterations=args.max_iterations,
            )
            sequence = seq.generate()
            
            all_sequences.append({
                "start": start_val,
                "sequence": sequence,
                "length": len(sequence),
                "max_value": seq.get_max_value(),
                "peak_value": seq.get_peak_value()[0],
                "peak_position": seq.get_peak_value()[1],
            })
            
            all_lengths.append(len(sequence))
            all_max_values.append(seq.get_max_value())
        
        logger.info(f"Analyzed {len(all_sequences)} sequences")
        
        # Compute statistics if requested
        results = {
            "range": {
                "start": args.start,
                "end": args.end,
                "step": args.step,
                "count": len(all_sequences),
            },
            "sequences": all_sequences,
        }
        
        if args.compute_stats:
            logger.info("Computing statistics...")
            stats = {
                "length": {
                    "min": min(all_lengths),
                    "max": max(all_lengths),
                    "mean": sum(all_lengths) / len(all_lengths),
                    "median": sorted(all_lengths)[len(all_lengths) // 2],
                },
                "max_value": {
                    "min": min(all_max_values),
                    "max": max(all_max_values),
                    "mean": sum(all_max_values) / len(all_max_values),
                    "median": sorted(all_max_values)[len(all_max_values) // 2],
                },
            }
            results["statistics"] = stats
        
        # Pattern detection if requested
        if args.compute_patterns:
            logger.info("Detecting patterns...")
            # Basic pattern detection - could be expanded
            results["patterns"] = {
                "note": "Pattern detection not yet implemented",
            }
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Analysis results saved to {output_path}")
            print(f"Analyzed {len(all_sequences)} sequences")
            if args.compute_stats and "statistics" in results:
                stats = results["statistics"]
                print(f"\nSequence Length: min={stats['length']['min']}, max={stats['length']['max']}, mean={stats['length']['mean']:.2f}")
                print(f"Max Value: min={stats['max_value']['min']}, max={stats['max_value']['max']}, mean={stats['max_value']['mean']:.2f}")
        else:
            # Print summary to stdout
            print(json.dumps(results, indent=2))
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

